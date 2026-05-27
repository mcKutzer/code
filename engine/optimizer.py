"""Markowitz mean-variance optimizer.

Reads inputs from data/inputs.xlsx:
    Returns sheet      -> per-asset mu and sigma
                          (override columns take precedence over calc)
    Covariance_* sheet -> EUR covariance / correlation matrix

Reads constraints from data/constraints.json:
    asset_caps    -> per-asset upper bounds
    asset_floors  -> per-asset lower bounds
    groups        -> named asset-class groups with combined floor + cap
                     (e.g. Equities >= 25%, Hard Assets <= 15%)

Builds the efficient frontier (200 target returns), locates the max-Sharpe
and min-variance portfolios, and runs a crisis-correlation stress test.
Writes data/frontier.json for the Streamlit app.

Constraints applied at every point:
    1. sum(w) = 1
    2. w'mu = target            (frontier sweep only)
    3. asset_floor <= w_i <= asset_cap
    4. group_floor <= sum_{i in g} w_i <= group_cap

SLSQP solves the quadratic Lagrangian with the inequality bounds handled
implicitly via KKT conditions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT: Path = Path(__file__).resolve().parent.parent
INPUTS_PATH: Path = ROOT / "data" / "inputs.xlsx"
OUT_PATH: Path = ROOT / "data" / "frontier.json"
CONSTRAINTS_PATH: Path = ROOT / "data" / "constraints.json"

N_POINTS: int = 200
MAX_WEIGHT: float = 0.60
RF_ASSET: str = "Cash equivalent"  # this asset's mu is used as the rf rate

# Covariance method: "ewma" (default, RiskMetrics half-life=36mo) or "sample".
# Switches which precomputed covariance sheet of inputs.xlsx is consumed.
# mu (Returns sheet) is unaffected by this choice.
COV_METHOD: str = "ewma"

# ---------- Bootstrap defaults ------------------------------------------------
# Only written to data/constraints.json when that file is missing. After first
# run, constraints.json is the source of truth — edit it via the app's
# "Constraints" tab, or directly on disk.

DEFAULT_ASSET_CAPS: dict[str, float] = {
    "VUAA.DE":   0.50,  # S&P 500
    "EXSA.DE":   0.50,  # Europe equity
    "IS3N.DE":   0.20,  # Emerging markets
    "CSBGE3.MI": 0.25,  # Short gov bonds
    "SXRP.DE":   0.25,  # Med gov bonds
    "IEAA.L":    0.20,  # Corp bonds IG
    "IGBY.AS":   0.15,  # Long gov bonds
    "XAD5.MI":   0.10,  # Gold
    "IQQ6.DE":   0.10,  # Global REITs
    "XEON.DE":   0.20,  # Cash equivalent
    "IHYG.L":    0.10,  # High Yield Bonds EUR
}

DEFAULT_ASSET_FLOORS: dict[str, float] = {t: 0.00 for t in DEFAULT_ASSET_CAPS}

# Five constraint groups. Equities' 25% floor replaces the old
# EQUITY_MIN_WEIGHT — same hard floor, just expressed as a group constraint.
DEFAULT_GROUPS: dict[str, dict] = {
    "Equities": {
        "tickers": ["VUAA.DE", "EXSA.DE", "IS3N.DE"],
        "floor": 0.25,
        "cap":   1.00,
    },
    "Government Bonds": {
        "tickers": ["CSBGE3.MI", "SXRP.DE", "IGBY.AS"],
        "floor": 0.00,
        "cap":   1.00,
    },
    "Corp + HY": {
        "tickers": ["IEAA.L", "IHYG.L"],
        "floor": 0.00,
        "cap":   1.00,
    },
    "Hard Assets": {
        "tickers": ["XAD5.MI", "IQQ6.DE"],
        "floor": 0.00,
        "cap":   1.00,
    },
    "Cash": {
        "tickers": ["XEON.DE"],
        "floor": 0.00,
        "cap":   1.00,
    },
}

# Crisis stress-test buckets (independent of the constraint groups). These
# only steer which off-diagonal correlations get overwritten in the 2008-style
# regime; they do NOT affect optimization.
EQUITY_CLASSES: set[str] = {
    "S&P 500", "Europe equity", "Emerging markets", "Global REITs",
    "High Yield Bonds EUR",
}
BOND_CLASSES: set[str] = {"Short gov bonds", "Med gov bonds", "Corp bonds IG", "Long gov bonds"}
GOLD_CLASSES: set[str] = {"Gold"}

# Crisis correlation overrides (2008-style: flight to quality).
# Bond-heavy portfolios can show crisis_vol < normal_vol because bonds
# diversify better in a deflationary crash than in normal times — this is
# correct behavior, not a bug.
CRISIS_EQ_EQ: float = 0.85
CRISIS_EQ_BD: float = -0.45   # flight to quality — bonds rally as equities fall
CRISIS_EQ_GD: float = 0.30    # gold sold off for liquidity


# ---------- Constraints I/O ---------------------------------------------------


def load_constraints() -> dict:
    """Return the active constraints dict. Writes defaults on first call."""
    if CONSTRAINTS_PATH.exists():
        cfg = json.loads(CONSTRAINTS_PATH.read_text())
        cfg.setdefault("asset_caps", dict(DEFAULT_ASSET_CAPS))
        cfg.setdefault("asset_floors", dict(DEFAULT_ASSET_FLOORS))
        cfg.setdefault("groups", {k: dict(v) for k, v in DEFAULT_GROUPS.items()})
        return cfg
    cfg = {
        "asset_caps": dict(DEFAULT_ASSET_CAPS),
        "asset_floors": dict(DEFAULT_ASSET_FLOORS),
        "groups": {k: dict(v) for k, v in DEFAULT_GROUPS.items()},
    }
    save_constraints(cfg)
    return cfg


def save_constraints(cfg: dict) -> None:
    CONSTRAINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONSTRAINTS_PATH.write_text(json.dumps(cfg, indent=2))


# ---------- I/O ---------------------------------------------------------------


COV_SHEET: dict[str, str] = {
    "sample": "Covariance_Sample",
    "ewma": "Covariance_EWMA",
}


def load_inputs(
    path: Path = INPUTS_PATH, cov_method: str = COV_METHOD
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Returns (assets, tickers, mu, cov, corr) from inputs.xlsx.

    mu:    override -> historical-calc fallback (Returns sheet)
    sigma: override -> sigma_implied fallback   (Returns sheet override; otherwise sqrt(diag(Sigma_chosen)))
    corr:  derived from the chosen Covariance_* sheet (cov_method only controls correlation structure)
    cov:   rebuilt as diag(sigma) @ corr @ diag(sigma) — honours sigma overrides under any cov_method
    """
    if not path.exists():
        print(f"[optimizer] ERROR: {path} not found. Run data/fetch_prices.py first.")
        sys.exit(1)
    if cov_method not in COV_SHEET:
        print(f"[optimizer] ERROR: cov_method='{cov_method}' not in {list(COV_SHEET)}")
        sys.exit(1)

    returns_df = pd.read_excel(path, sheet_name="Returns")
    cov_df = pd.read_excel(path, sheet_name=COV_SHEET[cov_method], index_col=0)

    assets = returns_df["Asset class"].astype(str).tolist()
    tickers = returns_df["Ticker"].astype(str).tolist()

    # ---- mu: override -> calc fallback ----
    mu_override_raw = pd.to_numeric(returns_df["Ann Return (override)"], errors="coerce")
    mu_calc_raw = pd.to_numeric(returns_df["Ann Return (calc)"], errors="coerce")
    mu = mu_override_raw.where(mu_override_raw.notna(), mu_calc_raw).to_numpy(dtype=float)
    mu_source = ["override" if pd.notna(v) else "historical" for v in mu_override_raw]

    if np.isnan(mu).any():
        missing = [a for a, m in zip(assets, mu) if np.isnan(m)]
        print(f"[optimizer] ERROR: missing mu for: {missing}")
        sys.exit(1)

    # ---- correlation: derive from the chosen covariance sheet ----
    if list(cov_df.index) != assets or list(cov_df.columns) != assets:
        cov_df = cov_df.reindex(index=assets, columns=assets)
    if cov_df.isna().any().any():
        print(f"[optimizer] ERROR: covariance matrix '{COV_SHEET[cov_method]}' has NaN entries.")
        print(cov_df.isna().sum())
        sys.exit(1)
    cov_implied = 0.5 * (cov_df.to_numpy() + cov_df.to_numpy().T)
    sigma_implied = np.sqrt(np.diag(cov_implied))
    inv = np.where(sigma_implied > 0, 1.0 / sigma_implied, 0.0)
    corr = cov_implied * np.outer(inv, inv)
    np.fill_diagonal(corr, 1.0)

    # ---- sigma: override -> sigma_implied fallback ----
    sigma_override_raw = pd.to_numeric(returns_df["Ann Vol (override)"], errors="coerce")
    sigma = sigma_override_raw.where(
        sigma_override_raw.notna(), pd.Series(sigma_implied, index=sigma_override_raw.index)
    ).to_numpy(dtype=float)
    sigma_source = [
        "override" if pd.notna(v) else f"implied ({cov_method})"
        for v in sigma_override_raw
    ]

    # ---- rebuild final cov from resolved sigma + chosen correlation ----
    cov = np.outer(sigma, sigma) * corr

    # Stash provenance for main() printouts
    load_inputs.last_mu_source = mu_source
    load_inputs.last_mu_calc = mu_calc_raw.to_numpy(dtype=float)
    load_inputs.last_mu_override = mu_override_raw.to_numpy(dtype=float)
    load_inputs.last_sigma_source = sigma_source
    load_inputs.last_sigma_implied = sigma_implied
    load_inputs.last_sigma_override = sigma_override_raw.to_numpy(dtype=float)

    return assets, tickers, mu, cov, corr


def covariance(sigma: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Sigma_cov = diag(sigma) C diag(sigma)."""
    return np.outer(sigma, sigma) * corr


# ---------- Bounds and group constraints --------------------------------------


def _effective_cap(t: str, asset_caps: dict[str, float]) -> float:
    """Missing -> MAX_WEIGHT. 0.0 means weight forced to 0 (explicit)."""
    return asset_caps.get(t, MAX_WEIGHT)


def _effective_floor(t: str, asset_floors: dict[str, float]) -> float:
    return asset_floors.get(t, 0.0)


def _bounds(tickers: list[str], cfg: dict) -> list[tuple[float, float]]:
    caps = cfg["asset_caps"]
    floors = cfg["asset_floors"]
    return [(_effective_floor(t, floors), _effective_cap(t, caps)) for t in tickers]


def _group_constraints(tickers: list[str], groups: dict) -> list[dict]:
    """SLSQP inequality constraints for each named group's combined weight.
    Only emit ineqs that actually bind something (floor>0 or cap<1)."""
    cs: list[dict] = []
    for _name, g in groups.items():
        idx = [tickers.index(t) for t in g.get("tickers", []) if t in tickers]
        if not idx:
            continue
        floor = float(g.get("floor", 0.0))
        cap = float(g.get("cap", 1.0))
        if floor > 0.0:
            cs.append({
                "type": "ineq",
                "fun": lambda w, i=idx, f=floor: float(np.sum(w[i])) - f,
            })
        if cap < 1.0:
            cs.append({
                "type": "ineq",
                "fun": lambda w, i=idx, c=cap: c - float(np.sum(w[i])),
            })
    return cs


def _x0(tickers: list[str], cfg: dict) -> np.ndarray:
    """Feasible-ish start: per-asset floors, group floors topped up uniformly
    across group members, then any remaining mass spread equally across all
    assets. Final clip+renormalize covers small cap violations. Equal-spread
    keeps the start near the centre of the simplex, which is the most robust
    initial point for low-return SLSQP targets."""
    n = len(tickers)
    floors_d = cfg["asset_floors"]
    caps_d = cfg["asset_caps"]
    groups = cfg["groups"]

    x = np.array([floors_d.get(t, 0.0) for t in tickers], dtype=float)
    cap_arr = np.array([caps_d.get(t, MAX_WEIGHT) for t in tickers], dtype=float)

    for _name, g in groups.items():
        idx = [tickers.index(t) for t in g.get("tickers", []) if t in tickers]
        if not idx:
            continue
        gf = float(g.get("floor", 0.0))
        cur = float(x[idx].sum())
        if gf > cur:
            x[idx] += (gf - cur) / len(idx)

    remaining = 1.0 - x.sum()
    if remaining > 1e-12:
        x += remaining / n

    x = np.minimum(x, cap_arr)
    s = x.sum()
    if s > 0:
        x = x / s
    return x


# ---------- Core optimizers ---------------------------------------------------


def min_variance_for_target(
    mu: np.ndarray, cov: np.ndarray, tickers: list[str], target: float, cfg: dict
) -> tuple[np.ndarray, float, float] | None:
    """Minimize w'Sigma w subject to w'mu = target, sum(w) = 1, per-asset bounds,
    and group floors/caps. Returns (weights, ret, vol) or None if infeasible."""
    bounds = _bounds(tickers, cfg)
    group_cs = _group_constraints(tickers, cfg["groups"])
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t},
        *group_cs,
    ]
    result = minimize(
        fun=lambda w: float(w @ cov @ w),
        x0=_x0(tickers, cfg),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        return None
    w = result.x
    return w, float(w @ mu), float(np.sqrt(max(w @ cov @ w, 0.0)))


def min_variance_portfolio(
    mu: np.ndarray, cov: np.ndarray, tickers: list[str], cfg: dict
) -> tuple[np.ndarray, float, float]:
    bounds = _bounds(tickers, cfg)
    group_cs = _group_constraints(tickers, cfg["groups"])
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}, *group_cs]
    result = minimize(
        fun=lambda w: float(w @ cov @ w),
        x0=_x0(tickers, cfg),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        print(f"[optimizer] ERROR: min-variance solve failed: {result.message}")
        sys.exit(1)
    w = result.x
    return w, float(w @ mu), float(np.sqrt(max(w @ cov @ w, 0.0)))


def max_sharpe_portfolio(
    mu: np.ndarray, cov: np.ndarray, tickers: list[str], rf: float, cfg: dict
) -> tuple[np.ndarray, float, float]:
    bounds = _bounds(tickers, cfg)
    group_cs = _group_constraints(tickers, cfg["groups"])
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}, *group_cs]

    def neg_sharpe(w: np.ndarray) -> float:
        ret = float(w @ mu)
        vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
        return -(ret - rf) / vol

    result = minimize(
        fun=neg_sharpe,
        x0=_x0(tickers, cfg),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 500},
    )
    if not result.success:
        print(f"[optimizer] ERROR: max-Sharpe solve failed: {result.message}")
        sys.exit(1)
    w = result.x
    return w, float(w @ mu), float(np.sqrt(max(w @ cov @ w, 0.0)))


# ---------- Frontier sweep ----------------------------------------------------


def build_frontier(
    mu: np.ndarray, cov: np.ndarray, tickers: list[str], cfg: dict,
    n_points: int = N_POINTS,
) -> dict[str, list]:
    targets = np.linspace(mu.min(), mu.max(), n_points)
    weights: list[list[float]] = []
    rets: list[float] = []
    vols: list[float] = []
    for t in targets:
        sol = min_variance_for_target(mu, cov, tickers, float(t), cfg)
        if sol is None:
            continue
        w, r, v = sol
        weights.append(w.tolist())
        rets.append(r)
        vols.append(v)
    if not rets:
        print("[optimizer] ERROR: no feasible frontier points. Check constraints.")
        sys.exit(1)
    return {"weights": weights, "returns": rets, "vols": vols}


# ---------- Crisis stress test ------------------------------------------------


def _resolve_indices(assets: list[str], classes: set[str]) -> list[int]:
    return [i for i, a in enumerate(assets) if a in classes]


def crisis_corr(base_corr: np.ndarray, assets: list[str]) -> np.ndarray:
    """2008-style crisis: equity x {equity, bond, gold} correlations are
    overwritten with prescribed values. Other pairs keep base correlations."""
    c = base_corr.copy()
    eq = _resolve_indices(assets, EQUITY_CLASSES)
    bd = _resolve_indices(assets, BOND_CLASSES)
    gd = _resolve_indices(assets, GOLD_CLASSES)

    def set_pair(a: int, b: int, v: float) -> None:
        c[a, b] = v
        c[b, a] = v

    for i in eq:
        for j in eq:
            if i != j:
                set_pair(i, j, CRISIS_EQ_EQ)
        for j in bd:
            set_pair(i, j, CRISIS_EQ_BD)
        for j in gd:
            set_pair(i, j, CRISIS_EQ_GD)

    np.fill_diagonal(c, 1.0)
    return c


def stress_test(
    sigma: np.ndarray,
    base_corr: np.ndarray,
    assets: list[str],
    frontier_weights: list[list[float]],
) -> list[float]:
    c_corr = crisis_corr(base_corr, assets)
    c_cov = covariance(sigma, c_corr)
    stressed: list[float] = []
    for w in frontier_weights:
        w_arr = np.asarray(w)
        stressed.append(float(np.sqrt(max(w_arr @ c_cov @ w_arr, 0.0))))
    return stressed


# ---------- Verification ------------------------------------------------------


def _verify_constraints(label: str, tickers: list[str], w: np.ndarray,
                        cfg: dict, tol: float = 1e-6) -> None:
    for t, wi in zip(tickers, w):
        cap = _effective_cap(t, cfg["asset_caps"])
        floor = _effective_floor(t, cfg["asset_floors"])
        if wi > cap + tol:
            print(f"  [CAP VIOLATION] {label}: {t} weight {wi:.6f} > cap {cap:.6f}")
            sys.exit(1)
        if wi < floor - tol:
            print(f"  [FLOOR VIOLATION] {label}: {t} weight {wi:.6f} < floor {floor:.6f}")
            sys.exit(1)
    for name, g in cfg["groups"].items():
        idx = [tickers.index(t) for t in g.get("tickers", []) if t in tickers]
        if not idx:
            continue
        s = float(np.sum([w[i] for i in idx]))
        gf = float(g.get("floor", 0.0))
        gc = float(g.get("cap", 1.0))
        if s < gf - tol:
            print(f"  [GROUP FLOOR VIOLATION] {label}: '{name}' sum {s:.6f} < {gf:.6f}")
            sys.exit(1)
        if s > gc + tol:
            print(f"  [GROUP CAP VIOLATION] {label}: '{name}' sum {s:.6f} > {gc:.6f}")
            sys.exit(1)


# ---------- Diagnostics -------------------------------------------------------


def _crisis_diagnostic(
    assets: list[str],
    tickers: list[str],
    sigma: np.ndarray,
    base_corr: np.ndarray,
    frontier: dict,
) -> None:
    print("\n" + "=" * 86)
    print("Crisis-vol diagnostic")
    print("-" * 86)
    corr_c = crisis_corr(base_corr, assets)
    cov_normal = covariance(sigma, base_corr)
    cov_crisis = covariance(sigma, corr_c)

    print(f"  {'Ticker':<10s}  {'sigma^2 normal':>15s}  {'sigma^2 crisis':>15s}  {'match':>6s}")
    diag_normal = np.diag(cov_normal)
    diag_crisis = np.diag(cov_crisis)
    for t, dn, dc in zip(tickers, diag_normal, diag_crisis):
        match = "OK" if abs(dn - dc) < 1e-12 else "DIFF"
        print(f"  {t:<10s}  {dn:>15.6f}  {dc:>15.6f}  {match:>6s}")
    print("  (Diagonals identical by construction — only off-diagonal correlations switch.)")

    weights = np.asarray(frontier["weights"])
    rets = np.asarray(frontier["returns"])
    vols_normal = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_normal, weights))
    vols_crisis = np.sqrt(np.einsum("ij,jk,ik->i", weights, cov_crisis, weights))
    diff = vols_crisis - vols_normal
    inversions = np.where(diff < -1e-6)[0]
    delta_max = float(diff.max())
    delta_min = float(diff.min())
    print(f"\n  Crisis overrides: eq-eq={CRISIS_EQ_EQ:+.2f}, "
          f"eq-bd={CRISIS_EQ_BD:+.2f}, eq-gd={CRISIS_EQ_GD:+.2f}  "
          f"(2008-style flight-to-quality; bond pairs unchanged)")
    print(f"  Crisis-vol delta across {len(rets)} frontier points: "
          f"min={delta_min:+.2%}  max={delta_max:+.2%}")
    if len(inversions) > 0:
        worst = int(np.argmin(diff))
        print(f"  {len(inversions)} bond-heavy points show crisis_vol < normal_vol  "
              f"(expected: bonds diversify better in deflationary crash).")
        print(f"  Largest negative delta: idx {worst}, return={rets[worst]:.2%}, "
              f"normal={vols_normal[worst]:.2%}, crisis={vols_crisis[worst]:.2%}, "
              f"delta={diff[worst]:+.2%}")
    else:
        print("  No inversions (all crisis_vol >= normal_vol).")
    print("=" * 86)


def _resolve_rf(assets: list[str], mu: np.ndarray) -> float:
    if RF_ASSET not in assets:
        print(f"[optimizer] ERROR: rf asset '{RF_ASSET}' not in universe.")
        sys.exit(1)
    return float(mu[assets.index(RF_ASSET)])


def _print_weights(label: str, assets: list[str], tickers: list[str],
                   w: np.ndarray, cfg: dict) -> None:
    print(f"\n{label}:")
    for a, t, wi in zip(assets, tickers, w):
        cap = _effective_cap(t, cfg["asset_caps"])
        floor = _effective_floor(t, cfg["asset_floors"])
        markers = []
        if cap < MAX_WEIGHT and wi >= cap - 1e-4:
            markers.append("CAP")
        if floor > 0 and wi <= floor + 1e-4:
            markers.append("FLOOR")
        marker = "  <-- " + "+".join(markers) if markers else ""
        has_real_bound = cap < MAX_WEIGHT or floor > 0
        if wi > 1e-4 or has_real_bound:
            print(f"  {a:<20s} ({t:<10s}) {wi:6.2%}   "
                  f"[{floor:5.0%} .. {cap:5.0%}]{marker}")
    print("  Group sums:")
    for name, g in cfg["groups"].items():
        idx = [tickers.index(t) for t in g.get("tickers", []) if t in tickers]
        if not idx:
            continue
        s = sum(w[i] for i in idx)
        gf = float(g.get("floor", 0.0))
        gc = float(g.get("cap", 1.0))
        markers = []
        if gf > 0 and s <= gf + 1e-4:
            markers.append("FLOOR")
        if gc < 1.0 and s >= gc - 1e-4:
            markers.append("CAP")
        marker = "  <-- " + "+".join(markers) if markers else ""
        print(f"    {name:<20s} {s:6.2%}   [{gf:5.0%} .. {gc:5.0%}]{marker}")


def _print_mu_vector(
    assets: list[str],
    tickers: list[str],
    mu: np.ndarray,
    mu_source: list[str],
    mu_calc: np.ndarray,
    mu_override: np.ndarray,
) -> None:
    print("=" * 86)
    print("mu vector entering optimizer (after override resolution):")
    print("-" * 86)
    print(f"  {'Asset class':<20s} {'Ticker':<10s}   {'mu (used)':>10s}  "
          f"{'source':<10s}  {'historical':>11s}  {'override':>10s}")
    print("-" * 86)
    for a, t, m, src, c, o in zip(assets, tickers, mu, mu_source, mu_calc, mu_override):
        c_str = f"{c:>10.2%}" if not np.isnan(c) else f"{'n/a':>10s}"
        o_str = f"{o:>9.2%}" if not np.isnan(o) else f"{'(blank)':>9s}"
        print(f"  {a:<20s} {t:<10s}   {m:>10.2%}  {src:<10s}  {c_str}  {o_str}")
    print("=" * 86)


def _print_sigma_vector(
    assets: list[str],
    tickers: list[str],
    sigma: np.ndarray,
    sigma_source: list[str],
    sigma_implied: np.ndarray,
    sigma_override: np.ndarray,
) -> None:
    print("=" * 86)
    print("sigma vector entering optimizer (after override resolution):")
    print("-" * 86)
    print(f"  {'Asset class':<20s} {'Ticker':<10s}   {'vol (used)':>10s}  "
          f"{'source':<14s}  {'implied':>10s}  {'override':>10s}")
    print("-" * 86)
    for a, t, s, src, imp, o in zip(assets, tickers, sigma, sigma_source, sigma_implied, sigma_override):
        o_str = f"{o:>9.2%}" if not np.isnan(o) else f"{'(blank)':>9s}"
        print(f"  {a:<20s} {t:<10s}   {s:>10.2%}  {src:<14s}  {imp:>10.2%}  {o_str}")
    print("=" * 86)


# ---------- Orchestration -----------------------------------------------------


def build_and_save(cfg: dict | None = None) -> dict:
    """Run the full pipeline with the given constraints (loaded from disk
    if cfg is None). Writes data/frontier.json and returns the payload."""
    if cfg is None:
        cfg = load_constraints()
    else:
        save_constraints(cfg)

    assets, tickers, mu, cov, corr = load_inputs(cov_method=COV_METHOD)
    sigma = np.sqrt(np.diag(cov))
    rf = _resolve_rf(assets, mu)

    frontier = build_frontier(mu, cov, tickers, cfg)
    mv_w, mv_r, mv_v = min_variance_portfolio(mu, cov, tickers, cfg)
    ms_w, ms_r, ms_v = max_sharpe_portfolio(mu, cov, tickers, rf, cfg)
    stressed_vols = stress_test(sigma, corr, assets, frontier["weights"])

    _verify_constraints("min_variance", tickers, mv_w, cfg)
    _verify_constraints("max_sharpe", tickers, ms_w, cfg)
    for i, w_row in enumerate(frontier["weights"]):
        _verify_constraints(f"frontier[{i}]", tickers, np.asarray(w_row), cfg)

    payload = {
        "assets": assets,
        "etfs": tickers,
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "rf": rf,
        "rf_source": RF_ASSET,
        "cov_method": COV_METHOD,
        "constraints": {
            "long_only": True,
            "max_weight_default": MAX_WEIGHT,
            "asset_caps": cfg["asset_caps"],
            "asset_floors": cfg["asset_floors"],
            "groups": cfg["groups"],
            "sum_to_one": True,
        },
        "frontier": {
            "returns": frontier["returns"],
            "vols": frontier["vols"],
            "weights": frontier["weights"],
            "crisis_vols": stressed_vols,
        },
        "min_variance": {"weights": mv_w.tolist(), "return": mv_r, "vol": mv_v},
        "max_sharpe": {
            "weights": ms_w.tolist(),
            "return": ms_r,
            "vol": ms_v,
            "sharpe": (ms_r - rf) / ms_v if ms_v > 0 else None,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    cfg = load_constraints()
    assets, tickers, mu, cov, corr = load_inputs(cov_method=COV_METHOD)
    sigma = np.sqrt(np.diag(cov))
    rf = _resolve_rf(assets, mu)

    print(f"Loaded {len(assets)} assets from {INPUTS_PATH.name}")
    _print_mu_vector(
        assets, tickers, mu,
        load_inputs.last_mu_source,
        load_inputs.last_mu_calc,
        load_inputs.last_mu_override,
    )
    historical_fallbacks = [
        t for t, src in zip(tickers, load_inputs.last_mu_source) if src == "historical"
    ]
    if historical_fallbacks:
        print(f"\n[WARN] mu override blank for {historical_fallbacks} — "
              "falling back to historical calc. Fill in 'Ann Return (override)' "
              "to use a forward view.")

    _print_sigma_vector(
        assets, tickers, sigma,
        load_inputs.last_sigma_source,
        load_inputs.last_sigma_implied,
        load_inputs.last_sigma_override,
    )
    print(f"\nCovariance method: {COV_METHOD}  "
          f"(correlation structure from Covariance_{COV_METHOD.capitalize()}; "
          f"sigma resolved per-asset from Returns sheet)")
    print(f"rf = {rf:.2%}  (from '{RF_ASSET}')")

    print("\nPer-asset bounds:")
    for a, t in zip(assets, tickers):
        cap = _effective_cap(t, cfg["asset_caps"])
        floor = _effective_floor(t, cfg["asset_floors"])
        print(f"  {a:<20s} ({t:<10s})  [{floor:5.0%} .. {cap:5.0%}]")
    print("\nGroup bounds:")
    for name, g in cfg["groups"].items():
        members = ", ".join(g.get("tickers", []))
        print(f"  {name:<20s} [{float(g.get('floor',0.0)):5.0%} .. "
              f"{float(g.get('cap',1.0)):5.0%}]  members=[{members}]")

    frontier = build_frontier(mu, cov, tickers, cfg)
    mv_w, mv_r, mv_v = min_variance_portfolio(mu, cov, tickers, cfg)
    ms_w, ms_r, ms_v = max_sharpe_portfolio(mu, cov, tickers, rf, cfg)
    stressed_vols = stress_test(sigma, corr, assets, frontier["weights"])

    _verify_constraints("min_variance", tickers, mv_w, cfg)
    _verify_constraints("max_sharpe", tickers, ms_w, cfg)
    frontier_w = np.asarray(frontier["weights"])
    max_per_asset = frontier_w.max(axis=0)
    min_per_asset = frontier_w.min(axis=0)
    for i, w_row in enumerate(frontier["weights"]):
        _verify_constraints(f"frontier[{i}]", tickers, np.asarray(w_row), cfg)

    payload = {
        "assets": assets,
        "etfs": tickers,
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "rf": rf,
        "rf_source": RF_ASSET,
        "cov_method": COV_METHOD,
        "constraints": {
            "long_only": True,
            "max_weight_default": MAX_WEIGHT,
            "asset_caps": cfg["asset_caps"],
            "asset_floors": cfg["asset_floors"],
            "groups": cfg["groups"],
            "sum_to_one": True,
        },
        "frontier": {
            "returns": frontier["returns"],
            "vols": frontier["vols"],
            "weights": frontier["weights"],
            "crisis_vols": stressed_vols,
        },
        "min_variance": {"weights": mv_w.tolist(), "return": mv_r, "vol": mv_v},
        "max_sharpe": {
            "weights": ms_w.tolist(),
            "return": ms_r,
            "vol": ms_v,
            "sharpe": (ms_r - rf) / ms_v if ms_v > 0 else None,
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    mv_crisis = stress_test(sigma, corr, assets, [mv_w.tolist()])[0]
    print(f"\nFrontier points: {len(frontier['returns'])}")
    print(f"Min variance:  return={mv_r:6.2%}  vol={mv_v:6.2%}  crisis_vol={mv_crisis:6.2%}")
    print(f"Max Sharpe:    return={ms_r:6.2%}  vol={ms_v:6.2%}  sharpe={(ms_r - rf) / ms_v:.2f}")
    _print_weights("Min variance weights", assets, tickers, mv_w, cfg)
    _print_weights("Max Sharpe weights", assets, tickers, ms_w, cfg)

    frontier_rets = np.asarray(frontier["returns"])
    frontier_vols = np.asarray(frontier["vols"])
    max_ret_idx = int(np.argmax(frontier_rets))
    print(f"\nFrontier return range: {frontier_rets.min():.2%}  ..  {frontier_rets.max():.2%}")
    print(f"Max feasible return: {frontier_rets[max_ret_idx]:.2%}  "
          f"(vol={frontier_vols[max_ret_idx]:.2%})")

    target_ret = 0.075
    target_idx = int(np.argmin(np.abs(frontier_rets - target_ret)))
    tr_r = frontier["returns"][target_idx]
    tr_v = frontier["vols"][target_idx]
    tr_w = np.asarray(frontier["weights"][target_idx])
    tr_crisis = stressed_vols[target_idx]
    tr_sharpe = (tr_r - rf) / tr_v if tr_v > 0 else float("nan")
    print(f"\nPortfolio at {target_ret:.1%} target return  "
          f"(closest frontier point: {tr_r:.2%}):")
    print(f"  vol={tr_v:6.2%}  sharpe={tr_sharpe:.2f}  crisis_vol={tr_crisis:6.2%}")
    _print_weights(f"Weights @ {target_ret:.1%} target", assets, tickers, tr_w, cfg)

    _crisis_diagnostic(assets, tickers, sigma, corr, frontier)

    print("\nFrontier sweep cap check (max weight per asset, must be <= cap):")
    for t, w_max in zip(tickers, max_per_asset):
        cap = _effective_cap(t, cfg["asset_caps"])
        if cap < MAX_WEIGHT:
            ok = "OK" if w_max <= cap + 1e-6 else "VIOLATION"
            print(f"  {t:<10s} max={w_max:6.2%}   cap={cap:5.0%}   {ok}")
    print("Frontier sweep floor check (min weight per asset, must be >= floor):")
    for t, w_min in zip(tickers, min_per_asset):
        floor = _effective_floor(t, cfg["asset_floors"])
        if floor > 0:
            ok = "OK" if w_min >= floor - 1e-6 else "VIOLATION"
            print(f"  {t:<10s} min={w_min:6.2%}   floor={floor:5.0%}   {ok}")
    print("Frontier sweep group check (min/max combined weight per group):")
    for name, g in cfg["groups"].items():
        idx = [tickers.index(t) for t in g.get("tickers", []) if t in tickers]
        if not idx:
            continue
        sums = frontier_w[:, idx].sum(axis=1)
        gf = float(g.get("floor", 0.0))
        gc = float(g.get("cap", 1.0))
        floor_ok = "OK" if sums.min() >= gf - 1e-6 else "VIOLATION"
        cap_ok = "OK" if sums.max() <= gc + 1e-6 else "VIOLATION"
        print(f"  {name:<20s} min={sums.min():6.2%}  max={sums.max():6.2%}   "
              f"[{gf:5.0%} .. {gc:5.0%}]   floor={floor_ok}  cap={cap_ok}")

    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
