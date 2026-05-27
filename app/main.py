"""Streamlit frontend for the MPT efficient frontier.

Run: streamlit run app/main.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT: Path = Path(__file__).resolve().parent.parent
FRONTIER_PATH: Path = ROOT / "data" / "frontier.json"
CONSTRAINTS_PATH: Path = ROOT / "data" / "constraints.json"
OPTIMIZER_SCRIPT: Path = ROOT / "engine" / "optimizer.py"
MAX_WEIGHT_DEFAULT: float = 0.60  # mirrors engine.optimizer.MAX_WEIGHT

BG = "#0e1117"
PANEL = "#161a24"
GRID = "#262a36"
TEXT = "#e6e6e6"
ACCENT = "#4cc9f0"
FRONTIER_COLOR = "#4cc9f0"
CRISIS_COLOR = "#f72585"
ASSET_COLOR = "#9aa5b1"
MAX_SHARPE_COLOR = "#ffd166"
MIN_VAR_COLOR = "#06d6a0"
UTILITY_COLOR = "#bf5af2"
SELECTED_COLOR = "#ffffff"

MODE_MANUAL = "Manual target return"
MODE_UTILITY = "Utility-optimal (A slider)"


# ---------- Data loading -------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_frontier() -> dict:
    if not FRONTIER_PATH.exists():
        st.error(f"{FRONTIER_PATH} not found. Run `make opt` first.")
        st.stop()
    return json.loads(FRONTIER_PATH.read_text())


def select_point(series: list[float], target: float) -> int:
    arr = np.asarray(series)
    return int(np.argmin(np.abs(arr - target)))


def compute_utility(returns: list[float], vols: list[float], A: float) -> np.ndarray:
    """Mean-variance utility: U = mu_p - (A/2) * vol_p^2.  Decimals in, decimals out."""
    r = np.asarray(returns, dtype=float)
    v = np.asarray(vols, dtype=float)
    return r - (A / 2.0) * v**2


def risk_label(A: float) -> str:
    if A <= 1.5:
        return "Aggressive — maximize return"
    if A <= 2.5:
        return "Growth — long horizon, high tolerance"
    if A <= 3.5:
        return "Moderate — balanced"
    if A <= 6.0:
        return "Conservative — capital preservation"
    return "Very conservative — near min variance"


# ---------- Plots --------------------------------------------------------------


def frontier_figure(
    data: dict,
    sel_idx: int,
    show_crisis: bool,
    util_idx: int,
    A: float,
) -> go.Figure:
    fr = data["frontier"]
    mu = data["mu"]
    sigma = data["sigma"]
    etfs = data["etfs"]
    assets = data["assets"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fr["vols"], y=fr["returns"], mode="lines",
        name="Efficient frontier",
        line=dict(color=FRONTIER_COLOR, width=3),
        hovertemplate="vol %{x:.2%}<br>return %{y:.2%}<extra></extra>",
    ))

    if show_crisis:
        fig.add_trace(go.Scatter(
            x=fr["crisis_vols"], y=fr["returns"], mode="lines",
            name="Crisis frontier",
            line=dict(color=CRISIS_COLOR, width=2, dash="dash"),
            hovertemplate="crisis vol %{x:.2%}<br>return %{y:.2%}<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=sigma, y=mu, mode="markers+text",
        name="Assets",
        marker=dict(color=ASSET_COLOR, size=10, line=dict(color=TEXT, width=1)),
        text=etfs, textposition="top center",
        textfont=dict(color=TEXT, size=11),
        customdata=assets,
        hovertemplate="%{customdata} (%{text})<br>vol %{x:.2%}<br>return %{y:.2%}<extra></extra>",
    ))

    mv = data["min_variance"]
    fig.add_trace(go.Scatter(
        x=[mv["vol"]], y=[mv["return"]], mode="markers",
        name="Min variance",
        marker=dict(color=MIN_VAR_COLOR, size=14, symbol="diamond",
                    line=dict(color=TEXT, width=1)),
        hovertemplate="Min variance<br>vol %{x:.2%}<br>return %{y:.2%}<extra></extra>",
    ))

    ms = data["max_sharpe"]
    fig.add_trace(go.Scatter(
        x=[ms["vol"]], y=[ms["return"]], mode="markers",
        name="Max Sharpe",
        marker=dict(color=MAX_SHARPE_COLOR, size=14, symbol="star",
                    line=dict(color=TEXT, width=1)),
        hovertemplate="Max Sharpe<br>vol %{x:.2%}<br>return %{y:.2%}<extra></extra>",
    ))

    # Utility-optimal point (depends on current A)
    fig.add_trace(go.Scatter(
        x=[fr["vols"][util_idx]], y=[fr["returns"][util_idx]],
        mode="markers",
        name=f"Utility-optimal (A={A:.1f})",
        marker=dict(color=UTILITY_COLOR, size=14, symbol="triangle-up",
                    line=dict(color=TEXT, width=1)),
        hovertemplate=f"Utility-optimal (A={A:.1f})<br>vol %{{x:.2%}}<br>return %{{y:.2%}}<extra></extra>",
    ))

    # Selected ring (whichever portfolio is currently displayed)
    sel_vol = fr["vols"][sel_idx]
    sel_ret = fr["returns"][sel_idx]
    fig.add_trace(go.Scatter(
        x=[sel_vol], y=[sel_ret], mode="markers",
        name="Selected",
        marker=dict(color=SELECTED_COLOR, size=18, symbol="circle-open",
                    line=dict(color=SELECTED_COLOR, width=3)),
        hovertemplate="Selected<br>vol %{x:.2%}<br>return %{y:.2%}<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
        title=dict(text="Efficient Frontier", font=dict(size=18, color=TEXT)),
        xaxis=dict(title="Annualised volatility", tickformat=".0%",
                   gridcolor=GRID, zerolinecolor=GRID, color=TEXT),
        yaxis=dict(title="Annualised return", tickformat=".0%",
                   gridcolor=GRID, zerolinecolor=GRID, color=TEXT),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID, borderwidth=1,
                    font=dict(color=TEXT), orientation="h",
                    yanchor="bottom", y=1.02, x=0),
        margin=dict(l=60, r=30, t=70, b=60),
        height=520,
    )
    return fig


def weights_figure(
    weights: list[float],
    etfs: list[str],
    assets: list[str],
    height: int = 420,
    title: str = "Portfolio weights",
    show_asset_class: bool = True,
) -> go.Figure:
    labels = [
        f"{a} ({e})" if show_asset_class else e
        for a, e in zip(assets, etfs)
    ]
    pairs = sorted(zip(weights, labels), key=lambda p: p[0])
    w_sorted = [p[0] for p in pairs]
    l_sorted = [p[1] for p in pairs]

    fig = go.Figure(go.Bar(
        x=w_sorted, y=l_sorted, orientation="h",
        marker=dict(color=ACCENT, line=dict(color=TEXT, width=0.5)),
        text=[f"{w:.1%}" for w in w_sorted],
        textposition="outside",
        textfont=dict(color=TEXT),
        hovertemplate="%{y}<br>weight %{x:.2%}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
        title=dict(text=title, font=dict(size=14, color=TEXT)) if title else None,
        xaxis=dict(tickformat=".0%", gridcolor=GRID, color=TEXT, range=[0, 1]),
        yaxis=dict(gridcolor=GRID, color=TEXT),
        margin=dict(l=180 if show_asset_class else 90, r=40, t=40 if title else 10, b=30),
        height=height,
    )
    return fig


def utility_curve_figure(
    vols: list[float],
    utility: np.ndarray,
    util_idx: int,
    sel_idx: int,
    A: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vols, y=utility, mode="lines",
        name="U(vol)",
        line=dict(color=UTILITY_COLOR, width=2.5),
        hovertemplate="vol %{x:.2%}<br>U %{y:.2%}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[vols[util_idx]], y=[utility[util_idx]], mode="markers",
        name=f"Optimal (A={A:.1f})",
        marker=dict(color=UTILITY_COLOR, size=14, symbol="triangle-up",
                    line=dict(color=TEXT, width=1)),
        hovertemplate=f"Optimal (A={A:.1f})<br>vol %{{x:.2%}}<br>U %{{y:.2%}}<extra></extra>",
    ))
    if sel_idx != util_idx:
        fig.add_trace(go.Scatter(
            x=[vols[sel_idx]], y=[utility[sel_idx]], mode="markers",
            name="Selected",
            marker=dict(color=SELECTED_COLOR, size=14, symbol="circle-open",
                        line=dict(color=SELECTED_COLOR, width=2)),
            hovertemplate="Selected<br>vol %{x:.2%}<br>U %{y:.2%}<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
        title=dict(text=f"Utility curve at A = {A:.1f}", font=dict(size=16, color=TEXT)),
        xaxis=dict(title="Annualised volatility", tickformat=".0%",
                   gridcolor=GRID, color=TEXT),
        yaxis=dict(title=f"U = mu - (A/2) * vol^2", tickformat=".0%",
                   gridcolor=GRID, color=TEXT),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID, borderwidth=1,
                    font=dict(color=TEXT), orientation="h",
                    yanchor="bottom", y=1.02, x=0),
        margin=dict(l=60, r=30, t=70, b=50),
        height=340,
    )
    return fig


# ---------- Comparison table ---------------------------------------------------


def named_portfolios(data: dict, util_idx: int, utility: np.ndarray, A: float) -> list[dict]:
    """Three named portfolios as a list of dicts. crisis_vol for min-var/max-sharpe
    is approximated by the closest frontier sweep point (their separate
    optimizations don't store one)."""
    fr = data["frontier"]
    vols_arr = np.asarray(fr["vols"])

    def crisis_at(vol: float) -> float:
        i = int(np.argmin(np.abs(vols_arr - vol)))
        return fr["crisis_vols"][i]

    mv = data["min_variance"]
    ms = data["max_sharpe"]
    return [
        {
            "name": "Min Variance",
            "return": mv["return"],
            "vol": mv["vol"],
            "weights": mv["weights"],
            "crisis_vol": crisis_at(mv["vol"]),
            "utility": mv["return"] - (A / 2.0) * mv["vol"] ** 2,
        },
        {
            "name": f"Utility Optimal (A={A:.1f})",
            "return": fr["returns"][util_idx],
            "vol": fr["vols"][util_idx],
            "weights": fr["weights"][util_idx],
            "crisis_vol": fr["crisis_vols"][util_idx],
            "utility": float(utility[util_idx]),
        },
        {
            "name": "Max Sharpe",
            "return": ms["return"],
            "vol": ms["vol"],
            "weights": ms["weights"],
            "crisis_vol": crisis_at(ms["vol"]),
            "utility": ms["return"] - (A / 2.0) * ms["vol"] ** 2,
        },
    ]


def render_comparison(data: dict, portfolios: list[dict], A: float, rf: float) -> None:
    rows = []
    for p in portfolios:
        sh = (p["return"] - rf) / p["vol"] if p["vol"] > 0 else float("nan")
        rows.append({
            "Portfolio": p["name"],
            "E[R]": f"{p['return']:.2%}",
            "Vol": f"{p['vol']:.2%}",
            "Sharpe": f"{sh:.2f}",
            f"U(A={A:.1f})": f"{p['utility']:.2%}",
            "Crisis Vol": f"{p['crisis_vol']:.2%}",
        })
    df = pd.DataFrame(rows).set_index("Portfolio")
    st.dataframe(df, use_container_width=True)

    cols = st.columns(3)
    for col, p in zip(cols, portfolios):
        with col:
            st.plotly_chart(
                weights_figure(
                    p["weights"], data["etfs"], data["assets"],
                    height=320, title=p["name"], show_asset_class=False,
                ),
                use_container_width=True,
                config={"displayModeBar": False},
            )


# ---------- Constraints I/O ----------------------------------------------------


def load_constraints_file() -> dict:
    """Read constraints.json directly. Optimizer creates it on first run with
    defaults — if it's still missing, the app surfaces a friendly error."""
    if not CONSTRAINTS_PATH.exists():
        st.error(
            f"{CONSTRAINTS_PATH.name} not found. "
            "Run `python engine/optimizer.py` once to seed defaults."
        )
        st.stop()
    return json.loads(CONSTRAINTS_PATH.read_text())


def save_constraints_file(cfg: dict) -> None:
    CONSTRAINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONSTRAINTS_PATH.write_text(json.dumps(cfg, indent=2))


def rebuild_frontier() -> tuple[bool, str]:
    """Run the optimizer in a subprocess. Returns (success, stderr-or-stdout-tail)."""
    result = subprocess.run(
        [sys.executable, str(OPTIMIZER_SCRIPT)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, (result.stderr or result.stdout)[-2000:]
    return True, result.stdout[-2000:]


def validate_new_cfg(cfg: dict, tickers: list[str]) -> list[str]:
    """Cheap sanity checks before kicking off a rebuild. Catches the common
    user errors so the optimizer subprocess doesn't have to bail out."""
    errs: list[str] = []
    caps = cfg["asset_caps"]
    floors = cfg["asset_floors"]

    for t in tickers:
        f = float(floors.get(t, 0.0))
        c = float(caps.get(t, MAX_WEIGHT_DEFAULT))
        if f < 0 or f > 1:
            errs.append(f"{t}: floor {f:.2%} must be between 0% and 100%.")
        if c < 0 or c > 1:
            errs.append(f"{t}: cap {c:.2%} must be between 0% and 100%.")
        if f > c + 1e-9:
            errs.append(f"{t}: floor {f:.2%} > cap {c:.2%}.")

    floor_sum = sum(float(floors.get(t, 0.0)) for t in tickers)
    if floor_sum > 1.0 + 1e-9:
        errs.append(
            f"Sum of asset floors = {floor_sum:.2%} > 100% — portfolio infeasible."
        )

    for name, g in cfg["groups"].items():
        gf = float(g.get("floor", 0.0))
        gc = float(g.get("cap", 1.0))
        if gf < 0 or gf > 1:
            errs.append(f"Group '{name}': floor {gf:.2%} must be between 0% and 100%.")
        if gc < 0 or gc > 1:
            errs.append(f"Group '{name}': cap {gc:.2%} must be between 0% and 100%.")
        if gf > gc + 1e-9:
            errs.append(f"Group '{name}': floor {gf:.2%} > cap {gc:.2%}.")
        member_cap_sum = sum(float(caps.get(t, MAX_WEIGHT_DEFAULT))
                             for t in g.get("tickers", []) if t in tickers)
        if gf > member_cap_sum + 1e-9:
            errs.append(
                f"Group '{name}': floor {gf:.2%} exceeds sum of member caps "
                f"{member_cap_sum:.2%}."
            )
    return errs


# ---------- Constraints tab ----------------------------------------------------


def render_constraints_tab(data: dict) -> None:
    cfg = load_constraints_file()
    tickers: list[str] = data["etfs"]
    assets: list[str] = data["assets"]

    st.subheader("Per-asset bounds")
    st.caption(
        "Floor and cap for each ETF. Floor = 0% means no lower bound. "
        "Cap blanks default to 60% in the optimizer."
    )
    asset_rows = []
    for a, t in zip(assets, tickers):
        asset_rows.append({
            "Asset class": a,
            "Ticker": t,
            "Floor %": round(cfg["asset_floors"].get(t, 0.0) * 100, 2),
            "Cap %":   round(cfg["asset_caps"].get(t, MAX_WEIGHT_DEFAULT) * 100, 2),
        })
    asset_df = pd.DataFrame(asset_rows)

    edited_assets = st.data_editor(
        asset_df,
        column_config={
            "Asset class": st.column_config.TextColumn("Asset class", disabled=True),
            "Ticker": st.column_config.TextColumn("Ticker", disabled=True),
            "Floor %": st.column_config.NumberColumn(
                "Floor %", format="%.2f", min_value=0.0, max_value=100.0, step=0.5,
            ),
            "Cap %": st.column_config.NumberColumn(
                "Cap %", format="%.2f", min_value=0.0, max_value=100.0, step=0.5,
            ),
        },
        hide_index=True,
        use_container_width=True,
        key="asset_bounds_editor",
    )

    st.subheader("Group bounds")
    st.caption(
        "Combined-weight floor and cap per asset-class group. "
        "Equities floor of 25% replaces the legacy combined-equity constraint."
    )
    group_rows = []
    group_order = list(cfg["groups"].keys())
    for name in group_order:
        g = cfg["groups"][name]
        group_rows.append({
            "Group": name,
            "Members": ", ".join(g.get("tickers", [])),
            "Floor %": round(float(g.get("floor", 0.0)) * 100, 2),
            "Cap %":   round(float(g.get("cap", 1.0)) * 100, 2),
        })
    group_df = pd.DataFrame(group_rows)

    edited_groups = st.data_editor(
        group_df,
        column_config={
            "Group": st.column_config.TextColumn("Group", disabled=True),
            "Members": st.column_config.TextColumn("Members", disabled=True, width="large"),
            "Floor %": st.column_config.NumberColumn(
                "Floor %", format="%.2f", min_value=0.0, max_value=100.0, step=0.5,
            ),
            "Cap %": st.column_config.NumberColumn(
                "Cap %", format="%.2f", min_value=0.0, max_value=100.0, step=0.5,
            ),
        },
        hide_index=True,
        use_container_width=True,
        key="group_bounds_editor",
    )

    st.markdown("---")
    col_btn, col_status = st.columns([1, 4])
    with col_btn:
        save_clicked = st.button(
            "Save & rebuild frontier",
            type="primary",
            use_container_width=True,
            help="Writes constraints.json, re-runs the SLSQP sweep, "
                 "and refreshes the Portfolio tab.",
        )

    if save_clicked:
        new_cfg = {
            "asset_caps": {
                row["Ticker"]: float(row["Cap %"]) / 100.0
                for _, row in edited_assets.iterrows()
            },
            "asset_floors": {
                row["Ticker"]: float(row["Floor %"]) / 100.0
                for _, row in edited_assets.iterrows()
            },
            "groups": {
                row["Group"]: {
                    "tickers": cfg["groups"][row["Group"]].get("tickers", []),
                    "floor": float(row["Floor %"]) / 100.0,
                    "cap": float(row["Cap %"]) / 100.0,
                }
                for _, row in edited_groups.iterrows()
            },
        }

        errors = validate_new_cfg(new_cfg, tickers)
        if errors:
            with col_status:
                for e in errors:
                    st.error(e)
            return

        save_constraints_file(new_cfg)
        with st.spinner("Solving 200 SLSQP problems..."):
            ok, log_tail = rebuild_frontier()
        if not ok:
            st.error("Optimizer failed. Tail of stderr:")
            st.code(log_tail)
            return
        load_frontier.clear()
        with col_status:
            st.success("Frontier rebuilt. Switch back to the Portfolio tab.")
        st.rerun()


# ---------- Portfolio tab ------------------------------------------------------


def render_portfolio_tab(data: dict) -> None:
    fr = data["frontier"]
    vols = fr["vols"]
    returns = fr["returns"]
    weights = fr["weights"]
    crisis_vols = fr["crisis_vols"]
    rf = float(data.get("rf", 0.0))

    ret_min = float(min(returns))
    ret_max = float(max(returns))
    slider_min_pct = round(ret_min * 100, 2)
    slider_max_pct = round(ret_max * 100, 2)
    slider_step_pct = max(round((ret_max - ret_min) * 100 / 200, 2), 0.01)

    # ----- Session state -----
    if "mode" not in st.session_state:
        st.session_state.mode = MODE_UTILITY
    if "target_return_pct" not in st.session_state:
        st.session_state.target_return_pct = round((ret_min + ret_max) / 2 * 100, 2)
    if "risk_aversion" not in st.session_state:
        st.session_state.risk_aversion = 3.0

    def pick_max_sharpe() -> None:
        st.session_state.mode = MODE_MANUAL
        v = round(float(data["max_sharpe"]["return"]) * 100, 2)
        st.session_state.target_return_pct = max(slider_min_pct, min(slider_max_pct, v))

    # ----- Sidebar -----
    with st.sidebar:
        st.header("Controls")

        if st.button(
            "Refresh from inputs.xlsx",
            use_container_width=True,
            help="Re-runs the optimizer against the current inputs.xlsx and "
                 "constraints.json. Use this after editing mu / sigma overrides "
                 "or any other Returns-sheet value in Excel.",
        ):
            with st.spinner("Solving 200 SLSQP problems..."):
                ok, log_tail = rebuild_frontier()
            if not ok:
                st.error("Optimizer failed. Tail of stderr:")
                st.code(log_tail)
            else:
                load_frontier.clear()
                st.success("Frontier refreshed.")
                st.rerun()

        st.markdown("---")
        st.radio("Selection mode", [MODE_MANUAL, MODE_UTILITY], key="mode")
        st.button("Pick Max Sharpe", on_click=pick_max_sharpe,
                  use_container_width=True,
                  help=f"Switches to Manual and snaps the slider to "
                       f"{data['max_sharpe']['return']:.2%}.")
        st.markdown("---")
        st.slider(
            "Target annual return",
            min_value=slider_min_pct, max_value=slider_max_pct,
            step=slider_step_pct, format="%.2f%%",
            key="target_return_pct",
            disabled=(st.session_state.mode == MODE_UTILITY),
        )
        st.slider(
            "Risk aversion (A)",
            min_value=1.0, max_value=8.0, step=0.5,
            key="risk_aversion",
            help=("A=1: aggressive (max return)\n"
                  "A=2: growth (long horizon)\n"
                  "A=3: moderate (balanced)\n"
                  "A=5: conservative\n"
                  "A=8: very conservative (near min variance)"),
        )
        A = float(st.session_state.risk_aversion)
        st.caption(f"A={A:.1f} → {risk_label(A)}")
        st.markdown("---")
        show_crisis = st.checkbox("Overlay crisis frontier", value=True)
        st.markdown("---")
        st.caption(f"Frontier points: {len(returns)}")
        st.caption(f"Return range: {ret_min:.2%} – {ret_max:.2%}")
        st.caption(f"Max Sharpe @ {data['max_sharpe']['return']:.2%}, "
                   f"vol {data['max_sharpe']['vol']:.2%}, "
                   f"Sharpe {data['max_sharpe']['sharpe']:.2f}")

    # ----- Compute utility on every frontier point -----
    utility = compute_utility(returns, vols, A)
    util_idx = int(np.argmax(utility))

    # ----- Resolve selected portfolio based on mode -----
    if st.session_state.mode == MODE_UTILITY:
        sel_idx = util_idx
    else:
        sel_idx = select_point(returns, float(st.session_state.target_return_pct) / 100.0)

    sel_vol = vols[sel_idx]
    sel_ret = returns[sel_idx]
    sel_weights = weights[sel_idx]
    sel_crisis = crisis_vols[sel_idx]
    sel_u = float(utility[sel_idx])
    sharpe = (sel_ret - rf) / sel_vol if sel_vol > 0 else float("nan")

    # ----- Header -----
    st.title("Markowitz Efficient Frontier")
    st.caption(
        "Long-only with per-asset caps and floors. Solved via SLSQP on the quadratic "
        "Lagrangian L(w, lam, gam) = w'Sigma w - lam (w'mu - R*) - gam (w'1 - 1). "
        "Choose a portfolio either by target return or by risk-aversion-driven utility: "
        "U(w) = mu_p - (A/2) * vol_p^2."
    )

    # ----- Frontier chart -----
    st.plotly_chart(
        frontier_figure(data, sel_idx, show_crisis, util_idx=util_idx, A=A),
        use_container_width=True, config={"displayModeBar": False},
    )

    # ----- Metric tiles for selected portfolio -----
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Expected return", f"{sel_ret:.2%}")
    c2.metric("Volatility", f"{sel_vol:.2%}")
    c3.metric(f"Sharpe (rf={rf:.2%})", f"{sharpe:.2f}")
    c4.metric(f"Utility (A={A:.1f})", f"{sel_u:.2%}")
    c5.metric("Crisis vol", f"{sel_crisis:.2%}", delta=f"{sel_crisis - sel_vol:+.2%}")

    # ----- Weights of selected portfolio -----
    st.plotly_chart(
        weights_figure(sel_weights, data["etfs"], data["assets"]),
        use_container_width=True, config={"displayModeBar": False},
    )

    # ----- Comparison: Min Variance / Utility Optimal / Max Sharpe -----
    st.markdown("---")
    st.subheader("Compare named portfolios")
    portfolios = named_portfolios(data, util_idx, utility, A)
    render_comparison(data, portfolios, A, rf)

    # ----- Utility curve -----
    st.markdown("---")
    st.plotly_chart(
        utility_curve_figure(vols, utility, util_idx, sel_idx, A),
        use_container_width=True, config={"displayModeBar": False},
    )
    st.caption(
        "A=2 suits a long-horizon growth investor; A=4–5 = moderate. "
        "The utility-optimal portfolio is not the same as Max Sharpe — they coincide "
        "only when A equals the slope of the CML at the tangency point."
    )


# ---------- App entry ----------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="MPT Efficient Frontier", layout="wide")

    st.markdown(
        f"""
        <style>
            .stApp {{ background-color: {BG}; color: {TEXT}; }}
            section[data-testid="stSidebar"] {{ background-color: {PANEL}; }}
            .stMetric {{ background-color: {PANEL}; padding: 0.6rem 0.9rem;
                         border-radius: 8px; border: 1px solid {GRID}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    data = load_frontier()

    tab_portfolio, tab_constraints = st.tabs(["Portfolio", "Constraints"])
    with tab_portfolio:
        render_portfolio_tab(data)
    with tab_constraints:
        render_constraints_tab(data)


if __name__ == "__main__":
    main()
