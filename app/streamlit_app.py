"""I-Know-Puck draft room.   Run:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iknowpuck.config import RUNS_DIR, SLOT_NAMES, load_credentials  # noqa: E402
from iknowpuck.data.espn import EspnClient, EspnError  # noqa: E402
from iknowpuck.draft import availability, predraft_plan, recommend  # noqa: E402
from iknowpuck.pipeline import build  # noqa: E402

st.set_page_config(page_title="I Know Puck", page_icon="🏒", layout="wide")


@st.cache_resource(show_spinner="Building projections and opponent models (first run ~2 min)...")
def get_bundle(season: int, refresh_token: int):
    return build(season, refresh=refresh_token > 0)


creds = load_credentials()
if "refresh" not in st.session_state:
    st.session_state.refresh = 0
if "picks" not in st.session_state:
    st.session_state.picks = []  # list[(team_id, player_id)]

# --- sidebar -------------------------------------------------------------------------------------
with st.sidebar:
    st.title("🏒 I Know Puck")
    season = st.number_input("ESPN season (year season ends)", value=creds.season, step=1)
    if st.button("Rebuild data + models", help="Re-pull ESPN/MoneyPuck and refit everything"):
        st.session_state.refresh += 1
        get_bundle.clear()

b = get_bundle(int(season), st.session_state.refresh)
S = b.settings
names = b.team_labels() or {t: f"Team {t}" for t in range(1, S.n_teams + 1)}  # team id -> manager name

with st.sidebar:
    order_default = S.pick_order or list(names)
    order_txt = st.text_input("Round-1 pick order (team ids)", ",".join(map(str, order_default)))
    order = [int(x) for x in order_txt.split(",") if x.strip().isdigit()]
    my_team = st.selectbox("My team", list(names), index=list(names).index(S.my_team_id) if S.my_team_id in names else 0, format_func=lambda t: f"{t}: {names[t]}")
    st.caption(f"My slot: **{order.index(my_team) + 1 if my_team in order else '?'}** of {len(order)} · {S.rounds} rounds · {S.scoring_type}")
    st.divider()
    n_cand = st.slider("Candidates evaluated", 4, 20, 10)
    n_roll = st.slider("Rollouts per candidate", 5, 100, 25, help="More = tighter estimates, slower")
    live = st.toggle("Live sync from ESPN draft", value=False, help="Polls ESPN every 5s; needs league cookies")
    for n in b.notes:
        st.warning(n)

ctx = b.context(order, my_team)
pool = b.pool.frame
pid_to_idx = {int(p): i for i, p in enumerate(pool.player_id)}


def picks_as_idx() -> list[tuple[int, int]]:
    return [(t, pid_to_idx[p]) for t, p in st.session_state.picks if p in pid_to_idx]


def sync_espn():
    try:
        d = EspnClient(creds).draft(int(season), live=True)
    except EspnError as e:
        st.sidebar.error(str(e))
        return
    if len(d):
        st.session_state.picks = [(int(r.team_id), int(r.player_id)) for r in d.itertuples()]


tab_room, tab_plan, tab_board, tab_intel, tab_research = st.tabs(["Draft room", "Pre-draft plan", "Player board", "League intel", "Research"])


# --- draft room ----------------------------------------------------------------------------------
@st.fragment(run_every=5 if live else None)
def draft_room():
    if live:
        sync_espn()
    picks = picks_as_idx()
    n_done = len(st.session_state.picks)
    total = len(ctx.order)
    if n_done >= total:
        st.success("Draft complete.")
    on_clock = ctx.order[min(n_done, total - 1)]
    rnd = n_done // len(order) + 1
    until = next((d for d, t in enumerate(ctx.order[n_done:]) if t == my_team), None)
    c1, c2, c3 = st.columns(3)
    c1.metric("Pick", f"{n_done + 1} / {total}", f"Round {rnd}")
    c2.metric("On the clock", names.get(on_clock, on_clock))
    c3.metric("Picks until mine", "NOW" if until == 0 else (until if until is not None else "-"))

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Recommendation" + (" - you're up!" if on_clock == my_team else f" for {names.get(on_clock)}"))
        if st.button("Run simulation", type="primary") or (live and on_clock == my_team):
            with st.spinner(f"{n_cand} candidates x {n_roll} rollouts..."):
                rec = recommend(ctx, picks, n_candidates=n_cand, n_rollouts=n_roll)
            st.session_state.rec = (n_done, rec)
        if "rec" in st.session_state and st.session_state.rec[0] == n_done:
            rec = st.session_state.rec[1]
            show = rec[["name", "pos", "proj_value", "adp", "win_prob", "gap_to_best", "gap_se", "p_best", "p_avail_next_pick"]].rename(
                columns={"proj_value": "proj pts", "win_prob": "P(win wk)", "gap_to_best": "gap", "gap_se": "gap SE", "p_best": "P(best)", "p_avail_next_pick": "avail next pick"}
            )
            st.dataframe(
                show.style.format({"proj pts": "{:.0f}", "adp": "{:.1f}", "P(win wk)": "{:.3f}", "gap": "{:+.3f}", "gap SE": "{:.3f}", "P(best)": "{:.0%}", "avail next pick": "{:.0%}"}),
                hide_index=True, use_container_width=True,
            )
            st.caption("P(win wk): simulated probability of winning a weekly matchup with the final roster. "
                       "A gap within ~2 SE of the best is statistically indistinguishable; prefer the one less likely to be available next pick.")
        st.subheader("Record a pick")
        taken_ids = {p for _, p in st.session_state.picks}
        avail = pool[~pool.player_id.isin(taken_ids)].sort_values("fpts", ascending=False)
        pc1, pc2, pc3 = st.columns([3, 2, 1])
        choice = pc1.selectbox("Player", avail.player_id.tolist(), format_func=lambda p: f"{pool.loc[pid_to_idx[p], 'name']} ({pool.loc[pid_to_idx[p], 'pos']}, {pool.loc[pid_to_idx[p], 'fpts']:.0f})")
        team_pick = pc2.selectbox("Team", list(names), index=list(names).index(on_clock) if on_clock in names else 0, format_func=lambda t: names[t])
        if pc3.button("Add"):
            st.session_state.picks.append((team_pick, int(choice)))
            st.rerun()
        u1, u2 = st.columns(2)
        if u1.button("Undo last pick") and st.session_state.picks:
            st.session_state.picks.pop()
            st.rerun()
        if u2.button("Sync once from ESPN"):
            sync_espn()
            st.rerun()

    with right:
        st.subheader("My roster")
        taken, teams = ctx.initial_state(picks)
        mine = teams[my_team]
        rows = [{"player": pool.loc[j, "name"], "pos": pool.loc[j, "pos"], "proj pts": round(ctx.value[j])} for j in mine.roster]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("Open slots: " + ", ".join(f"{SLOT_NAMES[s]} {c}" for s, c in mine.open.items() if c > 0))
        st.subheader("League so far (weekly pts)")
        wk = [{"team": names.get(t, t), "picks": len(teams[t].roster), "weekly pts": ctx.val.weekly_points(teams[t].roster)} for t in order]
        st.dataframe(pd.DataFrame(wk).sort_values("weekly pts", ascending=False).style.format({"weekly pts": "{:.1f}"}), hide_index=True, use_container_width=True)

    with st.expander("Pick log"):
        log = pd.DataFrame([{"#": i + 1, "team": names.get(t, t), "player": pool.loc[pid_to_idx[p], "name"] if p in pid_to_idx else p} for i, (t, p) in enumerate(st.session_state.picks)])
        st.dataframe(log, hide_index=True, use_container_width=True)


with tab_room:
    draft_room()

# --- pre-draft plan ------------------------------------------------------------------------------
with tab_plan:
    st.subheader("Simulated drafts from your slot")
    n_sims = st.slider("Simulated drafts", 20, 400, 100)
    if st.button("Simulate plan"):
        with st.spinner("Simulating full drafts..."):
            targets, summary = predraft_plan(ctx, n_sims=n_sims)
        st.session_state.plan = (targets, summary)
    if "plan" in st.session_state:
        targets, summary = st.session_state.plan
        c1, c2 = st.columns(2)
        c1.metric("Mean P(win weekly matchup)", f"{summary.win_prob.mean():.3f}", f"±{1.96 * summary.win_prob.std() / np.sqrt(len(summary)):.3f} (95% CI)")
        c2.metric("Mean weekly points", f"{summary.weekly_points.mean():.1f}")
        st.dataframe(targets.style.format({"adp": "{:.1f}", "proj_value": "{:.0f}", "share": "{:.0%}"}), hide_index=True, use_container_width=True)
    st.subheader("Chance each player is still there at my first pick")
    if st.button("Compute availability"):
        a = availability(ctx, picks_as_idx(), n_rollouts=80)
        df = pool.assign(p_avail=a)[["name", "pos", "adp", "fpts", "p_avail"]].sort_values("fpts", ascending=False).head(60)
        st.dataframe(df.style.format({"adp": "{:.1f}", "fpts": "{:.0f}", "p_avail": "{:.0%}"}), hide_index=True, use_container_width=True)

# --- player board --------------------------------------------------------------------------------
with tab_board:
    pos_f = st.multiselect("Positions", ["C", "LW", "RW", "D", "G"], default=["C", "LW", "RW", "D", "G"])
    hide_taken = st.checkbox("Hide drafted", value=True)
    board = pool[pool.pos.isin(pos_f)].copy()
    if hide_taken:
        board = board[~board.player_id.isin({p for _, p in st.session_state.picks})]
    board["value_vs_adp"] = board["fpts"].rank(ascending=False) - board["adp"].rank()
    cols = ["name", "pos", "adp", "fpts", "fpts_own", "fpts_espn", "p_30", "value_vs_adp", "injury_status"]
    st.dataframe(
        board.sort_values("fpts", ascending=False)[cols].rename(columns={"fpts": "blend pts", "fpts_own": "own pts", "fpts_espn": "ESPN pts", "p_30": "GP", "value_vs_adp": "rank gap vs ADP"}),
        hide_index=True, use_container_width=True, height=600,
    )
    st.caption("rank gap vs ADP < 0: the market drafts them later than our projection says -> value targets.")

# --- league intel --------------------------------------------------------------------------------
with tab_intel:
    if b.spectral is None:
        st.info("No league draft history available.")
    else:
        mgr_names = b.manager_names
        active = set(b.managers.loc[b.managers.active, "owner_id"].astype(str)) if len(b.managers) else set()

        def who(m):
            n = mgr_names.get(m, m)
            return n if m in active else f"{n} (former)"

        st.subheader("League managers across seasons")
        if len(b.managers):
            md = b.managers.copy()
            md["team slot by season"] = md["team_by_season"].map(lambda d: ", ".join(f"{s}: #{t}" for s, t in sorted(d.items())))
            md["draft picks modeled"] = md["owner_id"].map(b.drafts.groupby("owner_id").size()).fillna(0).astype(int)
            md["status"] = np.where(md.active, "active", "left league")
            st.dataframe(md[["manager", "status", "team slot by season", "draft picks modeled"]].sort_values(["status", "manager"]), hide_index=True, use_container_width=True)
            st.caption("Managers are tracked by ESPN account, not team slot, so history follows the person when slots change hands.")
        tbl = b.spectral.table().reset_index()
        tbl["who"] = tbl["manager"].map(who)
        st.subheader(f"Manager archetypes (spectral clustering, k={b.spectral.k})")
        fig = px.scatter(tbl, x="x", y="y", color=tbl["cluster"].astype(str), text="who", hover_data=["mean_reach", "first_goalie_round", "homer_index", "adp_rank_corr"])
        fig.update_traces(textposition="top center")
        fig.update_layout(xaxis_title="Laplacian eigenvector 2 (Fiedler)", yaxis_title="eigenvector 3", height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(tbl[["who", "cluster", "mean_reach", "reach_early", "d_share_early", "g_share_early", "first_goalie_round", "homer_index", "fiedler"]].round(2), hide_index=True, use_container_width=True)
        st.subheader("Fitted pick tendencies (conditional logit)")
        desc = b.opp_model.describe()
        desc["who"] = desc["manager"].map(lambda m: who(m) if m != "(league)" else m)
        st.dataframe(desc.drop(columns="manager").set_index("who").round(2), use_container_width=True)
        st.caption("adp_sensitivity: how strictly they follow ADP. Position biases are relative to forwards, by round phase; >0 means they take that position earlier than the league.")

# --- research ------------------------------------------------------------------------------------
with tab_research:
    st.subheader("Experiment reports")
    runs = sorted(RUNS_DIR.glob("*/results.json"), reverse=True) if RUNS_DIR.exists() else []
    if not runs:
        st.info("No experiment runs yet. Run:  python -m iknowpuck.experiments")
    for r in runs[:5]:
        payload = json.loads(r.read_text())
        with st.expander(f"{r.parent.name}  (git {payload['meta']['git_commit']})", expanded=r == runs[0]):
            st.markdown((r.parent / "report.md").read_text())
    st.subheader("Projection blend weights (share on own model)")
    from iknowpuck.config import STAT_NAMES

    st.dataframe(pd.DataFrame({"stat": [STAT_NAMES.get(k, k) for k in b.blend_weights], "w_own": list(b.blend_weights.values())}).round(2), hide_index=True)
