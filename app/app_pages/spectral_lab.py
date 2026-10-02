"""Graph spectral analysis of the league's managers: similarity graph, Laplacian spectrum, groups, and
whether the group structure is real (permutation test and bootstrap stability)."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import spearmanr
from ui import CAT, DIV, FAINT, INK, MUTED, SEQ, app_state, fig_show, get_panel, label_positions, load_runs, note, rgba, selected_custom

from iknowpuck.spectral import (bootstrap_coclustering, eigengap_null, manager_season_features, name_clusters,
                                spectral_clusters)

A = app_state()
b = A["b"]
st.markdown("## Graph spectral analysis")
st.caption("How the app groups the managers in your league by drafting style, and a check on whether those groups are real.")
if b.spectral is None:
    st.warning("Spectral analysis needs your league's draft history (ESPN cookies in the .env file).")
    st.stop()

names = b.manager_names
active = set(b.managers.loc[b.managers.active, "owner_id"].astype(str)) if len(b.managers) else set()


def who(m) -> str:
    n_ = names.get(str(m), str(m))
    return n_ if str(m) in active else f"{n_} (former)"


HABITS = {
    "mean_reach": "Reach, all rounds", "reach_early": "Reach, rounds 1-6", "reach_sd": "Reach spread",
    "adp_rank_corr": "Follows ESPN order", "d_share_early": "Defense in rounds 1-6", "g_share_early": "Goalies in rounds 1-6",
    "first_goalie_round": "First goalie round", "n_goalies": "Goalies drafted", "homer_index": "NHL-team loyalty",
    "auto_share": "Autodraft share",
}


@st.cache_resource(show_spinner="Measuring each manager's draft habits...")
def season_habits(_b, season: int) -> pd.DataFrame:
    return manager_season_features(_b.drafts, get_panel(season))


@st.cache_data(max_entries=32, show_spinner="Running the permutation test and bootstrap...")
def structure_tests(_sf: pd.DataFrame, cols: tuple, bw: float, k: int, season: int):
    feats = _sf.groupby("manager")[list(cols)].mean()
    obs, null, p = eigengap_null(feats, n_perm=1000, bandwidth_scale=bw)
    co, ari = bootstrap_coclustering(_sf, list(cols), k=k, n_boot=200, bandwidth_scale=bw)
    return obs, null, p, co, ari


sf = season_habits(b, A["season"])
all_cols = [c for c in HABITS if c in sf.columns]

st.markdown(
    "Each manager is described by ten draft habits, such as how far they reach ahead of ESPN's rankings and when they take "
    "their first goalie. Managers with similar habits are joined by strong links in a **similarity graph**. The eigenvectors "
    "of that graph's **Laplacian matrix** give coordinates for a map where similar drafters sit close together. Clustering "
    "those coordinates finds the style groups. Change the settings below and every chart on this page updates."
)
with st.container(border=True):
    cols = st.pills("Habits used", all_cols, selection_mode="multi", default=all_cols, format_func=HABITS.get, key="sp_cols") or []
    c2, c3 = st.columns(2)
    bw = c2.slider("Kernel width (x median distance)", 0.3, 3.0, 1.0, 0.1, key="sp_bw",
                   help="Smaller = only very similar managers stay linked; larger = everyone looks alike.")
    k_pick = c3.segmented_control("Number of groups", ["Auto", 2, 3, 4, 5], default="Auto", key="sp_k",
                                  help="Auto picks the number of groups at the largest eigengap.")
if len(cols) < 2:
    st.info("Pick at least two habits.")
    st.stop()

feats = sf.groupby("manager")[cols].mean()
res = spectral_clusters(feats, k=None if k_pick in (None, "Auto") else int(k_pick), bandwidth_scale=bw)
groups = name_clusters(res)
gname = {m: groups[int(c)]["name"] for m, c in zip(feats.index, res.labels)}
gorder = [groups[c]["name"] for c in sorted(groups)]
gcolor = {g: CAT[i % len(CAT)] for i, g in enumerate(gorder)}
n = len(feats)
vals = res.eigenvalues
gap = vals[res.k] - vals[res.k - 1]
obs, null, p_perm, co, ari = structure_tests(sf, tuple(cols), bw, res.k, A["season"])

with st.container(horizontal=True):
    st.metric("Managers (graph nodes)", n, border=True)
    st.metric("Style groups", res.k, border=True, help="Chosen at the largest eigengap unless you fixed it above.")
    st.metric("Eigengap", f"{gap:.3f}", border=True, help=f"lambda{res.k + 1} - lambda{res.k}: bigger = clearer groups.")
    st.metric("Permutation p-value", f"{p_perm:.3f}", border=True, help="Chance of a gap this large if managers had no real styles.")
    st.metric("Bootstrap stability (ARI)", f"{np.median(ari):.2f}", border=True, help="1 = the same groups every resample, 0 = no better than random.")

section = st.segmented_control("Section", ["Similarity graph", "Laplacian spectrum", "Groups and habits", "Is the structure real?"],
                               default="Similarity graph", key="sp_section", label_visibility="collapsed")

Z = ((feats - feats.mean()) / feats.std(ddof=0).replace(0, 1)).fillna(0)
W = res.affinity
ids = feats.index.tolist()
labels = [who(m) for m in ids]

# ======================================================================================================
if section == "Similarity graph" or section is None:
    st.markdown("### 1. The similarity graph")
    st.latex(r"w_{ij} = \exp\!\left(-\frac{\lVert z_i - z_j \rVert^2}{2\sigma^2}\right), \qquad \sigma = %.2f" % res.sigma)
    st.caption("z = standardised habits. Each link's strength w runs from 0 (nothing alike) to 1 (identical habits).")
    off = W[np.triu_indices(n, 1)]
    thr = st.slider("Show links stronger than", 0.0, float(np.round(off.max(), 2)), float(np.round(np.quantile(off, 0.6), 2)), 0.01, key="sp_thr")
    x, y = res.embedding[:, 0], res.embedding[:, 1]
    fig = go.Figure()
    for i in range(n):
        for j in range(i + 1, n):
            if W[i, j] >= thr:
                s = (W[i, j] - thr) / max(off.max() - thr, 1e-9)
                fig.add_scatter(x=[x[i], x[j]], y=[y[i], y[j]], mode="lines", line=dict(width=0.6 + 4 * s, color=rgba("#8A8A8A", 0.25 + 0.55 * s)),
                                hoverinfo="skip", showlegend=False)
    top3 = [", ".join(f"{labels[j]} ({W[i, j]:.2f})" for j in np.argsort(-W[i])[:3]) for i in range(n)]
    px_, py_ = 0.18 * max(np.ptp(x), 1e-9), 0.1 * max(np.ptp(y), 1e-9)
    xr, yr = (x.min() - px_, x.max() + px_), (y.min() - py_, y.max() + py_)
    tpos = label_positions(x, y, labels, xr, yr, width_px=950, height_px=460)
    for g in gorder:
        idx = [i for i, m in enumerate(ids) if gname[m] == g]
        fig.add_scatter(x=x[idx], y=y[idx], mode="markers+text", name=g, text=[labels[i] for i in idx], textposition=[tpos[i] for i in idx],
                        textfont=dict(color=INK, size=12), customdata=[ids[i] for i in idx],
                        marker=dict(size=20, color=gcolor[g], line=dict(color="white", width=2)),
                        hovertext=[f"<b>{labels[i]}</b><br>{g}<br>most similar: {top3[i]}" for i in idx], hoverinfo="text")
    fig.update_layout(title="Similarity graph on the spectral map (click a manager)", xaxis=dict(visible=False, range=list(xr)),
                      yaxis=dict(visible=False, range=list(yr)),
                      legend=dict(orientation="h", y=-0.05))
    ev = fig_show(fig, 520, key="sp_graph", on_select="rerun")
    picked = selected_custom(ev) or st.session_state.get("sp_pick") or ids[0]
    st.session_state["sp_pick"] = picked
    note("**How to read this:** each dot is a manager, and its position comes from the second and third eigenvectors of the graph Laplacian. "
         "Lines are links above the threshold, and thicker lines mean more similar habits. Tight bundles of lines are candidate groups. "
         "Slide the threshold up and the graph breaks apart where the links are weakest.")

    c1, c2 = st.columns(2)
    with c1:
        i = ids.index(picked)
        order = [j for j in np.argsort(-W[i]) if j != i]
        fig = go.Figure(go.Bar(x=W[i, order], y=[labels[j] for j in order], orientation="h",
                               marker_color=[gcolor[gname[ids[j]]] if gname[ids[j]] == gname[picked] else FAINT for j in order],
                               hovertemplate="%{y}: similarity %{x:.2f}<extra></extra>"))
        fig.update_layout(title=f"Who drafts most like {labels[i]}? (same group in colour)", xaxis_title="similarity w (0 to 1)",
                          yaxis=dict(autorange="reversed"))
        fig_show(fig, 60 + 24 * len(order))
    with c2:
        ordering = st.segmented_control("Order the matrix by", ["Group", "Fiedler vector", "Name"], default="Group", key="sp_order")
        if ordering == "Name":
            o = np.argsort(labels)
        elif ordering == "Fiedler vector":
            o = np.argsort(res.fiedler)
        else:
            o = np.lexsort((res.fiedler, res.labels))
        Wd = W.copy()
        np.fill_diagonal(Wd, np.nan)
        fig = go.Figure(go.Heatmap(z=Wd[np.ix_(o, o)], x=[labels[j] for j in o], y=[labels[j] for j in o], colorscale=SEQ, zmin=0, zmax=1,
                                   colorbar=dict(title="w", thickness=10), hovertemplate="%{y} and %{x}<br>similarity %{z:.2f}<extra></extra>",
                                   xgap=1, ygap=1))
        fig.update_layout(title="Affinity matrix W (darker = more alike)", yaxis=dict(autorange="reversed"), xaxis=dict(tickangle=-50))
        fig_show(fig, 520)
    st.caption("Ordering the matrix by group should show dark blocks along the diagonal if the groups are real. Faint, patchy blocks mean weak structure.")

# ======================================================================================================
elif section == "Laplacian spectrum":
    st.markdown("### 2. The Laplacian spectrum")
    st.latex(r"L = I - D^{-1/2} W D^{-1/2}, \qquad D_{ii} = \textstyle\sum_j w_{ij}, \qquad L v = \lambda v")
    st.markdown(
        "The normalised Laplacian's eigenvalues always start at 0. If the graph had **k** separate groups with no links between them, "
        "exactly **k** eigenvalues would be 0. Real data is noisier, so the method looks for a **gap**: a few small eigenvalues, then a jump. "
        "The number of eigenvalues before the largest jump is the number of groups (the *eigengap heuristic*)."
    )
    c1, c2 = st.columns(2)
    with c1:
        idx = np.arange(1, n + 1)
        fig = go.Figure()
        fig.add_vrect(x0=res.k + 0.05, x1=res.k + 0.95, fillcolor=rgba(CAT[1], 0.12), line_width=0)
        fig.add_scatter(x=idx, y=vals, mode="lines+markers", line=dict(color=CAT[0], width=2), marker=dict(size=9, color=CAT[0]),
                        hovertemplate="lambda%{x}: %{y:.3f}<extra></extra>", showlegend=False)
        fig.add_annotation(x=res.k + 0.5, y=vals[res.k], text=f"gap chosen: k = {res.k}", showarrow=False, yshift=18, font=dict(color=INK))
        fig.update_layout(title="Eigenvalues of L, smallest first", xaxis_title="index", yaxis_title="eigenvalue", xaxis=dict(dtick=1))
        fig_show(fig, 340)
    with c2:
        gaps = np.diff(vals)
        fig = go.Figure(go.Bar(x=np.arange(1, n), y=gaps, marker_color=[CAT[1] if i + 1 == res.k else FAINT for i in range(n - 1)],
                               hovertemplate="lambda%{x}+1 - lambda%{x}: %{y:.3f}<extra></extra>"))
        fig.update_layout(title="Gaps between neighbouring eigenvalues", xaxis_title="k (gap after the k-th eigenvalue)", yaxis_title="gap",
                          xaxis=dict(dtick=1))
        fig_show(fig, 340)
    note(f"**What stands out:** the biggest gap of all is the first one (from 0 to {vals[1]:.2f}). The graph looks like **one loose group** more than "
         f"several tight ones. The method only considers k of 2 or more, and among those the largest gap is after k = {res.k}, at just {gap:.3f}. "
         "The last section tests whether a gap that small could come from chance.")

    st.markdown("#### How the kernel width changes the answer")
    sweep = np.round(np.arange(0.3, 3.01, 0.1), 2)
    rows = []
    for s_ in sweep:
        r_ = spectral_clusters(feats, bandwidth_scale=float(s_), n_init=3)
        rows.append({"bw": s_, **{f"lambda{i + 1}": r_.eigenvalues[i] for i in range(1, min(5, n))}, "k": r_.k})
    sw = pd.DataFrame(rows)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        for i, col in enumerate([c for c in sw.columns if c.startswith("lambda")]):
            fig.add_scatter(x=sw.bw, y=sw[col], mode="lines", name=col.replace("lambda", "lambda "), line=dict(color=CAT[i % len(CAT)], width=2),
                            hovertemplate=f"{col}: %{{y:.3f}} at width %{{x}}<extra></extra>")
        fig.add_vline(x=bw, line_color=MUTED)
        fig.update_layout(title="Small eigenvalues as the kernel widens (grey line = current setting)", xaxis_title="kernel width (x median distance)",
                          yaxis_title="eigenvalue", legend=dict(orientation="h", y=-0.25))
        fig_show(fig, 340)
    with c2:
        st.markdown(
            "With a narrow kernel, only near-identical managers stay linked, so the eigenvalues spread out and groups look sharper. "
            "With a wide kernel, everyone is linked to everyone and all the eigenvalues bunch up near 1. The app uses the median distance "
            "(width 1.0), a standard default."
        )
        st.dataframe(sw[["bw", "k"]].drop_duplicates("k").rename(columns={"bw": "from width", "k": "groups chosen"}), hide_index=True)

    st.markdown("#### The Fiedler vector: the main axis of difference")
    c1, c2 = st.columns(2)
    fv = res.fiedler
    corr = pd.Series({HABITS[c]: spearmanr(fv, feats[c], nan_policy="omit")[0] for c in cols}).sort_values()
    with c1:
        o = np.argsort(fv)
        fig = go.Figure(go.Bar(x=fv[o], y=[labels[j] for j in o], orientation="h", marker_color=[CAT[0] if v > 0 else CAT[1] for v in fv[o]],
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(title="Managers along the Fiedler vector (second eigenvector)", xaxis_title="Fiedler value")
        fig_show(fig, 60 + 24 * n)
    with c2:
        fig = go.Figure(go.Bar(x=corr.values, y=corr.index, orientation="h", marker_color=[CAT[0] if v > 0 else CAT[1] for v in corr.values],
                               hovertemplate="%{y}: rank correlation %{x:.2f}<extra></extra>"))
        fig.add_vline(x=0, line_color=INK, line_width=1)
        fig.update_layout(title="What the axis measures: correlation with each habit", xaxis_title="rank correlation", xaxis=dict(range=[-1, 1]))
        fig_show(fig, 60 + 24 * len(corr))
    hi, lo = corr.idxmax(), corr.idxmin()
    note(f"**Reading the axis:** managers at the blue end score high on **{hi.lower()}** (rank correlation {corr.max():+.2f}); managers at the rust end score "
         f"high on **{lo.lower()}** ({corr.min():+.2f}). The sign of an eigenvector is arbitrary, so only the ordering matters.")

    if n > 3:
        st.markdown("#### Three-dimensional spectral map (drag to rotate, hover for names)")
        U = np.linalg.eigh(np.eye(n) - np.diag(1 / np.sqrt(W.sum(1))) @ W @ np.diag(1 / np.sqrt(W.sum(1))))[1][:, 1:4]
        fig = go.Figure()
        for g in gorder:
            idx = [i for i, m in enumerate(ids) if gname[m] == g]
            fig.add_scatter3d(x=U[idx, 0], y=U[idx, 1], z=U[idx, 2], mode="markers", name=g, text=[labels[i] for i in idx],
                              marker=dict(size=8, color=gcolor[g], line=dict(color="white", width=1)),
                              hovertemplate="%{text}<extra>" + g + "</extra>")
        fig.update_layout(scene=dict(xaxis_title="eigenvector 2", yaxis_title="eigenvector 3", zaxis_title="eigenvector 4",
                                     xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False), zaxis=dict(showticklabels=False)),
                          legend=dict(orientation="h", y=0))
        fig_show(fig, 560)

# ======================================================================================================
elif section == "Groups and habits":
    st.markdown("### 3. The groups and what drives them")
    st.markdown("k-means on the row-normalised first k eigenvectors assigns each manager to a group (Ng, Jordan and Weiss, 2002). "
                "Each group's name comes from its two most distinctive habits compared with the league.")
    for g_id in sorted(groups):
        g = groups[g_id]
        with st.container(border=True):
            st.markdown(f"**{g['name']}**: {', '.join(g['traits'])}.  \nMembers: {', '.join(who(m) for m in g['members'])}.")
    o = np.lexsort((res.fiedler, res.labels))
    zc = Z.iloc[o]
    raw = feats.iloc[o]
    fig = go.Figure(go.Heatmap(z=zc.to_numpy().clip(-2.5, 2.5), x=[HABITS[c] for c in cols], y=[f"{labels[j]} · {gname[ids[j]]}" for j in o],
                               colorscale=DIV, zmid=0, zmin=-2.5, zmax=2.5, xgap=1, ygap=1, colorbar=dict(title="z", thickness=10),
                               customdata=raw.to_numpy(), hovertemplate="%{y}<br>%{x}: %{customdata:.2f} (z = %{z:.2f})<extra></extra>"))
    fig.update_layout(title="Each manager's habits vs the league average (blue = above, rust = below)", yaxis=dict(autorange="reversed"), xaxis=dict(tickangle=-35))
    fig_show(fig, 80 + 28 * n)

    prof = Z.groupby(pd.Series(res.labels, index=Z.index).map(lambda c: groups[int(c)]["name"])).mean().T
    prof.index = [HABITS[c] for c in prof.index]
    fig = go.Figure()
    for g in gorder:
        if g in prof:
            fig.add_scatter(x=prof[g], y=prof.index, mode="markers", name=g, marker=dict(size=12, color=gcolor[g], line=dict(color="white", width=2)),
                            hovertemplate="%{y}: z = %{x:.2f}<extra>" + g + "</extra>")
    fig.add_vline(x=0, line_color=INK, line_width=1)
    fig.update_layout(title="Group profiles: average habit, in standard deviations from the league", xaxis_title="z-score (0 = league average)",
                      legend=dict(orientation="h", y=-0.15))
    fig_show(fig, 80 + 30 * len(cols))
    note("**How to read the profiles:** a dot far from zero is a habit that sets the group apart. When groups overlap on most habits, "
         "the split rests on one or two habits only.")

# ======================================================================================================
elif section == "Is the structure real?":
    st.markdown("### 4. Is the structure real?")
    st.markdown(
        "Clustering always returns groups, even for random data. Two checks ask whether these groups mean anything."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Test 1. Permutation test of the eigengap**")
        fig = go.Figure(go.Histogram(x=null, nbinsx=40, marker_color=rgba(CAT[0], 0.55), marker_line=dict(color="white", width=1),
                                     hovertemplate="gap %{x}<br>%{y} shuffles<extra></extra>", name="shuffled leagues"))
        fig.add_vline(x=obs, line_color=CAT[1], line_width=2, annotation_text=f"your league: {obs:.3f}", annotation_position="top right")
        fig.update_layout(title="Largest eigengap in 1,000 shuffled leagues", xaxis_title="eigengap", yaxis_title="shuffles", showlegend=False, bargap=0.02)
        fig_show(fig, 340)
        st.markdown(
            "Each habit is shuffled among the managers on its own. That keeps every habit's spread but breaks any link between habits, "
            f"which is what a style would be. **{(null >= obs).mean():.0%}** of shuffled leagues show a gap at least as large "
            f"(p = {p_perm:.3f})."
        )
    with c2:
        st.markdown("**Test 2. Bootstrap stability**")
        o = np.lexsort((res.fiedler, res.labels))
        co_o = co.loc[[ids[j] for j in o], [ids[j] for j in o]].to_numpy()
        fig = go.Figure(go.Heatmap(z=co_o, x=[labels[j] for j in o], y=[labels[j] for j in o], colorscale=SEQ, zmin=0, zmax=1, xgap=1, ygap=1,
                                   colorbar=dict(title="share", thickness=10, tickformat=".0%"),
                                   hovertemplate="%{y} and %{x}<br>same group in %{z:.0%} of resamples<extra></extra>"))
        fig.update_layout(title="How often two managers land in the same group (200 resamples)", yaxis=dict(autorange="reversed"), xaxis=dict(tickangle=-50))
        fig_show(fig, 420)
        st.markdown(
            "Each resample redraws every manager's seasons with replacement and clusters again. Solid dark blocks mean stable groups. "
            f"Agreement with the full-data groups (adjusted Rand index): median **{np.median(ari):.2f}**, middle 80% "
            f"{np.quantile(ari, 0.1):.2f} to {np.quantile(ari, 0.9):.2f}."
        )
    runs = load_runs(1)
    h4 = next((r for r in runs[0]["results"] if r["id"] == "H4"), None) if runs else None
    verdict_real = p_perm < 0.05 and np.median(ari) > 0.6
    note(
        ("**Verdict: the groups look real.** " if verdict_real else "**Verdict: weak structure.** ")
        + f"Permutation p = {p_perm:.3f} and median stability {np.median(ari):.2f}. "
        + ("" if verdict_real else "The style groups are tendencies, not fixed types: a manager near a group boundary can switch groups with one more season of data. ")
        + (f"This agrees with experiment H4: pooling the pick model by these groups made held-out predictions "
           f"{'slightly worse' if h4['test']['ci_high'] < 0 else 'no better'} ({h4['test']['mean_diff']:+.4f} log-likelihood per pick, "
           f"95% CI {h4['test']['ci_low']:+.4f} to {h4['test']['ci_high']:+.4f}), so the simulator does not use them. " if h4 else "")
        + "The groups are kept as scouting descriptions on the League history page."
    )
    with st.expander("Technical details"):
        st.markdown(
            "- Features: per-season habits averaged over each manager's seasons, standardised to z-scores.\n"
            "- Graph: Gaussian kernel, bandwidth = median pairwise distance x the width setting; no self-loops.\n"
            "- Test 1 statistic: the largest gap lambda(k+1) - lambda(k) for k = 2..4 (the quantity the eigengap rule maximises). "
            "Null: 1,000 independent column permutations. One-sided p-value with the +1 correction.\n"
            "- Test 2: 200 bootstrap resamples of seasons within manager, clustering with the same k, co-assignment frequency, "
            "and adjusted Rand index (ARI) vs the full-data labels. ARI of 1 means identical partitions; 0 means chance agreement.\n"
            "- With about 15 managers and 2 to 3 seasons each, the power to detect weak styles is limited. A non-significant result means "
            "the data cannot confirm styles, not that styles don't exist."
        )
