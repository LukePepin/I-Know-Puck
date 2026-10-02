"""Overview: what the project is, how it works in four steps, and what I learned."""

import streamlit as st
from ui import AUTHOR, csv, js, note

meta = js("meta.json")
tests = {t["id"]: t for t in js("tests.json")}
teams = csv("teams_2027.csv")
me = teams[teams.team == AUTHOR].iloc[0]

st.markdown("# I Know Puck")
st.markdown("#### A fantasy hockey draft assistant that picks the player most likely to help you win each week")
st.markdown(
    f"In a fantasy hockey league, {meta['teams']} friends take turns drafting real NHL players. Every week your players' real goals, assists, "
    "saves and hits turn into points, and whoever scores more that week wins. This project turns that draft into a math problem: "
    "**at each pick, which player gives my team the best chance of winning a typical week?** It combines ten seasons of NHL statistics, "
    "a model of how the other managers draft, and thousands of simulated drafts, and every claim it makes is tested on seasons it has not seen."
)

k1, k2, k3, k4 = st.columns(4)
h1 = tests.get("H1")
with k1:
    st.metric("Closer than ESPN's projections", f"{h1['diff']:.1f} pts per player" if h1 else "n/a", border=True,
              help="How many fewer fantasy points our season projection misses by, on average, than ESPN's. Tested on 2024-26 seasons the model never saw.")
k2.metric("Time to suggest a pick", "~2 seconds", border=True, help="ESPN gives 45 seconds per pick.")
k3.metric("My 2027 team, projected", f"#{int(me['rank'])} of {meta['teams']}", border=True, help="By expected weekly points of the starting lineup.")
k4.metric("Wins a typical week", f"{me.win_chance:.0%}", border=True, help="Chance to beat a typical opponent, averaged over the other 11 rosters.")

st.markdown("### How it works, in four steps")
st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent"; nodesep=0.3;
  node [shape=box, style="rounded,filled", fillcolor="#F4F3EF", color="#2A6BB0", fontname="Georgia", fontsize=12, margin="0.2,0.1"];
  edge [color="#555555"];
  a [label="1. Data\\n10 NHL seasons +\\n3 seasons of league drafts"];
  b [label="2. Projections\\nhow many points\\neach player will score"];
  c [label="3. Win chance\\nturn a roster into\\nP(win a week)"];
  d [label="4. Simulate\\nplay out the draft\\nthousands of times"];
  a -> b -> c -> d;
}
""", width="stretch")
c1, c2 = st.columns(2)
with c1:
    st.markdown(
        "1. **Data.** ESPN's player stats, projections and average draft position (ADP) for 2018-2027, advanced stats from MoneyPuck, and "
        "every pick from the league's 2024-26 drafts.\n"
        "2. **Projections.** A player's last three seasons, weighted toward the most recent, pulled toward the average for his position, "
        "then blended with ESPN's projection and with where the market drafts him."
    )
with c2:
    st.markdown(
        f"3. **Win chance.** Only the {sum(v for k, v in meta['lineup'].items() if k not in ('BN', 'IR'))} players in the starting lineup score each week. "
        "The model fills the lineup optimally, estimates the team's weekly points and how much they swing, and turns that into a chance of beating each opponent.\n"
        "4. **Simulate.** For each candidate pick, it plays the rest of the draft forward many times, with the other managers picking the way they "
        "usually do, and suggests the player whose simulated teams win the most weeks."
    )

st.markdown("### What I learned")
h2a = tests.get("H2a")
note(
    "- **Trust the market a little.** Always taking the player our model liked most compared with where he was ranked lost to simply following "
    "the rankings" + (f" ({100 * h2a['diff']:+.0f} percentage points of weekly win chance in the backtest)" if h2a else "") + ". Choosing the biggest gap "
    "between two noisy numbers picks the model's own mistakes (the winner's curse).\n"
    "- **Test everything, keep the failures.** Several ideas sounded smart and did not survive testing on unseen seasons: per-manager pick models, "
    "grouping managers by drafting style, and buying players on hot NHL teams. They are reported, not hidden.\n"
    "- **Read the rules.** The league locks lineups weekly, so bench players rarely score. Measuring that from real weekly scores changed how much "
    "the model values depth."
)

st.markdown("### Tools and methods")
st.markdown(
    "Python, pandas, NumPy, SciPy, scikit-learn, Streamlit and Plotly. Methods: optimal lineup assignment (Hungarian algorithm), Monte Carlo simulation "
    "with common random numbers, a conditional logit model of opponents' picks, graph spectral clustering, and paired permutation tests with "
    "Holm correction."
)
st.caption(f"Data snapshot {meta['snapshot']}. Sources: " + "; ".join(meta["sources"]) + ". Other managers are anonymized; no league credentials are included. "
           "Use the menu on the left to explore.")
