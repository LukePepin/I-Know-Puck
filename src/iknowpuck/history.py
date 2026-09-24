"""League-history analysis: injuries, roster moves, trades, pickups, activity, same-team stacking.

Built from ESPN game logs, NHL schedules and the league's transaction log. All results are
descriptive facts about past seasons; association tests use Spearman correlation with bootstrap
intervals and, where noted, control for draft quality.
"""

from __future__ import annotations

import warnings

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

warnings.filterwarnings("ignore", message="An input array is constant")

from .config import LeagueSettings
from .data import league_history as LH
from .data.espn import EspnClient

NHL_ABBREV = {1: "BOS", 2: "BUF", 3: "CGY", 4: "CHI", 5: "DET", 6: "EDM", 7: "CAR", 8: "LAK", 9: "DAL", 10: "MTL", 11: "NJD",
              12: "NYI", 13: "NYR", 14: "OTT", 15: "PHI", 16: "PIT", 17: "COL", 18: "SJS", 19: "STL", 20: "TBL", 21: "TOR",
              22: "VAN", 23: "WSH", 24: "ARI", 25: "ANA", 26: "FLA", 27: "NSH", 28: "WPG", 29: "CBJ", 30: "MIN", 37: "VGK",
              124292: "SEA", 129764: "UTA"}
MIN_SPELL = 3  # consecutive team games missed to count as an absence spell


def game_points(gl: pd.DataFrame, settings: LeagueSettings) -> pd.Series:
    return sum(gl.get(f"g_{c.stat_id}", pd.Series(0.0, index=gl.index)).fillna(0) * c.points for c in settings.scoring_categories)


def sp_dates(sched: pd.DataFrame) -> pd.Series:
    return sched.groupby("scoring_period")["date"].min()


def absences(gl: pd.DataFrame, sched: pd.DataFrame, positions: pd.Series) -> pd.DataFrame:
    """Absence spells for skaters: runs of consecutive team games a player did not appear in.

    Windows around a mid-season NHL trade are skipped (ambiguous). Spells at the very start or end
    of the season are labelled, because they can also mean the player was not in the NHL yet.
    """
    team_games = {t: np.sort(g.scoring_period.unique()) for t, g in sched.groupby("pro_team_id")}
    first_sp, last_sp = sched.scoring_period.min(), sched.scoring_period.max()
    rows = []
    for (season, pid), g in gl.groupby(["season", "player_id"]):
        if positions.get(pid) == "G":
            continue
        g = g.sort_values("scoring_period")
        sp, teams = g.scoring_period.to_numpy(), g.pro_team_id.to_numpy()
        missed: list[tuple[int, int, str]] = []  # (sp, team, where)
        tg = team_games.get(teams[0], np.array([]))
        missed += [(s, teams[0], "season start") for s in tg[(tg >= first_sp) & (tg < sp[0])]]
        for a, b, ta, tb in zip(sp[:-1], sp[1:], teams[:-1], teams[1:]):
            if ta != tb:
                continue
            tg = team_games.get(ta, np.array([]))
            missed += [(s, ta, "mid-season") for s in tg[(tg > a) & (tg < b)]]
        tg = team_games.get(teams[-1], np.array([]))
        missed += [(s, teams[-1], "season end") for s in tg[(tg > sp[-1]) & (tg <= last_sp)]]
        if not missed:
            continue
        missed.sort()
        # group consecutive team games into spells
        run = [missed[0]]
        for m in missed[1:]:
            tg = team_games.get(m[1], np.array([]))
            prev = run[-1]
            between = tg[(tg > prev[0]) & (tg < m[0])]
            if m[1] == prev[1] and m[2] == prev[2] and len(between) == 0:
                run.append(m)
            else:
                rows.append((season, pid, run))
                run = [m]
        rows.append((season, pid, run))
    out = pd.DataFrame(
        [{"season": s, "player_id": p, "pro_team_id": r[0][1], "start_sp": r[0][0], "end_sp": r[-1][0], "games": len(r), "where": r[0][2]} for s, p, r in rows]
    )
    if out.empty:
        return out
    return out[out.games >= MIN_SPELL].reset_index(drop=True)


def availability(gl: pd.DataFrame, spells: pd.DataFrame) -> pd.DataFrame:
    played = gl.groupby(["season", "player_id"]).agg(gp=("scoring_period", "size"))
    miss = spells.groupby(["season", "player_id"]).games.sum().rename("missed") if len(spells) else pd.Series(dtype=float, name="missed")
    out = played.join(miss, how="left").fillna({"missed": 0}).reset_index()
    out["missed_mid"] = out.set_index(["season", "player_id"]).index.map(
        spells[spells["where"] == "mid-season"].groupby(["season", "player_id"]).games.sum() if len(spells) else {}
    )
    out["missed_mid"] = out["missed_mid"].fillna(0)
    out["missed_share"] = out.missed / (out.gp + out.missed)
    return out


def tenure_points(stints: pd.DataFrame, gl: pd.DataFrame) -> pd.DataFrame:
    """Fantasy points each roster stint produced (all games while rostered, bench included)."""
    g = gl[["season", "player_id", "scoring_period", "fp"]]
    m = stints.merge(g, on=["season", "player_id"], how="left")
    m = m[(m.scoring_period >= m.start_sp) & (m.scoring_period < m.end_sp)]
    pts = m.groupby(["season", "team_id", "player_id", "start_sp"]).agg(points=("fp", "sum"), games=("fp", "size")).reset_index()
    out = stints.merge(pts, on=["season", "team_id", "player_id", "start_sp"], how="left").fillna({"points": 0.0, "games": 0})
    return out


@dataclass
class History:
    seasons: list[int]
    owners: pd.DataFrame  # season, team_id, owner_id
    gamelogs: pd.DataFrame
    schedules: pd.DataFrame
    spells: pd.DataFrame
    avail: pd.DataFrame
    stints: pd.DataFrame
    trades: pd.DataFrame
    lineup: pd.DataFrame
    player_names: dict[int, str]
    player_pos: dict[int, str]
    player_team: dict[tuple[int, int], int] = field(default_factory=dict)

    # --- derived tables -----------------------------------------------------------------------
    def owner_of(self, season: int, team_id: int) -> str | None:
        m = self.owners[(self.owners.season == season) & (self.owners.team_id == team_id)]
        return str(m.owner_id.iloc[0]) if len(m) else None

    def with_owner(self, df: pd.DataFrame, team_col: str = "team_id", out_col: str = "owner_id") -> pd.DataFrame:
        o = self.owners.rename(columns={"team_id": team_col, "owner_id": out_col})
        return df.merge(o[["season", team_col, out_col]], on=["season", team_col], how="left")

    def points_by_source(self) -> pd.DataFrame:
        s = self.with_owner(self.stints)
        s["source"] = s["how"].replace({"waiver": "waiver / free agent", "free agent": "waiver / free agent"})
        t = s.pivot_table(index=["season", "owner_id"], columns="source", values="points", aggfunc="sum", fill_value=0).reset_index()
        cols = [c for c in ("draft", "waiver / free agent", "trade") if c in t]
        t["total"] = t[cols].sum(axis=1)
        for c in cols:
            t[f"share_{c}"] = t[c] / t["total"].where(t["total"] > 0)
        return t

    def best_pickups(self, n: int = 15) -> pd.DataFrame:
        s = self.with_owner(self.stints)
        s = s[s.how.isin(["free agent", "waiver"])].nlargest(n * 3, "points")
        s["player"] = s.player_id.map(self.player_names)
        s["pos"] = s.player_id.map(self.player_pos)
        s["added"] = s.apply(lambda r: self._date(r.season, r.start_sp), axis=1)
        return s.sort_values(["season", "points"], ascending=[True, False]).groupby("season").head(n).reset_index(drop=True)

    def _date(self, season: int, sp: int):
        d = self.schedules[self.schedules.season == season]
        m = d[d.scoring_period >= sp]
        return m.date.min().date() if len(m) else None

    def trade_summary(self) -> pd.DataFrame:
        """Each completed trade: what each side received and the points those players scored for
        their new team after the trade (while still rostered)."""
        if self.trades.empty:
            return pd.DataFrame()
        s = self.stints[self.stints.how == "trade"]
        rows = []
        for (season, tx_id), g in self.trades.groupby(["season", "tx_id"]):
            sp = int(g.scoring_period.iloc[0])
            teams = sorted(set(g.from_team) | set(g.to_team))
            side = {}
            for t in teams:
                got = g[g.to_team == t].player_id.tolist()
                pts = s[(s.season == season) & (s.team_id == t) & s.player_id.isin(got) & (s.start_sp == sp)].points.sum()
                side[t] = (got, float(pts))
            if len(teams) != 2:
                continue
            a, b_ = teams
            rows.append({
                "season": season, "tx_id": tx_id, "date": self._date(season, sp),
                "team_a": a, "owner_a": self.owner_of(season, a), "received_a": ", ".join(self.player_names.get(p, str(p)) for p in side[a][0]), "points_a": side[a][1],
                "team_b": b_, "owner_b": self.owner_of(season, b_), "received_b": ", ".join(self.player_names.get(p, str(p)) for p in side[b_][0]), "points_b": side[b_][1],
            })
        out = pd.DataFrame(rows)
        if len(out):
            out["winner"] = np.where(out.points_a >= out.points_b, out.owner_a, out.owner_b)
            out["margin"] = (out.points_a - out.points_b).abs()
        return out

    def activity(self) -> pd.DataFrame:
        s = self.with_owner(self.stints)
        adds = s[s.how.isin(["free agent", "waiver"])].groupby(["season", "owner_id"]).size().rename("pickups")
        trades = pd.concat([
            self.with_owner(self.trades, "from_team").groupby(["season", "owner_id", ]).tx_id.nunique(),
            self.with_owner(self.trades, "to_team").groupby(["season", "owner_id"]).tx_id.nunique(),
        ]).groupby(level=[0, 1]).max().rename("trades") if len(self.trades) else pd.Series(dtype=float, name="trades")
        lu = self.with_owner(self.lineup).groupby(["season", "owner_id"]).size().rename("lineup_moves") if len(self.lineup) else pd.Series(dtype=float, name="lineup_moves")
        out = pd.concat([adds, trades, lu], axis=1).fillna(0).reset_index().rename(columns={"level_0": "season", "level_1": "owner_id"})
        # lineup-move logging differs a lot between seasons on ESPN's side: compare within season only
        out["lineup_moves_pct_rank"] = out.groupby("season").lineup_moves.rank(pct=True)
        out["pickups_pct_rank"] = out.groupby("season").pickups.rank(pct=True)
        return out

    def injury_luck(self, drafts: pd.DataFrame) -> pd.DataFrame:
        """Points each manager lost to absences of the skaters they drafted, relative to the league."""
        ppg = self.gamelogs.groupby(["season", "player_id"]).fp.mean().rename("ppg")
        d = drafts[["season", "owner_id", "player_id"]].merge(ppg.reset_index(), on=["season", "player_id"], how="left")
        d = d.merge(self.avail[["season", "player_id", "missed"]], on=["season", "player_id"], how="left").fillna({"missed": 0, "ppg": 0})
        d = d[d.player_id.map(self.player_pos) != "G"]
        d["lost"] = d.missed * d.ppg
        g = d.groupby(["season", "owner_id"]).agg(games_lost=("missed", "sum"), points_lost=("lost", "sum")).reset_index()
        g["injury_luck"] = g.groupby("season").points_lost.transform("mean") - g.points_lost  # >0 = luckier than average
        return g

    def stacking(self, drafts: pd.DataFrame) -> pd.DataFrame:
        d = drafts[["season", "owner_id", "player_id"]].copy()
        d["nhl"] = [NHL_ABBREV.get(self.player_team.get((s, p)), "?") for s, p in zip(d.season, d.player_id)]
        rows = []
        for (s, o), g in d[d.nhl != "?"].groupby(["season", "owner_id"]):
            vc = g.nhl.value_counts()
            rows.append({"season": s, "owner_id": o, "top_team": vc.index[0], "top_count": int(vc.iloc[0]),
                         "teams_with_3plus": int((vc >= 3).sum()), "picks": len(g)})
        return pd.DataFrame(rows)

    def stacking_null(self, drafts: pd.DataFrame, n: int = 400, seed: int = 0) -> tuple[float, float, float, float]:
        """Observed mean top-team count vs a permutation null that shuffles NHL teams among picks
        within each season (keeps each season's team mix, removes any manager preference)."""
        st = self.stacking(drafts)
        obs = float(st.top_count.mean())
        d = drafts[["season", "owner_id", "player_id"]].copy()
        d["nhl"] = [self.player_team.get((s, p)) for s, p in zip(d.season, d.player_id)]
        d = d.dropna(subset=["nhl"])
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(n):
            perm = d.groupby("season").nhl.transform(lambda x: rng.permutation(x.to_numpy()))
            null.append(d.assign(nhl=perm).groupby(["season", "owner_id"]).nhl.agg(lambda x: x.value_counts().iloc[0]).mean())
        null = np.array(null)
        return obs, float(null.mean()), float(np.quantile(null, 0.975)), float((np.sum(null >= obs) + 1) / (n + 1))


def spearman_ci(x: pd.Series, y: pd.Series, n_boot: int = 3000, seed: int = 0) -> tuple[float, float, float, float, int]:
    m = pd.concat([x, y], axis=1).dropna()
    if len(m) < 8:
        return (np.nan,) * 4 + (len(m),)
    a, b = m.iloc[:, 0].to_numpy(), m.iloc[:, 1].to_numpy()
    r, p = sps.spearmanr(a, b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    boots = [sps.spearmanr(a[i], b[i]).statistic for i in idx]
    lo, hi = np.nanquantile(boots, [0.025, 0.975])
    return float(r), float(lo), float(hi), float(p), len(m)


def partial_residual(y: pd.Series, control: pd.Series) -> pd.Series:
    """Residual of y after a linear fit on the control (used to ask 'beyond draft quality...')."""
    m = pd.concat([y, control], axis=1).dropna()
    X = np.column_stack([np.ones(len(m)), m.iloc[:, 1]])
    beta = np.linalg.lstsq(X, m.iloc[:, 0], rcond=None)[0]
    return pd.Series(m.iloc[:, 0] - X @ beta, index=m.index)


def build_history(client: EspnClient, drafts: pd.DataFrame, panel: pd.DataFrame, settings: LeagueSettings,
                  seasons: list[int], extra_player_ids: list[int] | None = None) -> History:
    owners, gls, scheds, stints, trades, lineups = [], [], [], [], [], []
    for s in seasons:
        t = client.teams(s)
        owners.append(t[["season", "team_id", "owner_id"]])
        tx, items = LH.transactions(client, s)
        st = LH.roster_tenures(drafts, tx, items, s)
        stints.append(st)
        trades.append(LH.executed_trades(tx, items).assign(season=s))
        lineups.append(LH.lineup_activity(tx, items))
        sched = LH.schedules(client, s)
        scheds.append(sched)
        ids = set(st.player_id.astype(int)) | set(int(p) for p in (extra_player_ids or []))
        gls.append(LH.gamelogs(client, s, sorted(ids)))
    gl = pd.concat(gls, ignore_index=True)
    gl["fp"] = game_points(gl, settings)
    sched = pd.concat(scheds, ignore_index=True)
    pn = panel.drop_duplicates("player_id", keep="last").set_index("player_id")
    names, pos = pn["name"].to_dict(), pn["pos"].to_dict()
    team_mode = gl.groupby(["season", "player_id"]).pro_team_id.agg(lambda x: x.mode().iloc[0])
    player_team = {k: int(v) for k, v in team_mode.items()}
    # fall back to ESPN's listed team for drafted players with no game logs
    for r in panel[["season", "player_id", "pro_team_id"]].dropna().itertuples():
        player_team.setdefault((int(r.season), int(r.player_id)), int(r.pro_team_id))
    spell_parts = [absences(gl[gl.season == s], sched[sched.season == s], pd.Series(pos)) for s in seasons]
    spells = pd.concat([p for p in spell_parts if len(p)], ignore_index=True) if any(len(p) for p in spell_parts) else pd.DataFrame()
    avail = availability(gl, spells) if len(spells) else pd.DataFrame()
    stints_df = tenure_points(pd.concat(stints, ignore_index=True), gl)
    return History(
        seasons=seasons, owners=pd.concat(owners, ignore_index=True), gamelogs=gl, schedules=sched, spells=spells, avail=avail,
        stints=stints_df, trades=pd.concat(trades, ignore_index=True), lineup=pd.concat([x for x in lineups if len(x)], ignore_index=True) if any(len(x) for x in lineups) else pd.DataFrame(),
        player_names=names, player_pos=pos, player_team=player_team,
    )
