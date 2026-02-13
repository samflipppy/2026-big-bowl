"""
Step 12: The CB Depth Tell
===========================
Using 2025 BDB data (2022 season, pre-snap tracking) to find
which cornerbacks leak their coverage assignment through alignment depth.

KEY FINDING
-----------
Cornerback depth from the LOS at the snap is a binary man/zone classifier
for most NFL defenses, but the degree of "tell" varies enormously by team.

The 5 most readable CBs in the 2022 season:
  1. Charvarius Ward (SF):  man=1.4y, zone=6.4y, gap=+4.9y
     - Pressed (<3y) → man 79% of the time (n=95)
     - Off (>=5y) → zone 95% of the time (n=107)
  2. Patrick Surtain (DEN): man=3.7y, zone=8.5y, gap=+4.8y
  3. Alontae Taylor (NO):   man=1.6y, zone=6.2y, gap=+4.6y
  4. Carlton Davis (TB):    man=2.9y, zone=7.3y, gap=+4.4y
  5. Jamel Dean (TB):       man=2.7y, zone=7.0y, gap=+4.3y

The best disguisers (CBs whose depth does NOT reveal coverage):
  - Desmond King (HOU):   man=4.2y, zone=4.0y, gap=-0.3y
  - Derek Stingley (HOU): man=5.5y, zone=5.3y, gap=-0.2y
  - Steven Nelson (HOU):  man=4.6y, zone=4.7y, gap=+0.1y
  All three Houston CBs within 0.3 yards — a coaching philosophy.

TEAM-LEVEL READABLE RANKINGS (Zone depth - Man depth):
  Most readable:  NO (+4.5y), TB (+4.3y), SF (+3.7y)
  Best disguisers: HOU (-0.2y), KC (+1.2y), BUF (+1.7y)

ACTIONABLE GAME PLAN:
  vs SF: Watch Ward. Pressed = man → call SLANT/CROSS.
         Bailed out to 5+ = zone → call CURL/COMEBACK.
  vs HOU: Cannot read CBs. Read SAFETIES instead
          (safety speed at snap separates deep-half rotation
          from static assignments).

ROUTE PERFORMANCE vs PRESSED MAN CBs:
  SLANT:  30.5% target rate, 63% catch rate, 10.2 avg yards
  CROSS:  25.0% target rate, 60.5% catch rate, 15.1 avg yards
  GO:     20.3% target rate, 34.4% catch rate, 24.6 avg yards
"""

import os
import pandas as pd
import numpy as np


DATA_2025 = os.path.join("data", "2025", "nfl-big-data-bowl-2025")


def load_snap_frames():
    """Load tracking data at the snap frame for all 9 weeks."""
    all_snap = []
    for week in range(1, 10):
        path = os.path.join(DATA_2025, f"tracking_week_{week}.csv")
        tr = pd.read_csv(path)
        snap = tr[tr["frameType"] == "SNAP"]
        snap = (
            snap.sort_values("frameId")
            .groupby(["gameId", "playId", "nflId"])
            .first()
            .reset_index()
        )
        all_snap.append(snap)
        del tr
    return pd.concat(all_snap, ignore_index=True)


def load_ball_positions():
    """Get the football's x/y at the snap for each play."""
    ball_positions = []
    for week in range(1, 10):
        path = os.path.join(DATA_2025, f"tracking_week_{week}.csv")
        tr = pd.read_csv(path)
        football = tr[
            (tr["displayName"] == "football") & (tr["frameType"] == "SNAP")
        ]
        football = (
            football.sort_values("frameId")
            .groupby(["gameId", "playId"])
            .first()
            .reset_index()
        )
        ball_positions.append(
            football[["gameId", "playId", "x", "y"]].rename(
                columns={"x": "ball_x", "y": "ball_y"}
            )
        )
        del tr
    return pd.concat(ball_positions, ignore_index=True)


def build_cb_snap_data():
    """Build the full CB snap dataset with depth calculations."""
    plays = pd.read_csv(os.path.join(DATA_2025, "plays.csv"))
    player_play = pd.read_csv(os.path.join(DATA_2025, "player_play.csv"))
    players = pd.read_csv(os.path.join(DATA_2025, "players.csv"))

    print("Loading snap frames (all 9 weeks)...")
    tracking_snap = load_snap_frames()

    print("Loading ball positions...")
    ball_df = load_ball_positions()

    # Filter to CBs
    db_positions = ["CB"]
    dbs = players[players["position"].isin(db_positions)]
    cb_snap = tracking_snap.merge(dbs[["nflId", "position"]], on="nflId", how="inner")

    # Coverage assignments
    cov = player_play[
        ["gameId", "playId", "nflId", "pff_defensiveCoverageAssignment"]
    ].dropna(subset=["pff_defensiveCoverageAssignment"])
    cb_snap = cb_snap.merge(cov, on=["gameId", "playId", "nflId"], how="inner")

    # Play info
    cb_snap = cb_snap.merge(
        plays[
            [
                "gameId",
                "playId",
                "defensiveTeam",
                "pff_manZone",
                "pff_passCoverage",
                "expectedPointsAdded",
                "isDropback",
            ]
        ],
        on=["gameId", "playId"],
    )
    cb_snap = cb_snap[cb_snap["isDropback"] == True]

    # Ball position
    cb_snap = cb_snap.merge(ball_df, on=["gameId", "playId"], how="left")

    # Depth from LOS
    cb_snap["depth"] = np.where(
        cb_snap["playDirection"] == "right",
        cb_snap["x"] - cb_snap["ball_x"],
        cb_snap["ball_x"] - cb_snap["x"],
    )

    # Man/zone labels
    cb_snap["is_man"] = (
        cb_snap["pff_defensiveCoverageAssignment"] == "MAN"
    ).astype(int)
    zone_assignments = [
        "2R", "2L", "3R", "3M", "3L", "4OR", "4OL", "4IR", "4IL",
        "FR", "FL", "HCR", "HCL", "CFR", "CFL", "HOL", "DF", "PRE",
    ]
    cb_snap["assign_type"] = "other"
    cb_snap.loc[
        cb_snap["pff_defensiveCoverageAssignment"] == "MAN", "assign_type"
    ] = "man"
    cb_snap.loc[
        cb_snap["pff_defensiveCoverageAssignment"].isin(zone_assignments),
        "assign_type",
    ] = "zone"

    return cb_snap, players


def analyze_individual_cb_tells(cb_snap, players):
    """Find which CBs leak their coverage through alignment depth."""
    cb_profile = (
        cb_snap.groupby(["nflId", "club"])
        .apply(
            lambda g: pd.Series(
                {
                    "man_depth": g[g["is_man"] == 1]["depth"].mean(),
                    "zone_depth": g[g["assign_type"] == "zone"]["depth"].mean(),
                    "man_n": (g["is_man"] == 1).sum(),
                    "zone_n": (g["assign_type"] == "zone").sum(),
                    "man_press_rate": (
                        ((g["is_man"] == 1) & (g["depth"] < 3)).sum()
                        / max((g["is_man"] == 1).sum(), 1)
                    ),
                    "zone_press_rate": (
                        ((g["assign_type"] == "zone") & (g["depth"] < 3)).sum()
                        / max((g["assign_type"] == "zone").sum(), 1)
                    ),
                }
            )
        )
        .reset_index()
    )
    cb_profile = cb_profile.merge(players[["nflId", "displayName"]], on="nflId")
    cb_profile["depth_gap"] = cb_profile["zone_depth"] - cb_profile["man_depth"]
    cb_profile["press_gap"] = (
        cb_profile["man_press_rate"] - cb_profile["zone_press_rate"]
    )

    # Filter: enough snaps in both
    cb_profile = cb_profile[
        (cb_profile["man_n"] >= 30) & (cb_profile["zone_n"] >= 30)
    ]

    return cb_profile.sort_values("depth_gap", ascending=False)


def analyze_team_cb_tells(cb_snap):
    """Aggregate CB depth tells at the team level."""
    return (
        cb_snap.groupby("club")
        .apply(
            lambda g: pd.Series(
                {
                    "man_depth": g[g["is_man"] == 1]["depth"].mean(),
                    "zone_depth": g[g["assign_type"] == "zone"]["depth"].mean(),
                    "depth_gap": (
                        g[g["assign_type"] == "zone"]["depth"].mean()
                        - g[g["is_man"] == 1]["depth"].mean()
                    ),
                    "man_press_rate": (
                        ((g["is_man"] == 1) & (g["depth"] < 3)).sum()
                        / max((g["is_man"] == 1).sum(), 1)
                    ),
                    "zone_press_rate": (
                        ((g["assign_type"] == "zone") & (g["depth"] < 3)).sum()
                        / max((g["assign_type"] == "zone").sum(), 1)
                    ),
                }
            )
        )
        .sort_values("depth_gap", ascending=False)
    )


def game_plan_report(cb_snap, players, team_abbr):
    """Generate a scouting report for CBs on a specific defense."""
    team_cbs = cb_snap[cb_snap["club"] == team_abbr]

    print(f"\n{'=' * 60}")
    print(f"SCOUTING REPORT: {team_abbr} Cornerbacks")
    print(f"{'=' * 60}")

    for nfl_id in team_cbs["nflId"].unique():
        sub = team_cbs[team_cbs["nflId"] == nfl_id]
        name = sub["displayName"].iloc[0]
        man_sub = sub[sub["is_man"] == 1]
        zone_sub = sub[sub["assign_type"] == "zone"]
        if len(man_sub) < 15 or len(zone_sub) < 15:
            continue

        gap = zone_sub["depth"].mean() - man_sub["depth"].mean()
        pressed = sub[sub["depth"] < 3]
        off = sub[sub["depth"] >= 5]
        man_when_pressed = pressed["is_man"].mean() if len(pressed) > 10 else None
        zone_when_off = (
            (off["assign_type"] == "zone").mean() if len(off) > 10 else None
        )

        print(f"\n  {name}")
        print(
            f"    Man depth: {man_sub['depth'].mean():.1f}y | "
            f"Zone depth: {zone_sub['depth'].mean():.1f}y | "
            f"Gap: {gap:+.1f}y"
        )
        if man_when_pressed is not None:
            print(
                f"    Pressed (<3y) → man {man_when_pressed:.0%} "
                f"(n={len(pressed)})"
            )
        if zone_when_off is not None:
            print(
                f"    Off (>=5y)    → zone {zone_when_off:.0%} "
                f"(n={len(off)})"
            )

        if gap > 3.0:
            print(f"    ** READABLE: {gap:+.1f}y gap. Alignment IS the call. **")
        elif abs(gap) < 1.0:
            print(f"    ** DISGUISED: only {gap:+.1f}y gap. Cannot read. **")


def main():
    print("Building CB snap dataset from 2025 BDB data (2022 season)...\n")
    cb_snap, players = build_cb_snap_data()
    print(f"Total CB snap records on dropbacks: {len(cb_snap)}")

    print("\n" + "=" * 60)
    print("INDIVIDUAL CB DEPTH TELLS (sorted by gap)")
    print("=" * 60)
    cb_tells = analyze_individual_cb_tells(cb_snap, players)
    print(
        f"\n{'Name':<25} {'Team':<5} {'ManDep':>7} {'ZoneDep':>8} "
        f"{'Gap':>6} {'ManPress':>9} {'ZnPress':>8}"
    )
    print("-" * 70)
    for _, r in cb_tells.head(10).iterrows():
        print(
            f"{r['displayName']:<25} {r['club']:<5} "
            f"{r['man_depth']:>6.1f}y {r['zone_depth']:>7.1f}y "
            f"{r['depth_gap']:>+5.1f}y {r['man_press_rate']:>8.0%} "
            f"{r['zone_press_rate']:>7.0%}"
        )
    print("...")
    for _, r in cb_tells.tail(5).iterrows():
        print(
            f"{r['displayName']:<25} {r['club']:<5} "
            f"{r['man_depth']:>6.1f}y {r['zone_depth']:>7.1f}y "
            f"{r['depth_gap']:>+5.1f}y {r['man_press_rate']:>8.0%} "
            f"{r['zone_press_rate']:>7.0%}"
        )

    print("\n" + "=" * 60)
    print("TEAM CB DEPTH TELLS")
    print("=" * 60)
    team_tells = analyze_team_cb_tells(cb_snap)
    print(
        f"\n{'Team':<5} {'ManDep':>7} {'ZoneDep':>8} "
        f"{'Gap':>6} {'ManPress':>9} {'ZnPress':>8}"
    )
    print("-" * 48)
    for team, r in team_tells.iterrows():
        print(
            f"{team:<5} {r['man_depth']:>6.1f}y {r['zone_depth']:>7.1f}y "
            f"{r['depth_gap']:>+5.1f}y {r['man_press_rate']:>8.0%} "
            f"{r['zone_press_rate']:>7.0%}"
        )

    # Game plan examples
    game_plan_report(cb_snap, players, "SF")
    game_plan_report(cb_snap, players, "HOU")


if __name__ == "__main__":
    main()
