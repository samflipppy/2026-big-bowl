"""
Step 11: Football Insights — The Coaching Report
==================================================
Translates the closing angle thesis into actionable football language.

This is what goes in the game-plan binder, not the academic paper.
"""

import pandas as pd
import numpy as np
import os


def angle_diff(a, b):
    d = a - b
    return ((d + 180) % 360 - 180).abs()


def detect_snap_frames(input_df):
    wr = input_df[input_df["player_role"].isin(["Targeted Receiver", "Other Route Runner"])]
    avg_speed = wr.groupby(["game_id", "play_id", "frame_id"])["s"].mean().reset_index()
    snap_frames = (
        avg_speed[avg_speed["s"] > 0.5]
        .groupby(["game_id", "play_id"])["frame_id"]
        .min()
        .reset_index()
        .rename(columns={"frame_id": "snap_frame"})
    )
    return snap_frames


def classify_routes(input_df, frame_refs):
    """Classify target receiver routes into football categories."""
    snap_df = detect_snap_frames(input_df)

    target = input_df[input_df["player_role"] == "Targeted Receiver"]
    tgt_snap = target.merge(snap_df, on=["game_id", "play_id"])
    tgt_at_snap = tgt_snap[tgt_snap["frame_id"] == tgt_snap["snap_frame"]][
        ["game_id", "play_id", "x", "y"]
    ].rename(columns={"x": "wr_snap_x", "y": "wr_snap_y"})

    tgt_throw = target.merge(
        frame_refs[["game_id", "play_id", "throw_frame"]], on=["game_id", "play_id"]
    )
    tgt_throw = tgt_throw[tgt_throw["frame_id"] == tgt_throw["throw_frame"]]
    tgt_throw_pos = tgt_throw[["game_id", "play_id", "x", "y"]].rename(
        columns={"x": "wr_throw_x", "y": "wr_throw_y"}
    )

    routes = tgt_at_snap.merge(tgt_throw_pos, on=["game_id", "play_id"], how="inner")
    routes["route_depth"] = routes["wr_throw_x"] - routes["wr_snap_x"]
    routes["route_lateral"] = (routes["wr_throw_y"] - routes["wr_snap_y"]).abs()

    routes["route_type"] = "intermediate"
    routes.loc[routes["route_depth"] < 5, "route_type"] = "short/screen"
    routes.loc[routes["route_depth"] > 15, "route_type"] = "deep"
    routes.loc[
        (routes["route_lateral"] > 8) & (routes["route_depth"] < 10),
        "route_type",
    ] = "crossing/out"

    return routes[["game_id", "play_id", "route_type", "route_depth", "route_lateral"]]


def generate_coaching_report(input_df, output_df, frame_refs, closing_df):
    """The full coaching report."""

    nearest = closing_df[closing_df["is_nearest"]].copy()
    nearest["wrong_way"] = nearest["closing_angle"] >= 60

    routes = classify_routes(input_df, frame_refs)
    nearest = nearest.merge(routes, on=["game_id", "play_id"], how="left")

    print()
    print("=" * 65)
    print("  COACHING REPORT: PURSUIT ANGLES IN PASS DEFENSE")
    print("  \"Distance lies, direction doesn't.\"")
    print("=" * 65)

    # ------------------------------------------------------------------
    # INSIGHT 1: The pursuit angle concept
    # ------------------------------------------------------------------
    print("""
  CONCEPT: PURSUIT ANGLES ON PASS PLAYS
  ──────────────────────────────────────
  Coaches teach pursuit angles on run plays — take the right angle
  to the ball carrier or he runs past you. The same physics apply
  during ball flight on pass plays.

  When the QB releases the ball, the nearest defender is either:
    (a) Trailing the receiver toward the catch point, or
    (b) Moving sideways / wrong direction and can't recover

  The data shows this is BINARY — not a spectrum:""")

    below = nearest[~nearest["wrong_way"]]
    above = nearest[nearest["wrong_way"]]
    print(f"    Good angle (<60°):  +{below['closure'].mean():.1f} yds closure, "
          f"{below['arrived'].mean()*100:.1f}% arrive at catch")
    print(f"    Wrong angle (≥60°): {above['closure'].mean():.1f} yds closure, "
          f"{above['arrived'].mean()*100:.1f}% arrive at catch")
    print(f"\n    → {above.shape[0] / nearest.shape[0] * 100:.0f}% of the time, "
          f"the nearest defender is already beaten at release.")

    # ------------------------------------------------------------------
    # INSIGHT 2: Press vs off
    # ------------------------------------------------------------------
    print("""
  INSIGHT 1: PRESS COVERAGE ELIMINATES WRONG ANGLES
  ──────────────────────────────────────────────────""")

    snap_df = detect_snap_frames(input_df)
    target = input_df[input_df["player_role"] == "Targeted Receiver"]
    tgt_snap = target.merge(snap_df, on=["game_id", "play_id"])
    tgt_at_snap = tgt_snap[tgt_snap["frame_id"] == tgt_snap["snap_frame"]][
        ["game_id", "play_id", "x", "y"]
    ].rename(columns={"x": "wr_snap_x", "y": "wr_snap_y"})

    def_at_snap = input_df[input_df["player_role"] == "Defensive Coverage"].merge(
        snap_df, on=["game_id", "play_id"]
    )
    def_snap_pos = def_at_snap[def_at_snap["frame_id"] == def_at_snap["snap_frame"]]
    def_snap_pos = def_snap_pos.merge(tgt_at_snap, on=["game_id", "play_id"], how="inner")
    def_snap_pos["snap_cushion"] = np.sqrt(
        (def_snap_pos["x"] - def_snap_pos["wr_snap_x"]) ** 2
        + (def_snap_pos["y"] - def_snap_pos["wr_snap_y"]) ** 2
    )
    snap_nearest = def_snap_pos.loc[
        def_snap_pos.groupby(["game_id", "play_id"])["snap_cushion"].idxmin()
    ]
    snap_info = snap_nearest[["game_id", "play_id", "nfl_id", "snap_cushion"]].copy()

    nearest_cush = nearest.merge(snap_info, on=["game_id", "play_id", "nfl_id"], how="inner")

    for lo, hi, label in [(0, 3, "Press (0-3 yds)"), (3, 6, "Off (3-6 yds)"),
                           (6, 10, "Soft (6-10 yds)")]:
        mask = (nearest_cush["snap_cushion"] >= lo) & (nearest_cush["snap_cushion"] < hi)
        if mask.sum() > 50:
            wrong = nearest_cush.loc[mask, "wrong_way"].mean() * 100
            arrive = nearest_cush.loc[mask, "arrived"].mean() * 100
            n = mask.sum()
            print(f"    {label:20s}: {wrong:>4.1f}% wrong-way, {arrive:>4.1f}% arrive (n={n:,})")

    print("""
    Press defenders are 5x less likely to have a wrong pursuit angle
    than off-coverage defenders. They're already in the receiver's
    hip pocket — there's no angle to get wrong.""")

    # ------------------------------------------------------------------
    # INSIGHT 2: LB vs CB closing
    # ------------------------------------------------------------------
    print("""
  INSIGHT 2: LINEBACKERS CAN'T CLOSE — EVEN FROM CLOSER
  ──────────────────────────────────────────────────────
  At the same starting distance (5-8 yards from catch point):""")

    for pos in ["CB", "SS", "FS", "ILB", "OLB", "MLB"]:
        mask = (nearest["player_position"] == pos) & (nearest["start_dist"] >= 5) & (nearest["start_dist"] < 8)
        if mask.sum() > 30:
            arr = nearest.loc[mask, "arrived"].mean() * 100
            n = mask.sum()
            print(f"    {pos:5s}: {arr:>5.1f}% arrive at catch point (n={n:,})")

    print("""
    A CB at 5-8 yards arrives 3x more often than an MLB at the same
    distance. The mechanism: CBs have better hip fluidity, closing
    speed, and ball-tracking ability. LBs are built for different jobs.

    COACHING ACTION: When you identify LB-on-WR in the secondary,
    that's your money throw. The LB won't close during ball flight.""")

    # ------------------------------------------------------------------
    # INSIGHT 3: Route-specific
    # ------------------------------------------------------------------
    print("""
  INSIGHT 3: CROSSING ROUTES ARE THE HARDEST TO CLOSE ON
  ──────────────────────────────────────────────────────""")

    for rt in ["short/screen", "crossing/out", "intermediate", "deep"]:
        mask = nearest["route_type"] == rt
        if mask.sum() > 100:
            wrong = nearest.loc[mask, "wrong_way"].mean() * 100
            arr = nearest.loc[mask, "arrived"].mean() * 100
            n = mask.sum()
            print(f"    {rt:18s}: {wrong:>4.1f}% wrong-way, {arr:>5.1f}% arrive (n={n:,})")

    print("""
    Short/screen and crossing routes produce the most wrong-way
    defenders. The lateral movement forces the defense into bad
    angles. Crossing routes have the LOWEST arrival rate because
    the defender must change direction 90+ degrees during ball flight.

    COACHING ACTION: Crossing routes naturally exploit closing angles.
    Combine them with play-action (freeze LBs) for maximum effect.""")

    # ------------------------------------------------------------------
    # INSIGHT 4: Player rankings
    # ------------------------------------------------------------------
    print("""
  INSIGHT 4: INDIVIDUAL CLOSING ABILITY RANKINGS
  ──────────────────────────────────────────────""")

    player_stats = nearest.groupby(
        ["nfl_id", "player_name", "player_position"]
    ).agg(
        n=("arrived", "count"),
        arrival_rate=("arrived", "mean"),
        wrong_way_rate=("wrong_way", "mean"),
        avg_closure=("closure", "mean"),
    ).reset_index()
    player_stats = player_stats[player_stats["n"] >= 50]

    print("  BEST CLOSERS (highest arrival %, min 50 plays):")
    for _, row in player_stats.nlargest(5, "arrival_rate").iterrows():
        print(f"    {row.player_name:25s} ({row.player_position}): "
              f"{row.arrival_rate*100:.1f}% arrive, {row.avg_closure:+.1f} yds closure")

    print("\n  MOST EXPLOITABLE (highest wrong-way %, min 50 plays):")
    for _, row in player_stats.nlargest(5, "wrong_way_rate").iterrows():
        print(f"    {row.player_name:25s} ({row.player_position}): "
              f"{row.wrong_way_rate*100:.1f}% wrong-way, {row.arrival_rate*100:.1f}% arrive")

    # ------------------------------------------------------------------
    # THE ONE-LINER
    # ------------------------------------------------------------------
    print(f"""
  {'=' * 65}
  THE ONE-LINER FOR YOUR GAME-PLAN BINDER
  {'=' * 65}

  "If the nearest defender's feet aren't pointed at your target
   when you release, throw it. He's not coming."

  The ball travels faster than a defender can redirect. During a
  typical 1.1-second ball flight, a defender on a bad angle drifts
  FURTHER from the catch point — not closer. 27% of the time, the
  nearest defender is already beaten the moment the QB lets go.

  SCOUTING APPLICATION:
  • Grade defenders on closing angle, not just proximity
  • Target LBs in coverage — they close at half the rate of CBs
  • Press corners are 5x harder to beat than off corners
  • Crossing routes naturally produce the worst closing angles
  • Individual players have consistent closing tendencies —
    scout for it on film, exploit it on game day
""")

    # Save player rankings
    player_stats.to_pickle("data/player_closing_rankings.pkl")
    print("  Player closing rankings saved to data/player_closing_rankings.pkl")


if __name__ == "__main__":
    input_df = pd.read_pickle("data/input_processed.pkl")
    output_df = pd.read_pickle("data/output_processed.pkl")
    frame_refs = pd.read_pickle("data/frame_refs.pkl")
    closing_df = pd.read_pickle("data/closing_angle.pkl")

    generate_coaching_report(input_df, output_df, frame_refs, closing_df)
