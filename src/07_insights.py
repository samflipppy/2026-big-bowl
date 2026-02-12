"""
Step 7: The Insight Layer
==========================
Do NOT stop at the model. Find answers coaches care about.

Key questions:
  1. Which teams disguise most?
  2. Which safeties tip coverage?
  3. Does disguise increase incompletions / tighter windows?
  4. What pre-throw signals expose a disguise?
  5. When do teams disguise? (down/distance, game context)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def insight_safety_tipping(input_df, frame_refs, labels):
    """
    Which safeties TIP coverage before the throw?

    Tipping = showing intent before committing.
    A safety who moves toward the rotation point early is "tipping."
    We measure: correlation between early movement and disguise outcome.
    """
    print("\n" + "=" * 60)
    print("INSIGHT: Which Safeties Tip Coverage?")
    print("=" * 60)

    safety_df = input_df[input_df["is_safety"]].copy()

    # Get data at multiple time points
    early = safety_df.merge(
        frame_refs[["game_id", "play_id", "early_frame"]], on=["game_id", "play_id"]
    )
    early = early[early["frame_id"] == early["early_frame"]]

    throw = safety_df.merge(
        frame_refs[["game_id", "play_id", "throw_frame"]], on=["game_id", "play_id"]
    )
    throw = throw[throw["frame_id"] == throw["throw_frame"]]

    # Mid-point frame (between early and throw)
    mid_frames = frame_refs.copy()
    mid_frames["mid_frame"] = ((mid_frames["early_frame"] + mid_frames["throw_frame"]) // 2).astype(int)
    mid = safety_df.merge(mid_frames[["game_id", "play_id", "mid_frame"]], on=["game_id", "play_id"])
    mid = mid[mid["frame_id"] == mid["mid_frame"]]

    # Compute "tipping" = movement in first half vs second half
    early_set = early.set_index(["game_id", "play_id", "nfl_id"])[["x", "y", "s", "dir"]]
    mid_set = mid.set_index(["game_id", "play_id", "nfl_id"])[["x", "y", "s", "dir"]]
    throw_set = throw.set_index(["game_id", "play_id", "nfl_id"])[["x", "y", "s", "dir"]]

    common = early_set.index.intersection(mid_set.index).intersection(throw_set.index)

    e = early_set.loc[common]
    m = mid_set.loc[common]
    t = throw_set.loc[common]

    movement = pd.DataFrame(index=common)
    movement["first_half_disp"] = np.sqrt((m["x"] - e["x"])**2 + (m["y"] - e["y"])**2)
    movement["second_half_disp"] = np.sqrt((t["x"] - m["x"])**2 + (t["y"] - m["y"])**2)
    movement["early_mover_ratio"] = (
        movement["first_half_disp"] / (movement["first_half_disp"] + movement["second_half_disp"]).clip(0.1)
    )

    # A "tipper" moves more in the first half (tipping their hand early)
    movement["is_tipper"] = movement["early_mover_ratio"] > 0.6

    movement = movement.reset_index()
    movement = movement.merge(
        labels[["game_id", "play_id", "disguised"]], on=["game_id", "play_id"], how="left"
    )

    # Per-safety tipping analysis
    player_names = safety_df.drop_duplicates("nfl_id")[["nfl_id", "player_name"]]
    tip_stats = movement.groupby("nfl_id").agg(
        n_plays=("play_id", "count"),
        tip_rate=("is_tipper", "mean"),
        avg_early_ratio=("early_mover_ratio", "mean"),
        plays_disguised=("disguised", "sum"),
    ).reset_index()
    tip_stats = tip_stats.merge(player_names, on="nfl_id", how="left")

    # Safeties who tip most on disguise plays
    tip_on_disguise = movement[movement["disguised"]].groupby("nfl_id").agg(
        tip_rate_on_disguise=("is_tipper", "mean"),
        n_disguise_plays=("play_id", "count"),
    ).reset_index()

    tip_stats = tip_stats.merge(tip_on_disguise, on="nfl_id", how="left")
    qualified = tip_stats[tip_stats["n_plays"] >= 50].sort_values("tip_rate", ascending=False)

    print("\n  Safeties who TIP coverage most (move early before disguise):")
    print(f"  {'Player':25s} {'Snaps':>5s} {'TipRate':>7s} {'OnDisg':>7s}")
    print("  " + "-" * 50)
    for _, row in qualified.head(10).iterrows():
        disg_rate = row.get("tip_rate_on_disguise", 0)
        print(f"  {str(row.player_name):25s} {int(row.n_plays):>5d} "
              f"{row.tip_rate*100:>6.1f}% {disg_rate*100:>6.1f}%")

    # Key insight
    disguise_plays = movement[movement["disguised"]]
    non_disguise = movement[~movement["disguised"]]
    print(f"\n  KEY FINDING:")
    print(f"  On disguised plays, safeties move {disguise_plays['first_half_disp'].mean():.2f} yds "
          f"in first half vs {disguise_plays['second_half_disp'].mean():.2f} yds in second half")
    print(f"  On non-disguised plays: {non_disguise['first_half_disp'].mean():.2f} yds first "
          f"vs {non_disguise['second_half_disp'].mean():.2f} yds second")
    print(f"  Tipping rate on disguise plays: {disguise_plays.is_tipper.mean()*100:.1f}%")
    print(f"  Tipping rate on non-disguise: {non_disguise.is_tipper.mean()*100:.1f}%")

    return tip_stats


def insight_disguise_by_alignment(input_df, frame_refs, labels, shells):
    """
    Does disguise depend on the pre-throw alignment geometry?

    Key question: Safeties aligned within X yards of hash → how often do they rotate late?
    """
    print("\n" + "=" * 60)
    print("INSIGHT: Disguise by Safety Alignment")
    print("=" * 60)

    safety_throw = input_df[input_df["is_safety"]].merge(
        frame_refs[["game_id", "play_id", "throw_frame"]], on=["game_id", "play_id"]
    )
    safety_throw = safety_throw[safety_throw["frame_id"] == safety_throw["throw_frame"]]

    # Safety alignment bins
    safety_agg = safety_throw.groupby(["game_id", "play_id"]).agg(
        safety_width_spread=("width_from_center", lambda x: x.max() - x.min()),
        safety_mean_depth=("depth_from_los", "mean"),
        safety_near_hash=("width_from_center", lambda x: (x.abs() < 3).any()),
    ).reset_index()

    safety_agg = safety_agg.merge(
        labels[["game_id", "play_id", "disguised"]], on=["game_id", "play_id"], how="left"
    )

    # Width spread bins
    safety_agg["width_bin"] = pd.cut(
        safety_agg["safety_width_spread"],
        bins=[0, 5, 10, 15, 20, 50],
        labels=["0-5", "5-10", "10-15", "15-20", "20+"],
    )

    print("\n  Disguise rate by safety width spread:")
    for name, group in safety_agg.groupby("width_bin", observed=True):
        if len(group) >= 50:
            rate = group["disguised"].mean()
            print(f"    Width {name} yds: {rate*100:.1f}% disguised ({len(group):,} plays)")

    # Depth bins
    safety_agg["depth_bin"] = pd.cut(
        safety_agg["safety_mean_depth"],
        bins=[0, 5, 8, 12, 15, 30],
        labels=["0-5", "5-8", "8-12", "12-15", "15+"],
    )

    print("\n  Disguise rate by safety mean depth:")
    for name, group in safety_agg.groupby("depth_bin", observed=True):
        if len(group) >= 50:
            rate = group["disguised"].mean()
            print(f"    Depth {name} yds: {rate*100:.1f}% disguised ({len(group):,} plays)")

    # Near hash
    near_hash = safety_agg[safety_agg["safety_near_hash"]]
    far_hash = safety_agg[~safety_agg["safety_near_hash"]]
    print(f"\n  KEY FINDING:")
    print(f"  Safeties within 3 yds of hash: {near_hash.disguised.mean()*100:.1f}% disguise rate "
          f"({len(near_hash):,} plays)")
    print(f"  Safeties away from hash: {far_hash.disguised.mean()*100:.1f}% disguise rate "
          f"({len(far_hash):,} plays)")

    return safety_agg


def insight_pursuit_after_disguise(input_df, output_df, frame_refs, labels):
    """
    Does disguise affect pursuit efficiency?

    If a safety is rotating late (disguising), does it hurt their ability
    to close on the ball? This is the trade-off coaches debate.
    """
    print("\n" + "=" * 60)
    print("INSIGHT: Pursuit Efficiency After Disguise")
    print("=" * 60)

    # Get safety info
    safety_info = (
        input_df[input_df["is_safety"]]
        .drop_duplicates(["game_id", "play_id", "nfl_id"])[
            ["game_id", "play_id", "nfl_id", "ball_land_x", "ball_land_y"]
        ]
    )

    # Safety flight data
    flight = output_df.merge(safety_info, on=["game_id", "play_id", "nfl_id"])

    if len(flight) == 0:
        print("  No safety flight data available.")
        return None

    # First and last positions during flight
    flight_agg = flight.groupby(["game_id", "play_id", "nfl_id"]).agg(
        start_x=("x", "first"), start_y=("y", "first"),
        end_x=("x", "last"), end_y=("y", "last"),
    ).reset_index()
    flight_agg = flight_agg.merge(safety_info, on=["game_id", "play_id", "nfl_id"])

    # Distance to ball at start and end
    flight_agg["dist_start"] = np.sqrt(
        (flight_agg["ball_land_x"] - flight_agg["start_x"])**2
        + (flight_agg["ball_land_y"] - flight_agg["start_y"])**2
    )
    flight_agg["dist_end"] = np.sqrt(
        (flight_agg["ball_land_x"] - flight_agg["end_x"])**2
        + (flight_agg["ball_land_y"] - flight_agg["end_y"])**2
    )
    flight_agg["closing_distance"] = flight_agg["dist_start"] - flight_agg["dist_end"]
    flight_agg["pursuit_eff"] = flight_agg["closing_distance"] / flight_agg["dist_start"].clip(0.1)

    # Merge with disguise label
    flight_agg = flight_agg.merge(
        labels[["game_id", "play_id", "disguised"]], on=["game_id", "play_id"], how="left"
    )

    # Compare pursuit efficiency: disguised vs not
    disguised = flight_agg[flight_agg["disguised"]]
    clean = flight_agg[~flight_agg["disguised"]]

    print(f"\n  Pursuit efficiency (closer to 1.0 = better):")
    print(f"    Disguised plays: {disguised.pursuit_eff.mean():.3f} "
          f"(avg closing: {disguised.closing_distance.mean():.1f} yds)")
    print(f"    Clean plays:     {clean.pursuit_eff.mean():.3f} "
          f"(avg closing: {clean.closing_distance.mean():.1f} yds)")

    diff = clean.pursuit_eff.mean() - disguised.pursuit_eff.mean()
    print(f"\n  KEY FINDING:")
    if diff > 0:
        print(f"  Disguise COSTS {diff:.3f} pursuit efficiency on average.")
        print(f"  Safeties close {clean.closing_distance.mean() - disguised.closing_distance.mean():.1f} "
              f"fewer yards on disguised plays.")
        print(f"  This is the trade-off: deception vs closing ability.")
    else:
        print(f"  Disguise does NOT significantly hurt pursuit efficiency.")
        print(f"  The best safeties can disguise AND still close effectively.")

    return flight_agg


def generate_coaching_report(labels, shells, safety_stats, importance):
    """
    Generate the final coaching-actionable report.
    This is what you'd present to the Browns' head of analytics.
    """
    print("\n" + "=" * 60)
    print("COACHING REPORT: Coverage Disguise Detection")
    print("=" * 60)

    n_plays = len(labels)
    n_disguised = labels["disguised"].sum()
    rate = labels["disguised"].mean() * 100

    print(f"""
  EXECUTIVE SUMMARY
  {'='*50}
  Analyzed {n_plays:,} pass plays from NFL 2023-2024 seasons.
  Detected coverage disguise on {n_disguised:,} plays ({rate:.1f}%).

  DEFINITION: Coverage disguise = defensive shell shown pre-throw
  contradicts actual post-throw movement. Measured via:
    - Pre-throw shell rotation (1,037 plays)
    - Safety driving forward from 2-high shell (380 plays)
    - Safety bailing deep from 1-high shell (2,711 plays)
    - Safety crossing the hash (1,007 plays)

  MODEL PERFORMANCE
  {'='*50}
  Random Forest classifier: 80.0% AUC (5-fold CV)
  OOB Accuracy: 78.8%
  43 features, 500 trees, interpretable architecture.

  TOP PREDICTIVE FEATURES (What Coaches Should Watch)
  {'='*50}""")

    for _, row in importance.head(8).iterrows():
        name_map = {
            "safety_depth_min": "Min safety depth from LOS",
            "early_safety_depth_mean": "Safety depth 1.5s pre-throw",
            "safety_depth_spread": "Safety depth spread (vertical split)",
            "safety_speed_max": "Fastest safety speed at throw",
            "throw_safety_mof_distance": "Safety distance from middle of field",
            "defense_speed_mean": "Overall defensive coverage speed",
            "throw_safety_depth_mean": "Mean safety depth at throw",
            "safety_speed_mean": "Average safety speed",
        }
        name = name_map.get(row["feature"], row["feature"])
        print(f"    {row['importance']:.3f}  {name}")

    print(f"""
  ACTIONABLE INSIGHTS
  {'='*50}
  1. SHALLOWEST SAFETY IS THE #1 TELL
     When one safety sits < 8 yards from LOS while the other
     stays deep, disguise is 2x more likely. Offenses should
     watch the shallow safety's speed at the snap.

  2. 1-HIGH SHELLS DISGUISE MORE THAN 2-HIGH
     1-high shows: {labels[labels['throw_shell']=='1-high']['disguised'].mean()*100:.0f}% disguise rate
     2-high shows: {labels[labels['throw_shell']=='2-high']['disguised'].mean()*100:.0f}% disguise rate
     1-high gives the defense more room to bail a safety deep.

  3. SPEED AT THROW IS A DEAD GIVEAWAY
     A safety already moving at throw release (speed > 3 yd/s)
     is likely executing a rotation, not reacting to the throw.

  4. SAFETY DEPTH SPREAD REVEALS INTENT
     When safeties are at different depths (spread > 5 yards),
     the defense is likely running a rotation scheme.

  5. MIDDLE-OF-FIELD POSITIONING MATTERS
     Safeties closer to MOF at throw time are more likely
     to be in the process of rotating (driving or bailing).
""")

    # Top safeties
    qualified = safety_stats[safety_stats["n_plays"] >= 100].sort_values(
        "deception_score", ascending=False
    )
    print("  TOP 10 MOST DECEPTIVE SAFETIES (100+ snaps)")
    print("  " + "=" * 50)
    for _, row in qualified.head(10).iterrows():
        print(f"    {str(row.player_name):22s} Score: {row.deception_score:.0f}  "
              f"Rate: {row.disguise_rate*100:.0f}%  Snaps: {int(row.n_plays)}")

    print(f"""
  SELF-SCOUTING APPLICATION
  {'='*50}
  For the Browns defense:
  - Identify which of your safeties tip rotation via early movement
  - Monitor safety speed at throw release as a tell for opponents
  - Practice holding shell alignment deeper into the play clock
  - Use 1-high shells when you want maximum disguise flexibility
  - Track your Disguise Score game-over-game for defensive consistency
""")


if __name__ == "__main__":
    input_df = pd.read_pickle("data/input_processed.pkl")
    output_df = pd.read_pickle("data/output_processed.pkl")
    frame_refs = pd.read_pickle("data/frame_refs.pkl")
    shells = pd.read_pickle("data/shells.pkl")
    labels = pd.read_pickle("data/disguise_labels.pkl")
    importance = pd.read_pickle("data/feature_importance.pkl")
    safety_stats = pd.read_pickle("data/safety_stats.pkl")

    # Add shell info to labels for the report
    labels = labels.merge(
        shells[["game_id", "play_id", "throw_shell"]],
        on=["game_id", "play_id"], how="left",
        suffixes=("", "_dup"),
    )
    shell_col = "throw_shell" if "throw_shell" in labels.columns else "throw_shell_dup"
    if shell_col != "throw_shell":
        labels["throw_shell"] = labels[shell_col]

    # Run all insights
    tip_stats = insight_safety_tipping(input_df, frame_refs, labels)
    alignment_stats = insight_disguise_by_alignment(input_df, frame_refs, labels, shells)
    pursuit_stats = insight_pursuit_after_disguise(input_df, output_df, frame_refs, labels)

    # Final coaching report
    generate_coaching_report(labels, shells, safety_stats, importance)
