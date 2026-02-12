#!/usr/bin/env python3
"""
=================================================================
Distance Lies, Direction Doesn't:
How Closing Angle Predicts Ball-in-Air Outcomes
=================================================================

NFL Big Data Bowl 2026 — Analytics Submission

Data: NFL Next Gen Stats tracking data, 2023-2024 seasons
      14,108 pass plays | 272 games | 1,384 players

KEY INSIGHT
-----------
A defender 5 yards from the catch point heading sideways ends up
FURTHER from the ball at the catch than one 12 yards away heading
straight at it. Above a 60-degree closing angle, defenders move
AWAY from the ball during flight — not toward it.

KEY FINDINGS
------------
1. The 60-degree threshold: defenders with a closing angle above 60
   degrees have NEGATIVE closure — they drift further from the catch
   point during ball flight
2. 27% of nearest defenders at the throw are heading the wrong way
3. A defender 12+ yds away on a beeline (end: ~4 yds) beats a
   defender 0-5 yds away at 90+ degrees (end: ~6 yds)
4. LBs are caught going the wrong way far more often than CBs
5. At the same starting distance (5-8 yds), CBs close ~2x more
   often than LBs — speed and hip fluidity are the mechanism
6. The defender's direction at throw release predicts their flight
   closing angle — the QB CAN read this pre-throw

METHODOLOGY
-----------
Interpretable, feature-driven, coach-relevant, actionable.
Follows the Dom Borsani philosophy.

  - Geometric closing angle computation from ball-in-air tracking
  - Distance x Angle interaction analysis
  - Position-specific closing ability comparison
  - Pre-throw predictors of closing angle (QB-readable)
  - Random Forest model for coverage disguise detection

PIPELINE
--------
  01_data_prep.py           -> Load, standardize, identify positions
  02_shell_classification.py -> Classify defensive shells (1/2/0-high)
  03_feature_engineering.py  -> 43 features: structure, deltas, pursuit
  04_disguise_model.py       -> Random Forest + variable importance
  05_metrics_and_rankings.py -> Disguise Score, Safety Deception Score
  06_visualizations.py       -> Field plots, leaderboards, distributions
  07_insights.py             -> Coaching report and actionable findings
  08_qb_reads.py             -> QB pre-snap read analysis
  09_closing_angle.py        -> THE CLOSING ANGLE THESIS (core analysis)
  10_closing_visuals.py      -> Closing angle visualizations

USAGE
-----
    python main.py
"""

import os
import time

# Ensure we're in the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def run_step(step_name, module_path):
    """Run a pipeline step with timing."""
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"{'='*60}")
    start = time.time()

    import importlib.util
    spec = importlib.util.spec_from_file_location("step", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    elapsed = time.time() - start
    print(f"\n  [{step_name}] completed in {elapsed:.1f}s")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  Distance Lies, Direction Doesn't                       ║
    ║  How Closing Angle Predicts Ball-in-Air Outcomes        ║
    ║                                                         ║
    ║  NFL Big Data Bowl 2026 — Analytics Track               ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    total_start = time.time()

    steps = [
        ("Data Preparation", "src/01_data_prep.py"),
        ("Shell Classification", "src/02_shell_classification.py"),
        ("Feature Engineering", "src/03_feature_engineering.py"),
        ("Disguise Model", "src/04_disguise_model.py"),
        ("Metrics & Rankings", "src/05_metrics_and_rankings.py"),
        ("Visualizations", "src/06_visualizations.py"),
        ("Insights & Report", "src/07_insights.py"),
        ("QB Reads", "src/08_qb_reads.py"),
        ("Closing Angle Thesis", "src/09_closing_angle.py"),
        ("Closing Angle Visuals", "src/10_closing_visuals.py"),
    ]

    for name, path in steps:
        run_step(name, path)

    total = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  COMPLETE — Total pipeline time: {total:.0f}s")
    print(f"{'='*60}")
    print(f"\n  Outputs:")
    print(f"    Figures:  output/figures/")
    print(f"    Data:     data/*.pkl")
    print(f"\n  Key files:")
    print(f"    Closing angle data:  data/closing_angle.pkl")
    print(f"    Model predictions:   data/model_predictions.pkl")
    print(f"    Feature importance:  data/feature_importance.pkl")
    print(f"    Safety rankings:     data/safety_stats.pkl")
    print(f"\n  Key insight:")
    print(f"    'Distance lies, direction doesn't.'")
    print(f"    A defender >60 degrees off the catch point drifts")
    print(f"    FURTHER during ball flight, not closer.")


if __name__ == "__main__":
    main()
