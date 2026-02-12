#!/usr/bin/env python3
"""
=================================================================
Coverage Shell Deception: Detecting & Quantifying Defensive
Disguise Using NFL Ball-in-Air Tracking Data
=================================================================

NFL Big Data Bowl 2026 — Analytics Submission

Author: [Your Name]
Data: NFL Next Gen Stats tracking data, 2023-2024 seasons
      14,108 pass plays | 272 games | 1,384 players

SUMMARY
-------
This project detects and quantifies coverage disguise — when defenses
show one coverage shell pre-throw but execute a different scheme once
the ball is in the air.

Using a Random Forest model (80% AUC) trained on 43 engineered features,
we identify the key pre-throw signals that predict disguise and produce:

  - A Disguise Score metric per game/team
  - A Safety Deception Score per player
  - Coaching-actionable insights on when and how disguise occurs

KEY FINDINGS
------------
1. 17.6% of pass plays show coverage disguise
2. Safety minimum depth from LOS is the #1 predictor of disguise
3. 1-high shells disguise at 2x the rate of 2-high shells (31% vs 15%)
4. Safeties within 3 yards of the hash disguise at 23.5% vs 13.1%
5. Disguise does NOT hurt pursuit efficiency — disguising safeties
   actually close MORE distance on the ball (5.3 yds vs 3.4 yds)

METHODOLOGY
-----------
Follows the Dom Borsani approach: interpretable, feature-driven,
coach-relevant, actionable.

  - Random Forest with OOB validation (not deep learning)
  - Variable importance as the primary insight mechanism
  - Focus on safeties as the key rotation players
  - Geometric rotation detection (not hand-labeled)

PIPELINE
--------
  01_data_prep.py          → Load, standardize, identify positions
  02_shell_classification.py → Classify defensive shells (1/2/0-high)
  03_feature_engineering.py  → 43 features: structure, deltas, pursuit
  04_disguise_model.py       → Random Forest + variable importance
  05_metrics_and_rankings.py → Disguise Score, Safety Deception Score
  06_visualizations.py       → Field plots, leaderboards, distributions
  07_insights.py             → Coaching report and actionable findings

USAGE
-----
    python main.py
"""

import os
import sys
import time

# Ensure we're in the project root
os.chdir(os.path.dirname(os.path.abspath(__file__)))


def run_step(step_name, module_path):
    """Run a pipeline step with timing."""
    print(f"\n{'='*60}")
    print(f"  STEP: {step_name}")
    print(f"{'='*60}")
    start = time.time()

    # Import and run the module's main block
    import importlib.util
    spec = importlib.util.spec_from_file_location("step", module_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    elapsed = time.time() - start
    print(f"\n  [{step_name}] completed in {elapsed:.1f}s")


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  Coverage Shell Deception: Detecting & Quantifying      ║
    ║  Defensive Disguise Using Ball-in-Air Tracking Data     ║
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
    print(f"    Model predictions:   data/model_predictions.pkl")
    print(f"    Feature importance:   data/feature_importance.pkl")
    print(f"    Safety rankings:     data/safety_stats.pkl")
    print(f"    Game rankings:       data/game_metrics.pkl")


if __name__ == "__main__":
    main()
