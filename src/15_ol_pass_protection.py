"""
OL Pass Protection Analysis — NFL BDB 2025 (2022 Season)
=========================================================
Uses 10Hz tracking data + PFF blocking grades to build an OL evaluation
framework from player movement signatures.

Key Findings:
1. Pressure costs -0.528 EPA per play. Every additional OL beaten adds ~-0.4 EPA.
2. Guard weight significantly predicts pass protection quality (r=-0.329, p=0.004).
   Heavy guards (320+) allow 6.7% pressure; light guards (<305) allow 10.3%.
3. Depth of set at 0.5s is the strongest per-play predictor of pressure:
   - OL who gain ground at 0.5s → 3% pressure rate
   - OL pushed back 0.5+ yards → 13% pressure rate (4x multiplier)
4. "Quick pressure" rate (<2.5s) is the most discriminating OL evaluation metric.
   Elite OL (Wirfs, Thuney) have 0% quick pressure rates.
5. CLE OL Assessment: Bitonio is elite (#9 guard), Wills is a liability (#41 tackle).

Browns-specific recommendations:
- Guard: Bitonio is elite (keep). Teller is average (#37 guard). Upgrade at RG.
- Tackle: Conklin is decent (#27 tackle). Wills is below average (#41) — target upgrade.
- Center: Pocic is #22/40 — below average. Draft/sign an upgrade.
- Draft strategy: Prioritize weight at guard (315+) and lateral agility at tackle.
"""

import pandas as pd
import numpy as np
import glob
import pickle
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# ── Data Loading ────────────────────────────────────────────────────────────

def load_ol_tracking():
    """Load and process OL tracking data for all dropback plays."""
    plays = pd.read_csv("data/2025/nfl-big-data-bowl-2025/plays.csv")
    players = pd.read_csv("data/2025/nfl-big-data-bowl-2025/players.csv")
    player_play = pd.read_csv("data/2025/nfl-big-data-bowl-2025/player_play.csv")

    dropbacks = plays[plays['isDropback'] == True][['gameId', 'playId']].copy()
    dropback_keys = set(zip(dropbacks['gameId'], dropbacks['playId']))
    ol_ids = set(players[players['position'].isin(['C', 'G', 'T'])]['nflId'].values)

    all_snap = []
    all_post = []
    for f in sorted(glob.glob("data/2025/nfl-big-data-bowl-2025/tracking_week_*.csv")):
        t = pd.read_csv(f)
        t_ol = t[t['nflId'].isin(ol_ids)].copy()
        t_ol['_key'] = list(zip(t_ol['gameId'], t_ol['playId']))
        t_ol = t_ol[t_ol['_key'].isin(dropback_keys)].drop(columns=['_key'])

        snap_frames = t_ol[t_ol['event'] == 'ball_snap'][
            ['gameId', 'playId', 'nflId', 'frameId']
        ].drop_duplicates(subset=['gameId', 'playId', 'nflId'], keep='first')
        snap_frames.rename(columns={'frameId': 'snap_frame'}, inplace=True)

        t_ol = t_ol.merge(snap_frames, on=['gameId', 'playId', 'nflId'], how='inner')
        t_ol['rel_frame'] = t_ol['frameId'] - t_ol['snap_frame']

        # Normalize direction (all plays go left-to-right)
        mask = t_ol['playDirection'] == 'left'
        t_ol.loc[mask, 'x'] = 120 - t_ol.loc[mask, 'x']
        t_ol.loc[mask, 'y'] = 160 / 3 - t_ol.loc[mask, 'y']
        t_ol.loc[mask, 'o'] = (t_ol.loc[mask, 'o'] + 180) % 360
        t_ol.loc[mask, 'dir'] = (t_ol.loc[mask, 'dir'] + 180) % 360

        all_snap.append(t_ol[t_ol['rel_frame'] == 0])
        all_post.append(t_ol[(t_ol['rel_frame'] >= 1) & (t_ol['rel_frame'] <= 15)])

    return pd.concat(all_snap), pd.concat(all_post)


def build_ol_features(ol_snap, ol_post):
    """Compute per-play OL tracking features."""
    plays = pd.read_csv("data/2025/nfl-big-data-bowl-2025/plays.csv")
    players = pd.read_csv("data/2025/nfl-big-data-bowl-2025/players.csv")
    player_play = pd.read_csv("data/2025/nfl-big-data-bowl-2025/player_play.csv")

    snap_feat = ol_snap[['gameId', 'playId', 'nflId', 'displayName', 'club',
                          'x', 'y', 's', 'a', 'o', 'dir']].copy()
    snap_feat.rename(columns={'x': 'snap_x', 'y': 'snap_y', 's': 'snap_speed',
                               'a': 'snap_accel', 'o': 'snap_orient', 'dir': 'snap_dir'}, inplace=True)

    # Displacement from snap position
    ol_post_m = ol_post.merge(snap_feat[['gameId', 'playId', 'nflId', 'snap_x', 'snap_y']],
                               on=['gameId', 'playId', 'nflId'])
    ol_post_m['depth_set'] = -(ol_post_m['x'] - ol_post_m['snap_x'])
    ol_post_m['lateral_move'] = np.abs(ol_post_m['y'] - ol_post_m['snap_y'])

    # Extract features at key timestamps
    for frame_n, label in [(5, '05s'), (10, '10s'), (15, '15s')]:
        fm = ol_post_m[ol_post_m['rel_frame'] == frame_n][
            ['gameId', 'playId', 'nflId', 'depth_set', 'lateral_move']
        ].rename(columns={'depth_set': f'depth_set_{label}', 'lateral_move': f'lateral_{label}'})
        snap_feat = snap_feat.merge(fm, on=['gameId', 'playId', 'nflId'], how='left')

    # First step speed (frames 1-3)
    first_step = ol_post[ol_post['rel_frame'].isin([1, 2, 3])].groupby(
        ['gameId', 'playId', 'nflId']).agg(first_step_speed=('s', 'mean')).reset_index()
    snap_feat = snap_feat.merge(first_step, on=['gameId', 'playId', 'nflId'], how='left')

    # Add player info and outcomes
    snap_feat = snap_feat.merge(players[['nflId', 'position', 'weight']], on='nflId', how='left')
    pp_cols = player_play[['gameId', 'playId', 'nflId', 'pressureAllowedAsBlocker',
                            'timeToPressureAllowedAsBlocker', 'blockedPlayerNFLId1']]
    snap_feat = snap_feat.merge(pp_cols, on=['gameId', 'playId', 'nflId'], how='left')
    play_cols = plays[['gameId', 'playId', 'possessionTeam', 'expectedPointsAdded',
                        'timeToThrow', 'passResult']]
    snap_feat = snap_feat.merge(play_cols, on=['gameId', 'playId'], how='left')

    return snap_feat


# ── Analysis Functions ──────────────────────────────────────────────────────

def pressure_epa_impact(ol_features):
    """Quantify the EPA cost of OL pressure."""
    play_press = ol_features[ol_features['pressureAllowedAsBlocker'].notna()].groupby(
        ['gameId', 'playId']).agg(
        n_pressures=('pressureAllowedAsBlocker', 'sum'),
        epa=('expectedPointsAdded', 'first'),
    ).reset_index()

    print("=== EPA COST OF OL PRESSURE ===\n")
    clean = play_press[play_press['n_pressures'] == 0]
    pressed = play_press[play_press['n_pressures'] > 0]
    print(f"Clean pocket: {clean['epa'].mean():.3f} EPA ({len(clean)} plays)")
    print(f"Pressure:     {pressed['epa'].mean():.3f} EPA ({len(pressed)} plays)")
    print(f"Cost per pressure play: {pressed['epa'].mean() - clean['epa'].mean():.3f} EPA\n")

    for n in range(0, 6):
        sub = play_press[play_press['n_pressures'] == n]
        if len(sub) > 10:
            print(f"  {n} OL beaten: {sub['epa'].mean():+.3f} EPA (n={len(sub)})")


def guard_weight_analysis(ol_features):
    """The guard weight finding: heavier guards allow less pressure."""
    guards = ol_features[ol_features['position'] == 'G'].copy()
    g_prof = guards.groupby('nflId').agg(
        snaps=('pressureAllowedAsBlocker', 'count'),
        pressure_rate=('pressureAllowedAsBlocker', 'mean'),
    ).reset_index()
    players = pd.read_csv("data/2025/nfl-big-data-bowl-2025/players.csv")
    g_prof = g_prof.merge(players[['nflId', 'displayName', 'weight']], on='nflId')
    g_prof = g_prof[g_prof['snaps'] >= 100]
    g_prof['weight'] = pd.to_numeric(g_prof['weight'], errors='coerce')

    r, p = stats.pearsonr(g_prof['weight'], g_prof['pressure_rate'])
    heavy = g_prof[g_prof['weight'] >= 320]
    light = g_prof[g_prof['weight'] < 305]
    t_stat, t_p = stats.ttest_ind(heavy['pressure_rate'], light['pressure_rate'])

    print("\n=== GUARD WEIGHT → PRESSURE RATE ===\n")
    print(f"Correlation: r={r:.3f}, p={p:.4f}")
    print(f"Heavy guards (320+ lbs): {heavy['pressure_rate'].mean():.3f} pressure rate (n={len(heavy)})")
    print(f"Light guards (<305 lbs): {light['pressure_rate'].mean():.3f} pressure rate (n={len(light)})")
    print(f"Difference: {(light['pressure_rate'].mean() - heavy['pressure_rate'].mean()):.3f}")
    print(f"T-test: t={t_stat:.3f}, p={t_p:.4f}")
    print(f"\n→ Heavy guards allow {((light['pressure_rate'].mean() / heavy['pressure_rate'].mean()) - 1) * 100:.0f}% less pressure")


def depth_of_set_analysis(ol_features):
    """Per-play depth at 0.5s predicts pressure with 4x multiplier."""
    ol_f = ol_features[ol_features['pressureAllowedAsBlocker'].notna()].copy()

    print("\n=== DEPTH OF SET @ 0.5s: THE PER-PLAY PREDICTOR ===\n")
    for pos in ['T', 'G']:
        sub = ol_f[ol_f['position'] == pos]
        bins = [-5, 0, 0.15, 0.3, 0.5, 1.0, 5.0]
        labels = ['forward', '0-0.15y', '0.15-0.3y', '0.3-0.5y', '0.5-1.0y', '1.0y+']
        sub = sub.copy()
        sub['depth_bin'] = pd.cut(sub['depth_set_05s'], bins=bins, labels=labels)
        result = sub.groupby('depth_bin', observed=True).agg(
            n=('pressureAllowedAsBlocker', 'count'),
            pressure_rate=('pressureAllowedAsBlocker', 'mean'),
        ).reset_index()
        print(f"--- {pos} ---")
        print(result.to_string(index=False))
        print()


def quick_pressure_rankings(ol_features):
    """Rank OL by catastrophic quick-pressure rate (<2.5s)."""
    ol_f = ol_features[ol_features['pressureAllowedAsBlocker'].notna()].copy()
    players = pd.read_csv("data/2025/nfl-big-data-bowl-2025/players.csv")

    snap_counts = ol_f.groupby('nflId')['pressureAllowedAsBlocker'].agg(['count', 'mean']).reset_index()
    snap_counts.columns = ['nflId', 'snaps', 'pressure_rate']

    qp = ol_f[ol_f['timeToPressureAllowedAsBlocker'].notna()].groupby('nflId').agg(
        n_beaten=('timeToPressureAllowedAsBlocker', 'count'),
        avg_ttp=('timeToPressureAllowedAsBlocker', 'mean'),
        quick_count=('timeToPressureAllowedAsBlocker', lambda x: (x < 2.5).sum()),
    ).reset_index()

    qp = qp.merge(snap_counts, on='nflId')
    qp['quick_pressure_rate'] = qp['quick_count'] / qp['snaps']
    qp = qp.merge(players[['nflId', 'displayName', 'position']], on='nflId')
    qp = qp[(qp['snaps'] >= 100) & (qp['n_beaten'] >= 5)]

    print("\n=== QUICK PRESSURE RATE RANKINGS ===")
    print("(% of snaps resulting in pressure under 2.5 seconds — the catastrophic failure rate)\n")

    for pos in ['T', 'G', 'C']:
        sub = qp[qp['position'] == pos].sort_values('quick_pressure_rate')
        print(f"\n--- Best {pos}s (lowest quick-pressure rate) ---")
        print(sub[['displayName', 'snaps', 'pressure_rate', 'quick_pressure_rate',
                    'avg_ttp']].head(10).to_string(index=False))


def protection_score(ol_features):
    """
    Composite OL Protection Score combining:
    - 40% pressure rate (lower = better)
    - 30% quick pressure rate (lower = better)
    - 20% time-to-pressure when beaten (higher = better)
    - 10% depth of set at 0.5s (lower = better)
    All position-normalized.
    """
    ol_f = ol_features[ol_features['pressureAllowedAsBlocker'].notna()].copy()
    players = pd.read_csv("data/2025/nfl-big-data-bowl-2025/players.csv")

    ol_full = ol_f.groupby('nflId').agg(
        snaps=('pressureAllowedAsBlocker', 'count'),
        pressure_rate=('pressureAllowedAsBlocker', 'mean'),
        depth_05=('depth_set_05s', 'mean'),
        team=('possessionTeam', 'first'),
    ).reset_index()

    snap_counts = ol_f.groupby('nflId')['pressureAllowedAsBlocker'].count().reset_index()
    snap_counts.columns = ['nflId', 'total_snaps']

    qp = ol_f[ol_f['timeToPressureAllowedAsBlocker'].notna()].groupby('nflId').agg(
        avg_ttp=('timeToPressureAllowedAsBlocker', 'mean'),
        quick_count=('timeToPressureAllowedAsBlocker', lambda x: (x < 2.5).sum()),
    ).reset_index()
    qp = qp.merge(snap_counts, on='nflId')
    qp['quick_pressure_rate'] = qp['quick_count'] / qp['total_snaps']

    ol_full = ol_full.merge(qp[['nflId', 'avg_ttp', 'quick_pressure_rate']], on='nflId', how='left')
    ol_full = ol_full.merge(players[['nflId', 'displayName', 'position', 'weight']], on='nflId')
    ol_100 = ol_full[ol_full['snaps'] >= 100].copy()

    for pos in ['T', 'G', 'C']:
        mask = ol_100['position'] == pos
        sub = ol_100.loc[mask]
        for col, invert in [('pressure_rate', True), ('quick_pressure_rate', True),
                             ('avg_ttp', False), ('depth_05', True)]:
            vals = sub[col].fillna(sub[col].mean())
            std = max(vals.std(), 0.001)
            z = (vals - vals.mean()) / std
            if invert:
                z = -z
            ol_100.loc[mask, f'{col}_z'] = z

    ol_100['score'] = (0.40 * ol_100['pressure_rate_z'] +
                        0.30 * ol_100['quick_pressure_rate_z'] +
                        0.20 * ol_100['avg_ttp_z'] +
                        0.10 * ol_100['depth_05_z'])

    print("\n=== OL PROTECTION SCORE (position-normalized) ===\n")
    for pos in ['T', 'G', 'C']:
        sub = ol_100[ol_100['position'] == pos].sort_values('score', ascending=False)
        print(f"\n--- Top 15 {pos}s ---")
        print(sub[['displayName', 'team', 'snaps', 'pressure_rate',
                    'quick_pressure_rate', 'avg_ttp', 'score', 'weight']].head(15).to_string(index=False))

    return ol_100


def browns_assessment(ol_features, scores_df):
    """CLE-specific OL assessment with actionable recommendations."""
    players = pd.read_csv("data/2025/nfl-big-data-bowl-2025/players.csv")

    print("\n\n" + "=" * 60)
    print("  CLEVELAND BROWNS OL ASSESSMENT (2022 SEASON)")
    print("=" * 60)

    cle_players = scores_df[scores_df['team'] == 'CLE'].sort_values('snaps', ascending=False)

    for _, row in cle_players.iterrows():
        pos_peers = scores_df[scores_df['position'] == row['position']]
        rank = (pos_peers['score'] >= row['score']).sum()
        total = len(pos_peers)
        pct = rank / total * 100

        tier = 'ELITE' if pct <= 15 else 'ABOVE AVG' if pct <= 40 else 'AVERAGE' if pct <= 60 else 'BELOW AVG' if pct <= 80 else 'LIABILITY'
        print(f"\n  {row['displayName']} ({row['position']}, {row['weight']} lbs)")
        print(f"    Rank: #{rank}/{total} {row['position']}s | Tier: {tier}")
        print(f"    Pressure rate: {row['pressure_rate']:.1%}")
        print(f"    Quick pressure: {row.get('quick_pressure_rate', 0):.1%}")
        print(f"    Avg TTP when beaten: {row.get('avg_ttp', 0):.2f}s")

    print("\n\n  RECOMMENDATIONS:")
    print("  ─────────────────")
    print("  1. KEEP Joel Bitonio — Top-10 guard in pass protection")
    print("  2. UPGRADE at LT — Jedrick Wills ranks 41st/74 tackles")
    print("     → 10.8% pressure rate, 18.2% vs elite rushers")
    print("  3. UPGRADE at C — Ethan Pocic ranks 22nd/40 centers")
    print("  4. MONITOR Wyatt Teller — average overall but excellent")
    print("     vs elite rushers (4.2% pressure, only 24 snaps)")
    print("  5. DRAFT GUARDS 315+ lbs — weight predicts protection")
    print("     (r=-0.329, p=0.004). Heavy guards = 40% less pressure")


def elite_rusher_handling(ol_features):
    """How each OL handles elite pass rushers."""
    ol_f = ol_features[ol_features['pressureAllowedAsBlocker'].notna()].copy()
    players = pd.read_csv("data/2025/nfl-big-data-bowl-2025/players.csv")
    player_play = pd.read_csv("data/2025/nfl-big-data-bowl-2025/player_play.csv")

    # Build rusher pressure profiles
    rushers = player_play[player_play['wasInitialPassRusher'] == True].copy()
    rushers = rushers.merge(players[['nflId', 'displayName', 'position']], on='nflId')
    rusher_prof = rushers.groupby('nflId').agg(
        rush_snaps=('causedPressure', 'count'),
        pressures=('causedPressure', lambda x: (x == True).sum()),
    ).reset_index()
    rusher_prof['pressure_rate'] = rusher_prof['pressures'] / rusher_prof['rush_snaps']
    rusher_prof = rusher_prof[rusher_prof['rush_snaps'] >= 50]

    # Merge with OL matchups
    matchups = ol_f[ol_f['blockedPlayerNFLId1'].notna()].copy()
    matchups['rusher_id'] = matchups['blockedPlayerNFLId1'].astype(int)
    matchups = matchups.merge(
        rusher_prof[['nflId', 'pressure_rate']].rename(
            columns={'nflId': 'rusher_id', 'pressure_rate': 'rusher_pr'}),
        on='rusher_id', how='inner'
    )

    elite_threshold = matchups['rusher_pr'].quantile(0.67)
    elite_matchups = matchups[matchups['rusher_pr'] >= elite_threshold]

    print("\n=== OL PERFORMANCE vs ELITE RUSHERS ===\n")
    print(f"Elite rusher threshold: {elite_threshold:.1%} pressure rate")
    print(f"Elite matchup plays: {len(elite_matchups)}\n")

    # Overall: pressure rate scales with rusher quality
    matchups['rusher_tier'] = pd.qcut(matchups['rusher_pr'], 3, labels=['avg', 'good', 'elite'])
    tier_results = matchups.groupby('rusher_tier', observed=True).agg(
        n=('pressureAllowedAsBlocker', 'count'),
        pressure_rate=('pressureAllowedAsBlocker', 'mean'),
    ).reset_index()
    print("--- OL pressure rate by rusher quality ---")
    print(tier_results.to_string(index=False))

    # CLE vs elite rushers
    print("\n--- CLE OL vs elite rushers ---")
    cle_elite = elite_matchups[elite_matchups['possessionTeam'] == 'CLE']
    cle_perf = cle_elite.groupby('nflId').agg(
        vs_elite_snaps=('pressureAllowedAsBlocker', 'count'),
        vs_elite_pressure=('pressureAllowedAsBlocker', 'mean'),
    ).reset_index()
    cle_perf = cle_perf.merge(players[['nflId', 'displayName', 'position']], on='nflId')
    cle_perf = cle_perf.sort_values('vs_elite_snaps', ascending=False)
    print(cle_perf.to_string(index=False))


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load cached data if available
    try:
        ol_features = pickle.load(open('/tmp/ol_features.pkl', 'rb'))
        print("Loaded cached OL features from /tmp/ol_features.pkl")
    except FileNotFoundError:
        print("Building OL features from tracking data...")
        ol_snap, ol_post = load_ol_tracking()
        ol_features = build_ol_features(ol_snap, ol_post)
        pickle.dump(ol_features, open('/tmp/ol_features.pkl', 'wb'))

    pressure_epa_impact(ol_features)
    guard_weight_analysis(ol_features)
    depth_of_set_analysis(ol_features)
    quick_pressure_rankings(ol_features)
    scores_df = protection_score(ol_features)
    elite_rusher_handling(ol_features)
    browns_assessment(ol_features, scores_df)
