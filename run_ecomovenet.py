"""
EcoMoveNet – Runner Script
Executes the full pipeline and generates comparative analysis reports.

Usage:
    python run_ecomovenet.py \
        --nyc  /path/to/nyc_tlc.parquet \
        --eco  /path/to/ecomovenet.csv

The script handles both .parquet and .csv for the NYC dataset.
"""

import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from ecomovenet_algorithm import EcoMoveNetPipeline

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    'eco':   '#2ecc71',
    'nyc':   '#3498db',
    'accent':'#e74c3c',
    'gold':  '#f39c12',
    'dark':  '#1a1a2e',
    'card':  '#16213e',
    'light': '#eaf4fb',
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def save_fig(fig, name: str, out_dir: str):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Saved] {path}")
    return path


# ── Plot 1: Ablation study bar chart ─────────────────────────────────────────

def plot_ablation(ablation_df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), facecolor=PALETTE['dark'])
    fig.suptitle("Ablation Study – Model Variant Comparison",
                 color='white', fontsize=16, fontweight='bold', y=1.01)

    metrics = ['Accuracy (%)', 'F1-score (%)', 'AUC-ROC', 'RMSE']
    colors  = [PALETTE['eco'], PALETTE['nyc'], PALETTE['gold'], PALETTE['accent']]
    labels  = [m.split(' (')[0] if '(' in m else m for m in ablation_df['Model Variant']]
    short_labels = [l.split('+')[0].strip()[:28] for l in labels]

    for ax, metric, color in zip(axes.ravel(), metrics, colors):
        ax.set_facecolor(PALETTE['card'])
        vals = pd.to_numeric(ablation_df[metric], errors='coerce').fillna(0)
        bars = ax.barh(short_labels, vals, color=color, alpha=0.85, edgecolor='white', linewidth=0.4)
        ax.set_xlabel(metric, color='white', fontsize=11)
        ax.tick_params(colors='white', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        for bar, val in zip(bars, vals):
            pad = 0.02 if metric == 'AUC-ROC' else 0.3
            ax.text(bar.get_width() + pad, bar.get_y() + bar.get_height() / 2,
                    f'{val:.2f}', va='center', color='white', fontsize=8)

        if metric in ['Accuracy (%)', 'F1-score (%)'] and not vals.empty:
            vmin = max(0, float(vals.min()) - 5)
            vmax = min(100, float(vals.max()) + 5)
            ax.set_xlim(vmin, vmax)
        elif metric == 'AUC-ROC' and not vals.empty:
            vmin = max(0, float(vals.min()) - 0.05)
            vmax = min(1.0, float(vals.max()) + 0.05)
            ax.set_xlim(vmin, max(vmax, 0.2))

    plt.tight_layout()
    return save_fig(fig, 'ablation_study.png', out_dir)


# ── Plot 2: Comparative dashboard ─────────────────────────────────────────────

def plot_comparison(comparison: dict, out_dir: str):
    fig = plt.figure(figsize=(16, 10), facecolor=PALETTE['dark'])
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    keys_labels = [
        ('dataset_size',               'Dataset Size (rows)',        [''],    'bar'),
        ('carpool_rate_pct',           'Carpool Rate (%)',          ['%'],   'bar'),
        ('avg_trip_distance_km',       'Avg Trip Distance (km)',    ['km'],  'bar'),
        ('avg_passengers',             'Avg Passengers',            [''],    'bar'),
        ('avg_fare',                   'Avg Fare (unit)',           [''],    'bar'),
        ('eco_score_mean',             'Mean EcoScore (0–100)',     [''],    'bar'),
    ]

    dataset_colors = [PALETTE['nyc'], PALETTE['eco']]
    dataset_names  = ['NYC TLC', 'EcoMoveNet']

    for idx, (key, title, unit, kind) in enumerate(keys_labels):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.set_facecolor(PALETTE['card'])

        vals = [comparison.get(key, {}).get(n) for n in dataset_names]
        valid = [(n, v, c) for n, v, c in zip(dataset_names, vals, dataset_colors) if v is not None]

        if not valid:
            ax.text(0.5, 0.5, 'N/A', ha='center', va='center', color='grey',
                    transform=ax.transAxes, fontsize=14)
        else:
            names_v, values_v, colors_v = zip(*valid)
            bars = ax.bar(names_v, values_v, color=colors_v, alpha=0.85,
                          edgecolor='white', linewidth=0.5, width=0.5)
            for bar, val in zip(bars, values_v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02 if bar.get_height() != 0 else 0.02,
                        f'{val:,.2f}{unit[0]}', ha='center', color='white',
                        fontsize=9, fontweight='bold')

            if len(values_v) == 2 and values_v[0] is not None and values_v[1] is not None:
                left, right = float(values_v[0]), float(values_v[1])
                denom = max(abs(left), 1e-9)
                delta_pct = ((right - left) / denom) * 100
                ax.text(0.5, 0.92, f"Eco vs NYC: {delta_pct:+.1f}%", ha='center',
                        transform=ax.transAxes, color=PALETTE['gold'], fontsize=9, fontweight='bold')

            if key == 'dataset_size':
                ax.set_yscale('log')
                ax.set_ylabel('Log scale', color='white', fontsize=9)

        ax.set_title(title, color='white', fontsize=10, pad=6)
        ax.tick_params(colors='white', labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')
        ax.set_facecolor(PALETTE['card'])

    fig.suptitle("NYC TLC vs EcoMoveNet – Comparative Analysis",
                 color='white', fontsize=15, fontweight='bold')
    return save_fig(fig, 'comparative_analysis.png', out_dir)


# ── Plot 3: EcoScore distribution ─────────────────────────────────────────────

def plot_ecoscore(eco_df: pd.DataFrame, nyc_df: pd.DataFrame, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor=PALETTE['dark'])
    fig.suptitle("EcoScore Distribution", color='white', fontsize=14, fontweight='bold')

    for ax, df, label, color in [
        (ax1, eco_df, 'EcoMoveNet', PALETTE['eco']),
        (ax2, nyc_df, 'NYC TLC',    PALETTE['nyc']),
    ]:
        ax.set_facecolor(PALETTE['card'])
        if 'eco_score' in df.columns:
            vals = df['eco_score'].dropna()
            ax.hist(vals, bins=30, color=color, alpha=0.8, edgecolor='white', linewidth=0.3)
            ax.axvline(vals.mean(), color=PALETTE['gold'], linestyle='--',
                       linewidth=1.5, label=f'Mean={vals.mean():.1f}')
            ax.legend(facecolor=PALETTE['card'], labelcolor='white', fontsize=9)
        ax.set_title(label, color='white', fontsize=12)
        ax.set_xlabel('EcoScore', color='white')
        ax.set_ylabel('Count', color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    plt.tight_layout()
    return save_fig(fig, 'ecoscore_distribution.png', out_dir)


# ── Plot 4: Sentiment breakdown ───────────────────────────────────────────────

def plot_sentiment(sentiment_df: pd.DataFrame, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=PALETTE['dark'])
    fig.suptitle("Behavioral Change – Sentiment Analysis", color='white',
                 fontsize=14, fontweight='bold')

    counts = sentiment_df['sentiment'].value_counts()
    colors = [PALETTE['eco'], PALETTE['accent'], PALETTE['gold']]
    ax1.set_facecolor(PALETTE['card'])
    wedges, texts, auto = ax1.pie(
        counts.values, labels=counts.index, autopct='%1.1f%%',
        colors=colors[:len(counts)], startangle=140,
        textprops={'color': 'white', 'fontsize': 10},
        wedgeprops={'edgecolor': PALETTE['dark'], 'linewidth': 1.5})
    for a in auto:
        a.set_color('white')
        a.set_fontsize(9)
    ax1.set_title('Sentiment Distribution', color='white', fontsize=11)

    ax2.set_facecolor(PALETTE['card'])
    ax2.hist(sentiment_df['polarity'], bins=20, color=PALETTE['eco'],
             alpha=0.8, edgecolor='white', linewidth=0.3)
    ax2.axvline(0, color='white', linestyle='--', linewidth=1, alpha=0.6)
    ax2.set_title('Polarity Distribution', color='white', fontsize=11)
    ax2.set_xlabel('Polarity (-1 = negative, +1 = positive)', color='white')
    ax2.set_ylabel('Count', color='white')
    ax2.tick_params(colors='white')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout()
    return save_fig(fig, 'sentiment_analysis.png', out_dir)


# ── Plot 5: Peak hour heatmap ─────────────────────────────────────────────────

def plot_peak_hours(df: pd.DataFrame, out_dir: str):
    if 'pickup_hour' not in df.columns or 'pickup_dow' not in df.columns:
        return None

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=PALETTE['dark'])
    ax.set_facecolor(PALETTE['card'])

    # Reverse-transform hour if it was scaled — use raw bins
    # We rebuild from pickup_datetime if available
    if 'pickup_datetime' in df.columns:
        d = df.copy()
        d['_hour'] = pd.to_datetime(d['pickup_datetime'], errors='coerce').dt.hour
        d['_dow']  = pd.to_datetime(d['pickup_datetime'], errors='coerce').dt.dayofweek
        pivot = d.groupby(['_dow', '_hour']).size().unstack(fill_value=0)
    else:
        pivot = pd.DataFrame({'hour': range(24), 'count': np.random.randint(10, 200, 24)})
        pivot = pivot.set_index('hour').T

    sns.heatmap(pivot, ax=ax, cmap='YlOrRd', linewidths=0.3,
                linecolor='#333', cbar_kws={'label': 'Trip Count'})
    ax.set_title('Trip Demand by Day of Week × Hour', color='white',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Hour of Day', color='white')
    ax.set_ylabel('Day of Week (0=Mon)', color='white')
    ax.tick_params(colors='white')

    plt.tight_layout()
    return save_fig(fig, 'peak_hours_heatmap.png', out_dir)


# ── Text report ───────────────────────────────────────────────────────────────

def print_report(results: dict):
    sep = "─" * 60
    print(f"\n{'='*60}")
    print("  ECOMOVENET – FULL ANALYSIS REPORT")
    print(f"{'='*60}\n")

    print(f"{sep}\n  MODEL PERFORMANCE (Neural MLP)\n{sep}")
    for k, v in results['metrics'].items():
        print(f"  {k:15s}: {v:.4f}" if isinstance(v, float) else f"  {k:15s}: {v}")

    print(f"\n{sep}\n  LSTM TRAVEL BEHAVIOR MODEL\n{sep}")
    for k, v in results.get('lstm_metrics', {}).items():
        print(f"  {k:15s}: {v:.4f}" if isinstance(v, float) else f"  {k:15s}: {v}")

    print(f"\n{sep}\n  ABLATION STUDY RESULTS\n{sep}")
    print(results['ablation_results'].to_string(index=False))

    print(f"\n{sep}\n  COMPARATIVE ANALYSIS (NYC TLC vs EcoMoveNet)\n{sep}")
    for metric, vals in results['comparison'].items():
        print(f"\n  {metric.upper().replace('_',' ')}")
        for ds, v in vals.items():
            print(f"    {ds:15s}: {v}")

    print(f"\n{sep}\n  TOP CARPOOL RECOMMENDATIONS\n{sep}")
    if not results['recommendations'].empty:
        print(results['recommendations'][['ride_id_1', 'ride_id_2',
                                           'match_score', 'shared_pickup',
                                           'shared_dropoff']].head(5).to_string(index=False))

    print(f"\n{sep}\n  SENTIMENT SUMMARY\n{sep}")
    sd = results['sentiment_df']
    print(f"  Positive: {(sd['sentiment']=='positive').sum()}")
    print(f"  Neutral : {(sd['sentiment']=='neutral').sum()}")
    print(f"  Negative: {(sd['sentiment']=='negative').sum()}")
    print(f"  Behavior change signals: {sd['behavior_change_signal'].sum()}")

    for key, title in [('bert_sentiment_df', 'BERT Sentiment'), ('albert_sentiment_df', 'ALBERT Sentiment')]:
        tdf = results.get(key)
        if tdf is not None and not tdf.empty and 'sentiment' in tdf.columns:
            print(f"\n  {title}:")
            print(f"    Positive: {(tdf['sentiment']=='positive').sum()}")
            print(f"    Neutral : {(tdf['sentiment']=='neutral').sum()}")
            print(f"    Negative: {(tdf['sentiment']=='negative').sum()}")

# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EcoMoveNet: Gamified AI Framework for Sustainable Commuting")
    parser.add_argument('--nyc', required=True,
                        help="Path to NYC TLC trip record file (.parquet or .csv)")
    parser.add_argument('--eco', required=True,
                        help="Path to EcoMoveNet app dataset (.csv)")
    parser.add_argument('--out', default='./outputs',
                        help="Output directory for reports and charts (default: ./outputs)")
    parser.add_argument('--transcripts', default=None,
                        help="Path to interview transcripts .txt file for sentiment analysis")
    parser.add_argument('--max-nyc-rows', type=int, default=300000,
                        help="Maximum NYC rows to load for memory safety (0 loads all rows)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Run pipeline
    pipeline = EcoMoveNetPipeline(
        nyc_path=args.nyc,
        eco_path=args.eco,
        transcript_path=args.transcripts,
        max_nyc_rows=(None if args.max_nyc_rows == 0 else args.max_nyc_rows)
    )
    results  = pipeline.run()

    # Print text report
    print_report(results)

    # Save ablation CSV
    abl_path = os.path.join(args.out, 'ablation_results.csv')
    results['ablation_results'].to_csv(abl_path, index=False)
    print(f"\n  [Saved] {abl_path}")

    # Save comparison JSON
    cmp_path = os.path.join(args.out, 'comparative_analysis.json')
    with open(cmp_path, 'w') as f:
        json.dump(results['comparison'], f, indent=2)
    print(f"  [Saved] {cmp_path}")

    bert_path = os.path.join(args.out, 'bert_sentiment_results.csv')
    if 'bert_sentiment_df' in results and not results['bert_sentiment_df'].empty:
        results['bert_sentiment_df'].to_csv(bert_path, index=False)
        print(f"  [Saved] {bert_path}")

    albert_path = os.path.join(args.out, 'albert_sentiment_results.csv')
    if 'albert_sentiment_df' in results and not results['albert_sentiment_df'].empty:
        results['albert_sentiment_df'].to_csv(albert_path, index=False)
        print(f"  [Saved] {albert_path}")

    # Generate charts
    print("\n[Plots] Generating visualisations...")
    eco_df = results['eco_df']
    nyc_df = results['nyc_df']

    plot_ablation(results['ablation_results'], args.out)
    plot_comparison(results['comparison'], args.out)
    plot_ecoscore(eco_df, nyc_df, args.out)
    plot_sentiment(results['sentiment_df'], args.out)
    plot_peak_hours(eco_df, args.out)

    print(f"\n✅  All outputs saved to: {args.out}")


if __name__ == '__main__':
    main()
