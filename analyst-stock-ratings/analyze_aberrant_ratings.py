"""
Analyze analyst price targets to identify the most aberrant ratings.

Aberrant ratings are defined as cases where:
  1. Analyst consensus diverges significantly from what fundamentals would suggest
  2. Individual target spread is unusually wide (high disagreement)
  3. Price target implies extreme upside/downside vs peers in same sector
  4. Bullish consensus conflicts with deteriorating fundamentals (or vice versa)

Reads from: data/analyst_targets_fundamentals.csv
Outputs:    output/aberrant_ratings_report.csv
            output/summary_stats.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} tickers from {path}")
    return df


def zscore_within_sector(df: pd.DataFrame, col: str) -> pd.Series:
    """Compute z-scores of `col` within each sector."""
    def _z(s):
        if s.std() == 0 or s.std() != s.std():
            return s * 0
        return (s - s.mean()) / s.std()
    return df.groupby("sector")[col].transform(_z)


def compute_aberrance_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Score each stock on multiple aberrance dimensions."""
    df = df.copy()

    # Only analyze tickers with analyst coverage
    mask = df["apt_mean"].notna() & df["currentPrice"].notna()
    scored = df[mask].copy()

    # 1. Implied upside z-score within sector
    scored["z_upside_sector"] = zscore_within_sector(scored, "implied_upside_mean")

    # 2. Target spread z-score (analyst disagreement)
    scored["z_spread"] = zscore_within_sector(scored, "target_spread_pct")

    # 3. Valuation disconnect: high implied upside but already high PE
    #    (analysts bullish despite expensive valuation)
    scored["z_forwardPE_sector"] = zscore_within_sector(scored, "forwardPE")
    scored["valuation_disconnect"] = scored["z_upside_sector"] * scored["z_forwardPE_sector"]

    # 4. Sentiment-fundamental conflict:
    #    bullish_ratio is high but earnings growth is negative (or vice versa)
    eg = scored["earningsGrowth"].fillna(0)
    br = scored["bullish_ratio"].fillna(0.5)
    # Conflict = bullish sentiment * negative growth, or bearish * positive growth
    scored["sentiment_fundamental_conflict"] = np.where(
        (br > 0.6) & (eg < 0), br * abs(eg),
        np.where(
            (br < 0.4) & (eg > 0.15), (1 - br) * eg,
            0
        )
    )

    # 5. Composite aberrance score (weighted combination)
    scored["aberrance_score"] = (
        abs(scored["z_upside_sector"]) * 0.30
        + abs(scored["z_spread"]) * 0.20
        + abs(scored["valuation_disconnect"]) * 0.25
        + scored["sentiment_fundamental_conflict"] * 0.25
    )

    return scored.sort_values("aberrance_score", ascending=False)


def print_report(scored: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    report_cols = [
        "symbol", "shortName", "sector", "currentPrice",
        "apt_mean", "apt_median", "apt_high", "apt_low",
        "implied_upside_mean", "target_spread_pct",
        "bullish_ratio", "bearish_ratio", "rec_totalAnalysts",
        "forwardPE", "trailingPE", "earningsGrowth", "revenueGrowth",
        "profitMargins", "debtToEquity", "beta",
        "z_upside_sector", "z_spread", "valuation_disconnect",
        "sentiment_fundamental_conflict", "aberrance_score",
    ]
    available_cols = [c for c in report_cols if c in scored.columns]

    # Save full ranked list
    scored[available_cols].to_csv(out_dir / "aberrant_ratings_report.csv", index=False)

    # Summary to stdout and file
    lines = []
    lines.append("=" * 80)
    lines.append("ANALYST RATING ABERRANCE REPORT")
    lines.append(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Universe: {len(scored)} tickers with analyst coverage")
    lines.append("=" * 80)

    lines.append("\n--- TOP 20 MOST ABERRANT RATINGS ---")
    top20 = scored.head(20)
    display_cols = ["symbol", "sector", "currentPrice", "apt_mean",
                    "implied_upside_mean", "forwardPE", "earningsGrowth",
                    "bullish_ratio", "aberrance_score"]
    display_cols = [c for c in display_cols if c in top20.columns]
    lines.append(top20[display_cols].to_string(index=False))

    lines.append("\n--- MOST BULLISH DESPITE POOR FUNDAMENTALS ---")
    bull_poor = scored[
        (scored["bullish_ratio"] > 0.65) &
        (scored["earningsGrowth"].fillna(0) < -0.05)
    ].head(10)
    if len(bull_poor) > 0:
        lines.append(bull_poor[display_cols].to_string(index=False))
    else:
        lines.append("  (none found)")

    lines.append("\n--- MOST BEARISH DESPITE STRONG FUNDAMENTALS ---")
    bear_strong = scored[
        (scored["bearish_ratio"] > 0.10) &
        (scored["earningsGrowth"].fillna(0) > 0.15)
    ].head(10)
    if len(bear_strong) > 0:
        lines.append(bear_strong[display_cols].to_string(index=False))
    else:
        lines.append("  (none found)")

    lines.append("\n--- HIGHEST ANALYST DISAGREEMENT (spread) ---")
    high_spread = scored.nlargest(10, "target_spread_pct")
    spread_cols = ["symbol", "sector", "currentPrice", "apt_low", "apt_high",
                   "target_spread_pct", "rec_totalAnalysts"]
    spread_cols = [c for c in spread_cols if c in high_spread.columns]
    lines.append(high_spread[spread_cols].to_string(index=False))

    lines.append("\n--- SECTOR SUMMARY ---")
    sector_stats = scored.groupby("sector").agg(
        count=("symbol", "size"),
        avg_implied_upside=("implied_upside_mean", "mean"),
        avg_bullish_ratio=("bullish_ratio", "mean"),
        avg_aberrance=("aberrance_score", "mean"),
    ).sort_values("avg_aberrance", ascending=False)
    lines.append(sector_stats.to_string())

    report_text = "\n".join(lines)
    print(report_text)

    with open(out_dir / "summary_stats.txt", "w") as f:
        f.write(report_text)
    print(f"\n[INFO] Report saved to {out_dir / 'summary_stats.txt'}")
    print(f"[INFO] Full data saved to {out_dir / 'aberrant_ratings_report.csv'}")


def main():
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / "data" / "analyst_targets_fundamentals.csv"

    if not data_path.exists():
        print(f"[ERR] Data file not found: {data_path}")
        print("  Run aggregate_analyst_data.py first.")
        sys.exit(1)

    df = load_data(data_path)
    scored = compute_aberrance_scores(df)
    print_report(scored, project_dir / "output")


if __name__ == "__main__":
    main()
