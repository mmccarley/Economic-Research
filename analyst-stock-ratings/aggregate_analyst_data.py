"""
Aggregate analyst price targets and stock fundamentals for a broad universe of tickers.

Pulls from Yahoo Finance via yfinance:
  - Analyst consensus price targets (current, mean, median, high, low)
  - Analyst recommendation counts (strongBuy, buy, hold, sell, strongSell)
  - Key fundamentals (PE, EPS, margins, growth, debt, etc.)

Outputs a consolidated CSV to data/analyst_targets_fundamentals.csv
"""

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Ticker universe – S&P 500 constituents (a representative large-cap set).
# Sourced from Wikipedia; refreshed periodically.
# ---------------------------------------------------------------------------

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

FALLBACK_TICKERS = [
    # Top ~100 by market cap as a fallback if Wikipedia scrape fails
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B",
    "LLY", "AVGO", "JPM", "TSLA", "UNH", "XOM", "V", "MA", "PG",
    "COST", "JNJ", "HD", "ABBV", "WMT", "NFLX", "BAC", "CRM", "CVX",
    "MRK", "KO", "ORCL", "AMD", "PEP", "TMO", "ACN", "LIN", "MCD",
    "CSCO", "ADBE", "ABT", "WFC", "DHR", "GE", "PM", "ISRG", "NOW",
    "CAT", "QCOM", "TXN", "INTU", "IBM", "VZ", "CMCSA", "AMGN",
    "AMAT", "PFE", "GS", "SPGI", "NEE", "UBER", "RTX", "LOW", "T",
    "BKNG", "HON", "BLK", "UNP", "SYK", "MS", "ELV", "SCHW", "PLD",
    "DE", "ADP", "COP", "MDLZ", "CB", "LRCX", "MMC", "BMY", "VRTX",
    "ADI", "GILD", "TMUS", "PANW", "SLB", "KLAC", "AXP", "CI",
    "SO", "DUK", "CME", "REGN", "MO", "ICE", "SNPS", "CDNS", "FI",
    "BSX", "WM", "PGR", "EOG", "MCK",
]

INFO_FIELDS = [
    # Valuation
    "marketCap", "enterpriseToRevenue", "forwardPE", "trailingPE",
    "priceToBook", "priceToSalesTrailing12Months", "trailingPegRatio",
    # Earnings & growth
    "epsForward", "epsTrailingTwelveMonths", "earningsGrowth",
    "earningsQuarterlyGrowth", "revenueGrowth",
    # Profitability
    "profitMargins", "operatingMargins", "grossMargins", "ebitdaMargins",
    "returnOnAssets", "returnOnEquity",
    # Balance sheet
    "debtToEquity", "totalCashPerShare", "totalDebt", "totalRevenue",
    "bookValue",
    # Dividends & risk
    "dividendYield", "beta",
    # Price context
    "currentPrice", "fiftyTwoWeekHighChangePercent",
    "fiftyTwoWeekLowChangePercent",
    # Analyst targets (also in info dict)
    "targetHighPrice", "targetLowPrice", "targetMeanPrice", "targetMedianPrice",
    # Misc
    "sector", "industry", "shortName",
]


def fetch_sp500_tickers() -> list[str]:
    """Try to pull the current S&P 500 ticker list from Wikipedia."""
    try:
        tables = pd.read_html(SP500_URL, header=0)
        df = tables[0]
        tickers = df["Symbol"].tolist()
        # Yahoo uses '-' instead of '.' for class shares
        tickers = [t.replace(".", "-") for t in tickers]
        print(f"[INFO] Fetched {len(tickers)} S&P 500 tickers from Wikipedia.")
        return tickers
    except Exception as e:
        print(f"[WARN] Could not fetch S&P 500 list ({e}); using fallback list.")
        return FALLBACK_TICKERS


def fetch_ticker_data(symbol: str) -> dict | None:
    """Fetch analyst targets, recommendations, and fundamentals for one ticker."""
    try:
        tk = yf.Ticker(symbol)
        info = tk.info or {}

        # Basic sanity check – skip if no quote type
        if info.get("quoteType") not in ("EQUITY", None):
            return None

        row = {"symbol": symbol}

        # --- Analyst price targets ---
        try:
            apt = tk.analyst_price_targets
            if isinstance(apt, dict):
                row["apt_current"] = apt.get("current")
                row["apt_mean"] = apt.get("mean")
                row["apt_median"] = apt.get("median")
                row["apt_high"] = apt.get("high")
                row["apt_low"] = apt.get("low")
        except Exception:
            pass

        # --- Analyst recommendations (most recent month) ---
        try:
            rec = tk.recommendations
            if rec is not None and len(rec) > 0:
                latest = rec.iloc[0]
                for col in ["strongBuy", "buy", "hold", "sell", "strongSell"]:
                    row[f"rec_{col}"] = latest.get(col)
                total = sum(
                    latest.get(c, 0)
                    for c in ["strongBuy", "buy", "hold", "sell", "strongSell"]
                )
                row["rec_totalAnalysts"] = total
        except Exception:
            pass

        # --- Fundamentals from info ---
        for field in INFO_FIELDS:
            val = info.get(field)
            row[field] = val

        return row

    except Exception as e:
        print(f"  [ERR] {symbol}: {e}")
        return None


def compute_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns useful for aberrant-rating detection."""
    df = df.copy()

    # Implied upside/downside from analyst consensus vs current price
    price = df["currentPrice"]
    df["implied_upside_mean"] = (df["apt_mean"] - price) / price
    df["implied_upside_median"] = (df["apt_median"] - price) / price
    df["implied_upside_high"] = (df["apt_high"] - price) / price
    df["implied_downside_low"] = (df["apt_low"] - price) / price

    # Target spread (high - low) relative to current price
    df["target_spread_pct"] = (df["apt_high"] - df["apt_low"]) / price

    # Consensus skew: how far median is from midpoint of high/low
    midpoint = (df["apt_high"] + df["apt_low"]) / 2
    df["consensus_skew"] = (df["apt_median"] - midpoint) / (
        df["apt_high"] - df["apt_low"]
    ).replace(0, np.nan)

    # Bullish ratio: (strongBuy + buy) / total
    bulls = df.get("rec_strongBuy", 0) + df.get("rec_buy", 0)
    total = df.get("rec_totalAnalysts", 0)
    df["bullish_ratio"] = bulls / total.replace(0, np.nan)

    # Bearish ratio
    bears = df.get("rec_sell", 0) + df.get("rec_strongSell", 0)
    df["bearish_ratio"] = bears / total.replace(0, np.nan)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate analyst price targets and fundamentals."
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Specific tickers to fetch (default: S&P 500).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: data/analyst_targets_fundamentals.csv).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Seconds to wait between ticker requests (be polite to Yahoo).",
    )
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    out_path = Path(args.output) if args.output else project_dir / "data" / "analyst_targets_fundamentals.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tickers = args.tickers if args.tickers else fetch_sp500_tickers()

    print(f"[INFO] Fetching data for {len(tickers)} tickers...")
    rows = []
    errors = 0
    for i, sym in enumerate(tickers, 1):
        if i % 25 == 0 or i == 1:
            print(f"  [{i}/{len(tickers)}] Processing {sym}...")
        row = fetch_ticker_data(sym)
        if row:
            rows.append(row)
        else:
            errors += 1
        time.sleep(args.delay)

    print(f"[INFO] Successfully fetched {len(rows)} tickers ({errors} errors).")

    df = pd.DataFrame(rows)
    df = compute_derived_fields(df)

    # Sort by implied upside descending – most "optimistic" analyst targets first
    df.sort_values("implied_upside_mean", ascending=False, inplace=True)

    df.to_csv(out_path, index=False)
    print(f"[INFO] Saved {len(df)} rows to {out_path}")

    # Quick summary stats
    print("\n=== Quick Summary ===")
    print(f"Tickers with analyst targets: {df['apt_mean'].notna().sum()}")
    print(f"Tickers with recommendations: {df['rec_totalAnalysts'].notna().sum()}")
    print(f"\nImplied upside (mean target vs current price):")
    print(df["implied_upside_mean"].describe().to_string())
    print(f"\nTop 10 highest implied upside:")
    cols = ["symbol", "shortName", "currentPrice", "apt_mean", "implied_upside_mean",
            "forwardPE", "earningsGrowth", "sector"]
    top = df.dropna(subset=["implied_upside_mean"]).head(10)
    print(top[cols].to_string(index=False))
    print(f"\nTop 10 lowest implied upside (most overvalued vs targets):")
    bottom = df.dropna(subset=["implied_upside_mean"]).tail(10)
    print(bottom[cols].to_string(index=False))


if __name__ == "__main__":
    main()
