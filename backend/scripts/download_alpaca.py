import os
import sys
import asyncio
import httpx
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://data.alpaca.markets/v2"

DATA_DIR = os.path.join("backend", "data_cache_alpaca")
os.makedirs(DATA_DIR, exist_ok=True)

# S&P 500 Top 50 (Subset for demo, typically we'd fetch all 500)
# Expanding to 100 liquid tickers for the 10-year dataset
import json

# Load Microcosm Manifest
MANIFEST_FILE = os.path.join("backend", "data", "microcosm_manifest.json")
if os.path.exists(MANIFEST_FILE):
    with open(MANIFEST_FILE, 'r') as f:
        data = json.load(f)
        # Flatten and unique
        TICKERS = list(set(data['leaders'] + data['vol_beasts'] + data['liquidity_proxies']))
        print(f"Loaded {len(TICKERS)} tickers from Microcosm Manifest.")
else:
    print("Microcosm manifest not found, using default fallback.")
    TICKERS = ["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AAPL", "MSFT", "AMD", "COIN", "MSTR"]


headers = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": SECRET_KEY
}

async def fetch_bars(client, symbol, start, end):
    url = f"{BASE_URL}/stocks/{symbol}/bars"
    all_bars = []
    page_token = None

    params = {
        "start": start,
        "end": end,
        "timeframe": "1Min",
        "limit": 10000,
        "adjustment": "split"
    }

    while True:
        if page_token:
            params["page_token"] = page_token

        try:
            resp = await client.get(url, params=params, headers=headers, timeout=30.0)
            if resp.status_code != 200:
                print(f"Error fetching {symbol}: {resp.text}")
                break

            data = resp.json()
            bars = data.get("bars", [])
            if not bars:
                break

            all_bars.extend(bars)
            page_token = data.get("next_page_token")

            if not page_token:
                break

            # Rate limit politeness
            await asyncio.sleep(0.5)

        except Exception as e:
            print(f"Exception fetching {symbol}: {e}")
            break

    return all_bars

async def process_symbol(sem, client, symbol):
    async with sem:
        filepath = os.path.join(DATA_DIR, f"{symbol}_1m.parquet")
        if os.path.exists(filepath):
            print(f"Skipping {symbol} (Already exists)")
            return

        print(f"Downloading {symbol} (2015-2025)...")
        # 10 Years
        start_date = "2015-01-01T00:00:00Z"
        end_date = "2025-01-01T00:00:00Z"

        # Retry Logic
        for attempt in range(3):
            bars = await fetch_bars(client, symbol, start_date, end_date)
            if bars:
                break
            print(f"Retry {attempt+1} for {symbol}...")
            await asyncio.sleep(2 * (attempt + 1))

        if not bars:
            print(f"No data for {symbol} after retries.")
            return

        # Convert to DF
        df = pd.DataFrame(bars)
        # Alpaca returns: t, o, h, l, c, v, n, vw
        df = df.rename(columns={
            "t": "timestamp",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "n": "trade_count",
            "vw": "vwap"
        })

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Save Parquet
        df.to_parquet(filepath, compression='snappy')
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"Saved {symbol}: {len(df)} rows ({size_mb:.2f} MB)")

        # Cool down
        await asyncio.sleep(1.0)

async def main():
    connector = httpx.AsyncClient(limits=httpx.Limits(max_keepalive_connections=5, max_connections=5))
    sem = asyncio.Semaphore(2) # Reduced to 2 concurrent requests for Free Tier logic

    async with connector as client:
        tasks = [process_symbol(sem, client, sym) for sym in TICKERS]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    if not API_KEY:
        print("Error: ALPACA_API_KEY not found in .env")
        sys.exit(1)

    start_time = datetime.now()
    asyncio.run(main())
    print(f"Download Complete. Duration: {datetime.now() - start_time}")
