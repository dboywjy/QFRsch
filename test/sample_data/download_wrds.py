import wrds
import pandas as pd
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq

# Try to import your project config
try:
    from test.test_configs import PROJECT_DIR
except ImportError:
    PROJECT_DIR = "." 

# ================= CONFIGURATION =================
DATA_DIR = Path(PROJECT_DIR) / "test" / "data"
PRICES_DIR = DATA_DIR / "market_prices"
PRICES_FILE = PRICES_DIR / "market_prices.parquet"
META_DIR = DATA_DIR / "metadata"

PRICES_DIR.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

START_DATE_DEFAULT = "2004-12-31"
# ===============================================

def get_db_connection():
    print(">> Connecting to WRDS...")
    return wrds.Connection()

def get_last_update_date():
    if not PRICES_FILE.exists():
        return None
    try:
        table = pq.read_table(PRICES_FILE, columns=["date"])
        if table.num_rows == 0:
            return None
        max_date = table.column("date").to_pandas().max()
        return max_date.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"Warning: Could not determine last date from local data ({e}).")
        return None

def fetch_prices_and_delisting(db, start_date):
    """
    Final revised SQL:
    1. Join dsf (prices), dsedelist (delisting), and stocknames (exchange info).
    2. Fix the issue where 'exchcd' is not in dsf.
    3. Fix the incorrect table name 'dse'.
    """
    print(f">> Fetching market data since {start_date}...")
    
    sql_query = f"""
    SELECT 
        a.permno, 
        a.date, 
        abs(a.prc) as prc,
        a.vol, 
        a.ret, 
        a.shrout, 
        abs(a.openprc) as openprc, 
        abs(a.askhi) as askhi, 
        abs(a.bidlo) as bidlo, 
        a.cfacpr,               
        a.cfacshr,
        coalesce(b.dlret, 0) as dlret, 
        b.dlstcd,
        s.exchcd,
        s.shrcd
    FROM crsp.dsf AS a
    -- join delisting table (Postgres name: dsedelist)
    LEFT JOIN crsp.dsedelist AS b 
        ON a.permno = b.permno AND a.date = b.dlstdt
    -- join metadata table to get exchange codes (Postgres name: stocknames)
    -- use JOIN instead of LEFT JOIN because we only want main-exchange stocks (1,2,3)
    JOIN crsp.stocknames AS s
        ON a.permno = s.permno 
        AND a.date >= s.namedt 
        AND a.date <= s.nameenddt
    WHERE 
        a.date > '{start_date}' 
        AND s.exchcd IN (1, 2, 3)     -- NYSE, AMEX, NASDAQ
        AND s.shrcd IN (10, 11)       -- common shares only
        AND a.shrout > 0
        AND a.ret BETWEEN -0.99 AND 10 
    """
    
    print(">> Executing SQL (This might take a while due to joins)...")
    df = db.raw_sql(sql_query).sort_values(['date', 'permno'])
    return df

def process_and_save_prices(df):
    if df is None or df.empty:
        print(">> No new data to process.")
        return

    print(f">> Processing {len(df)} rows...")

    df['date'] = pd.to_datetime(df['date'])
    df['permno'] = df['permno'].astype('int32')
    df['ret'] = df['ret'].fillna(0)
    df['dlret'] = df['dlret'].fillna(0)

    df['ret_adj'] = (1 + df['ret']) * (1 + df['dlret']) - 1

    df['adj_close'] = df.apply(
        lambda x: x['prc'] / x['cfacpr'] if x['cfacpr'] != 0 else x['prc'], axis=1
    )

    df['mkt_cap'] = df['prc'] * df['shrout'] * 1000
    if PRICES_FILE.exists():
        df_old = pd.read_parquet(PRICES_FILE)
        df_combined = pd.concat([df_old, df])
    else:
        df_combined = df

    df_combined = df_combined.drop_duplicates(subset=['date', 'permno'], keep='last')
    df_combined = df_combined.sort_values(['date', 'permno'])
    df_combined.to_parquet(PRICES_FILE, index=False, compression='snappy')
    print(f"   -> Saved parquet with {len(df_combined)} rows at {PRICES_FILE}")

def update_metadata(db):
    print(">> Updating Metadata (Stocknames)...")
    
    # Confirm we're using the corrected nameenddt
    sql = """
    SELECT permno, ticker, comnam, namedt, nameenddt, hexcd, siccd 
    FROM crsp.stocknames
    ORDER BY permno, namedt
    """
    df_meta = db.raw_sql(sql)
    
    df_meta['permno'] = df_meta['permno'].astype('int32')
    df_meta['namedt'] = pd.to_datetime(df_meta['namedt'])
    df_meta['nameenddt'] = pd.to_datetime(df_meta['nameenddt'])
    
    save_path = META_DIR / "stocknames.parquet"
    df_meta.to_parquet(save_path, index=False, compression='snappy')
    print(f"   -> Metadata saved to {save_path}")

def main():
    db = None
    try:
        db = get_db_connection()
        
        update_metadata(db)
        
        last_date = get_last_update_date()
        if last_date:
            print(f">> Local data found up to: {last_date}")
            start_date = last_date
        else:
            print(f">> No local data. Starting from default: {START_DATE_DEFAULT}")
            start_date = (datetime.strptime(START_DATE_DEFAULT, "%Y-%m-%d") - pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        df_merged = fetch_prices_and_delisting(db, start_date)
        
        process_and_save_prices(df_merged)
        
        print(">> Update job completed successfully.")

    except Exception as e:
        print(f"!! Error occurred: {e}")
        # If this is a connection timeout, don't raise—just notify the user
        import sqlalchemy
        if isinstance(e, sqlalchemy.exc.OperationalError):
            print("Database connection failed. Please check your VPN or internet.")
        else:
            raise e 
    finally:
        if db:
            db.close()
            print(">> WRDS connection closed.")

if __name__ == "__main__":
    main()