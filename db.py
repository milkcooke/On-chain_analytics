import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "local.db")

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        time        TEXT NOT NULL,
        coin        TEXT NOT NULL,
        side        TEXT NOT NULL,
        price       REAL NOT NULL,
        size        REAL NOT NULL,
        trade_id    INTEGER,
        UNIQUE(trade_id, coin)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS liquidations (
        time        TEXT NOT NULL,
        coin        TEXT NOT NULL,
        side        TEXT NOT NULL,
        price       REAL NOT NULL,
        size_usd    REAL NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS funding_rates (
        time        TEXT NOT NULL,
        coin        TEXT NOT NULL,
        funding_rate REAL NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS open_interest (
        time        TEXT NOT NULL,
        coin        TEXT NOT NULL,
        open_interest REAL NOT NULL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_fills (
        time        TEXT NOT NULL,
        user_address TEXT NOT NULL,
        coin        TEXT NOT NULL,
        side        TEXT NOT NULL,
        price       REAL NOT NULL,
        size        REAL NOT NULL,
        fee         REAL,
        pnl         REAL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS positions (
        time        TEXT NOT NULL,
        user_address TEXT NOT NULL,
        coin        TEXT NOT NULL,
        size        REAL,
        entry_price REAL,
        leverage    REAL
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_coin_time ON trades (coin, time DESC);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_liquidations_coin_time ON liquidations (coin, time DESC);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_funding_coin_time ON funding_rates (coin, time DESC);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_fills_addr_time ON user_fills (user_address, time DESC);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_positions_addr_time ON positions (user_address, time DESC);")

    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized (SQLite).")

def insert_many(table, rows, columns):
    conn = get_conn()
    cur = conn.cursor()
    placeholders = ",".join(["?" for _ in columns])
    cols = ",".join(columns)
    for row in rows:
        try:
            cur.execute(f"INSERT OR IGNORE INTO {table} ({cols}) VALUES ({placeholders})", row)
        except Exception as e:
            print(f"Insert error: {e}")
    conn.commit()
    cur.close()
    conn.close()
