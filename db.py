import psycopg2
from psycopg2.extras import execute_values
from config import DATABASE_URL

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    """Создаёт все таблицы и гипертаблицы TimescaleDB."""
    conn = get_conn()
    cur = conn.cursor()

    # Таблица сделок
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        time        TIMESTAMPTZ NOT NULL,
        coin        TEXT NOT NULL,
        side        TEXT NOT NULL,
        price       DOUBLE PRECISION NOT NULL,
        size        DOUBLE PRECISION NOT NULL,
        trade_id    BIGINT,
        UNIQUE(trade_id, coin)
    );
    """)

    # Таблица ликвидаций
    cur.execute("""
    CREATE TABLE IF NOT EXISTS liquidations (
        time        TIMESTAMPTZ NOT NULL,
        coin        TEXT NOT NULL,
        side        TEXT NOT NULL,
        price       DOUBLE PRECISION NOT NULL,
        size_usd    DOUBLE PRECISION NOT NULL
    );
    """)

    # Таблица ставок финансирования
    cur.execute("""
    CREATE TABLE IF NOT EXISTS funding_rates (
        time        TIMESTAMPTZ NOT NULL,
        coin        TEXT NOT NULL,
        funding_rate DOUBLE PRECISION NOT NULL
    );
    """)

    # Таблица открытого интереса
    cur.execute("""
    CREATE TABLE IF NOT EXISTS open_interest (
        time        TIMESTAMPTZ NOT NULL,
        coin        TEXT NOT NULL,
        open_interest DOUBLE PRECISION NOT NULL
    );
    """)

    # Сделки пользователя
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_fills (
        time        TIMESTAMPTZ NOT NULL,
        user_address TEXT NOT NULL,
        coin        TEXT NOT NULL,
        side        TEXT NOT NULL,
        price       DOUBLE PRECISION NOT NULL,
        size        DOUBLE PRECISION NOT NULL,
        fee         DOUBLE PRECISION,
        pnl         DOUBLE PRECISION
    );
    """)

    # Позиции пользователя
    cur.execute("""
    CREATE TABLE IF NOT EXISTS positions (
        time        TIMESTAMPTZ NOT NULL,
        user_address TEXT NOT NULL,
        coin        TEXT NOT NULL,
        size        DOUBLE PRECISION,
        entry_price DOUBLE PRECISION,
        leverage    DOUBLE PRECISION
    );
    """)

    # Включаем TimescaleDB и делаем гипертаблицы (если расширение доступно)
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        for table in ["trades", "liquidations", "funding_rates", "open_interest"]:
            cur.execute(f"SELECT create_hypertable('{table}', 'time', if_not_exists => TRUE);")
    except Exception as e:
        print(f"TimescaleDB extension not available, using regular PostgreSQL. ({e})")
        conn.rollback()
    else:
        conn.commit()

    # Индексы для быстрых запросов
    cur.execute("CREATE INDEX IF NOT EXISTS idx_trades_coin_time ON trades (coin, time DESC);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_liquidations_coin_time ON liquidations (coin, time DESC);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_funding_coin_time ON funding_rates (coin, time DESC);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_fills_addr_time ON user_fills (user_address, time DESC);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_positions_addr_time ON positions (user_address, time DESC);")
    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized.")

def insert_many(table, rows, columns):
    """Вставка пачки строк с игнорированием дубликатов."""
    conn = get_conn()
    cur = conn.cursor()
    query = f"INSERT INTO {table} ({','.join(columns)}) VALUES %s ON CONFLICT DO NOTHING"
    execute_values(cur, query, rows)
    conn.commit()
    cur.close()
    conn.close()
