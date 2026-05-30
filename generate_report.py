import os
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
from config import USER_ADDRESSES
from db import DB_PATH

def main():
    conn = sqlite3.connect(DB_PATH)

    # 1. Последние 100 сделок
    trades = pd.read_sql("SELECT * FROM trades ORDER BY time DESC LIMIT 100", conn)
    if not trades.empty:
        trades["time"] = pd.to_datetime(trades["time"])
    fig_trades = px.scatter(trades, x="time", y="price", color="side", size="size",
                            title="Последние 100 сделок")

    # 2. Ликвидации за последние 24 часа
    # В SQLite даты храним как ISO строки, поэтому сравниваем строки
    liq = pd.read_sql("""
        SELECT coin, COUNT(*) as cnt, SUM(size_usd) as total_usd
        FROM liquidations
        WHERE time > datetime('now', '-1 day')
        GROUP BY coin
    """, conn)
    fig_liq = px.bar(liq, x="coin", y="total_usd", title="Объём ликвидаций за 24ч, USD")

    # 3. Текущая ставка финансирования (последняя запись по каждой монете)
    funding = pd.read_sql("""
        SELECT coin, funding_rate
        FROM funding_rates
        WHERE time = (SELECT MAX(time) FROM funding_rates f2 WHERE f2.coin = funding_rates.coin)
        ORDER BY coin
    """, conn)
    fig_funding = px.bar(funding, x="coin", y="funding_rate",
                         title="Текущая ставка финансирования")

    # 4. Win rate для первого адреса (если задан)
    win_text = ""
    if USER_ADDRESSES and USER_ADDRESSES[0] != "0x0000000000000000000000000000000000000000":
        user = USER_ADDRESSES[0]
        fills = pd.read_sql("SELECT pnl FROM user_fills WHERE user_address = ?", conn, params=(user,))
        if not fills.empty:
            wr = (fills["pnl"] > 0).mean()
            win_text = f"<p>Win Rate для {user[:8]}...: <b>{wr:.1%}</b></p>"
        else:
            win_text = "<p>Нет данных по сделкам пользователя.</p>"

    # Собираем HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><title>Hyperliquid Research Dashboard</title></head>
    <body>
        <h1>Hyperliquid On-Chain Analytics</h1>
        <p>Обновлено: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        {win_text}
        <div>{fig_trades.to_html(full_html=False)}</div>
        <div>{fig_liq.to_html(full_html=False)}</div>
        <div>{fig_funding.to_html(full_html=False)}</div>
    </body>
    </html>
    """

    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w", encoding="utf-8") as f:
        f.write(html)

    conn.close()
    print("Report generated: docs/index.html")

if __name__ == "__main__":
    main()
