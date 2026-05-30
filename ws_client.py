import asyncio
import json
import websockets
from datetime import datetime, timezone
import aiohttp
from config import WS_URL, BASE_REST_URL
import db

async def get_universe():
    """Получаем список всех торговых пар с Hyperliquid."""
    async with aiohttp.ClientSession() as session:
        async with session.post(f"{BASE_REST_URL}/info", json={"type": "meta"}) as resp:
            data = await resp.json()
            return [item["name"] for item in data["universe"]]

async def listen_ws():
    coins = await get_universe()
    print(f"Subscribing to trades and liquidations for: {coins}")

    async with websockets.connect(WS_URL) as ws:
        # Подписываемся на сделки и ликвидации по каждой монете
        for coin in coins:
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "trades", "coin": coin}
            }))
            await ws.send(json.dumps({
                "method": "subscribe",
                "subscription": {"type": "liquidations", "coin": coin}
            }))
        print("WebSocket subscriptions active.")

        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            channel = data.get("channel")

            if channel == "trades":
                rows = []
                for trade in data["data"]:
                    ts = datetime.fromtimestamp(trade["time"] / 1000, tz=timezone.utc)
                    rows.append((
                        ts, trade["coin"], trade["side"],
                        float(trade["px"]), float(trade["sz"]), int(trade["tid"])
                    ))
                if rows:
                    db.insert_many("trades", rows, ["time","coin","side","price","size","trade_id"])

            elif channel == "liquidations":
                rows = []
                for liq in data["data"]:
                    ts = datetime.fromtimestamp(liq["time"] / 1000, tz=timezone.utc)
                    rows.append((
                        ts, liq["coin"], liq["side"],
                        float(liq["px"]), float(liq["usdValue"])
                    ))
                if rows:
                    db.insert_many("liquidations", rows, ["time","coin","side","price","size_usd"])
