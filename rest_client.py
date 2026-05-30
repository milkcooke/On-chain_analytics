import asyncio
import aiohttp
from datetime import datetime, timezone
from config import BASE_REST_URL, USER_ADDRESSES, \
    FUNDING_INTERVAL, USER_FILLS_INTERVAL, POSITIONS_INTERVAL
import db

async def fetch_json(session, url, payload=None):
    async with session.post(url, json=payload) as resp:
        return await resp.json()

async def run_rest_scheduler(coins):
    async with aiohttp.ClientSession() as session:
        while True:
            # 1. Funding rates
            for coin in coins:
                end = int(datetime.now(timezone.utc).timestamp() * 1000)
                start = end - FUNDING_INTERVAL * 1000
                payload = {"type": "fundingHistory", "coin": coin, "startTime": start, "endTime": end}
                data = await fetch_json(session, f"{BASE_REST_URL}/info", payload)
                if isinstance(data, list):
                    rows = []
                    for entry in data:
                        ts = datetime.fromtimestamp(entry["time"] / 1000, tz=timezone.utc)
                        rows.append((ts, coin, float(entry["fundingRate"])))
                    if rows:
                        db.insert_many("funding_rates", rows, ["time","coin","funding_rate"])

            # 2. Open Interest
            payload = {"type": "openInterest"}
            data = await fetch_json(session, f"{BASE_REST_URL}/info", payload)
            if isinstance(data, list):
                rows = []
                ts = datetime.now(timezone.utc)
                for item in data:
                    if item["coin"] in coins:
                        rows.append((ts, item["coin"], float(item["openInterest"])))
                if rows:
                    db.insert_many("open_interest", rows, ["time","coin","open_interest"])

            # 3. User fills и позиции для указанных адресов
            if USER_ADDRESSES and USER_ADDRESSES[0] != "0x0000000000000000000000000000000000000000":
                for addr in USER_ADDRESSES:
                    # Сделки пользователя
                    payload = {"type": "userFills", "user": addr}
                    data = await fetch_json(session, f"{BASE_REST_URL}/info", payload)
                    if isinstance(data, list):
                        rows = []
                        for fill in data:
                            ts = datetime.fromtimestamp(fill["time"] / 1000, tz=timezone.utc)
                            rows.append((
                                ts, addr, fill["coin"], fill["side"],
                                float(fill["px"]), float(fill["sz"]),
                                float(fill["fee"]), float(fill.get("pnl", 0.0))
                            ))
                        if rows:
                            db.insert_many("user_fills", rows,
                                ["time","user_address","coin","side","price","size","fee","pnl"])

                    # Открытые позиции
                    payload = {"type": "clearinghouseState", "user": addr}
                    data = await fetch_json(session, f"{BASE_REST_URL}/info", payload)
                    if "assetPositions" in data:
                        rows = []
                        ts = datetime.now(timezone.utc)
                        for pos in data["assetPositions"]:
                            p = pos["position"]
                            rows.append((
                                ts, addr, p["coin"], float(p["szi"]),
                                float(p["entryPx"]), float(p.get("leverage", {}).get("value", 0))
                            ))
                        if rows:
                            db.insert_many("positions", rows,
                                ["time","user_address","coin","size","entry_price","leverage"])

            # Ждём перед следующим циклом
            await asyncio.sleep(USER_FILLS_INTERVAL)  # 10 минут
