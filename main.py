import asyncio
from ws_client import listen_ws, get_universe
from rest_client import run_rest_scheduler
from db import init_db

async def main():
    init_db()
    coins = await get_universe()
    # Запускаем параллельно WebSocket и REST-опросы
    await asyncio.gather(
        listen_ws(),
        run_rest_scheduler(coins)
    )

if __name__ == "__main__":
    asyncio.run(main())
