import os
from dotenv import load_dotenv

load_dotenv()

# Hyperliquid API (публичные, ключи не нужны)
BASE_REST_URL = "https://api.hyperliquid.xyz"
WS_URL = "wss://api.hyperliquid.xyz/ws"

# База данных
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://hyperuser:hyperpass@localhost:5432/hyperliquid")

# Адреса для сбора позиций и сделок (можешь оставить любой для теста)
USER_ADDRESSES = os.getenv("USER_ADDRESSES", "0x0000000000000000000000000000000000000000").split(",")

# Тайминги REST-запросов (в секундах)
FUNDING_INTERVAL = 3600       # раз в час
OI_INTERVAL = 3600
USER_FILLS_INTERVAL = 600    # раз в 10 минут
POSITIONS_INTERVAL = 600

# Telegram (если будешь использовать)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
