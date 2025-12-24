"""
🔔 ЭТАП 5: Реалтайм-мониторинг и алерты
Следим за новыми транзакциями и отправляем оповещения
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json

# Загружаем ключи
load_dotenv()
API_KEY = os.getenv("ETHERSCAN_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

class TransactionMonitor:
    """Мониторинг новых транзакций"""
    
    def __init__(self, api_key, check_interval=60):
        self.api_key = api_key
        self.check_interval = check_interval  # секунды
        self.last_checked_block = None
        self.whale_addresses = set()
        
        # Загружаем список китов
        self.load_whale_addresses()
    
    def load_whale_addresses(self):
        """Загружаем адреса китов из файла"""
        try:
            whales_df = pd.read_csv('data/whales.csv')
            self.whale_addresses = set(whales_df['address'].tolist())
            print(f"📋 Загружено {len(self.whale_addresses)} адресов китов")
        except:
            print("⚠️  Файл с китами не найден")
            # Примерные адреса для демо
            self.whale_addresses = {
                "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B",
                "0xDA9dfA130Df4dE4673b89022EE50ff26f6EA73Cf"
            }
    
    def get_latest_block(self):
        """Получаем номер последнего блока"""
        url = "https://api.etherscan.io/api"
        params = {
            "module": "proxy",
            "action": "eth_blockNumber",
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            block_number = int(data["result"], 16)  # Конвертируем из hex
            return block_number
        except Exception as e:
            print(f"❌ Ошибка получения блока: {e}")
            return None
    
    def get_block_transactions(self, block_number):
        """Получаем транзакции из блока"""
        url = "https://api.etherscan.io/api"
        params = {
            "module": "proxy",
            "action": "eth_getBlockByNumber",
            "tag": hex(block_number),
            "boolean": "true",
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("result"):
                transactions = data["result"]["transactions"]
                return transactions
            return []
        except Exception as e:
            print(f"❌ Ошибка получения транзакций блока: {e}")
            return []
    
    def process_transaction(self, tx):
        """Обработка одной транзакции"""
        # Конвертируем значение из wei в ETH
        value_wei = int(tx.get("value", "0x0"), 16)
        value_eth = value_wei / 10**18
        
        tx_info = {
            'hash': tx.get("hash", "")[:10] + "...",
            'from': tx.get("from", ""),
            'to': tx.get("to", ""),
            'value_eth': value_eth,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        
        # Проверяем условия для алерта
        alerts = []
        
        # 1. Крупная транзакция (> 100 ETH)
        if value_eth > 100:
            alerts.append(f"💰 КРУПНАЯ ТРАНЗАКЦИЯ: {value_eth:.1f} ETH")
        
        # 2. Транзакция от кита
        if tx_info['from'] in self.whale_addresses:
            alerts.append(f"🐋 КИТ ОТПРАВИЛ: {value_eth:.1f} ETH")
        
        # 3. Транзакция киту
        if tx_info['to'] in self.whale_addresses:
            alerts.append(f"🐋 ПЕРЕВОД КИТУ: {value_eth:.1f} ETH")
        
        # Отправляем алерты если есть
        if alerts:
            message = self.format_alert(tx_info, alerts)
            print(f"\n🔔 НОВЫЙ АЛЕРТ: {message}")
            
            # Отправляем в Telegram если настроено
            if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
                self.send_telegram_alert(message)
    
    def format_alert(self, tx_info, alerts):
        """Форматирование алерта"""
        alert_text = "🚨 | " + " | ".join(alerts) + "\n"
        alert_text += f"📤 От: {tx_info['from'][:12]}...\n"
        alert_text += f"📥 Кому: {tx_info['to'][:12]}...\n"
        alert_text += f"💎 Сумма: {tx_info['value_eth']:.2f} ETH\n"
        alert_text += f"🕒 Время: {tx_info['timestamp']}\n"
        alert_text += f"🔗 Хэш: {tx_info['hash']}"
        
        return alert_text
    
    def send_telegram_alert(self, message):
        """Отправка алерта в Telegram"""
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            return
        
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        
        try:
            response = requests.post(url, json={
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'HTML'
            })
            
            if response.status_code == 200:
                print("✅ Алерт отправлен в Telegram")
            else:
                print(f"❌ Ошибка отправки в Telegram: {response.text}")
        except Exception as e:
            print(f"❌ Ошибка подключения к Telegram: {e}")
    
    def start_monitoring(self, duration_minutes=5):
        """Запуск мониторинга"""
        print("🚀 ЗАПУСК МОНИТОРИНГА...")
        print(f"⏱️  Длительность: {duration_minutes} минут")
        print(f"🔄 Интервал проверки: {self.check_interval} секунд")
        print("📱 Алерты будут отправляться в консоль")
        
        if TELEGRAM_TOKEN:
            print("🤖 Telegram бот подключен")
        else:
            print("⚠️  Telegram не настроен (добавьте токен в .env)")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        blocks_checked = 0
        alerts_sent = 0
        
        # Получаем текущий блок как начальный
        current_block = self.get_latest_block()
        if current_block:
            self.last_checked_block = current_block - 10  # Начинаем с 10 блоков назад
        
        print(f"\n🎯 Начинаю мониторинг с блока {self.last_checked_block}")
        print("=" * 50)
        
        try:
            while datetime.now() < end_time:
                # Получаем текущий блок
                latest_block = self.get_latest_block()
                
                if latest_block and self.last_checked_block:
                    # Проверяем новые блоки
                    for block_num in range(self.last_checked_block + 1, latest_block + 1):
                        print(f"🔍 Проверяю блок {block_num}...", end="\r")
                        
                        transactions = self.get_block_transactions(block_num)
                        
                        for tx in transactions:
                            self.process_transaction(tx)
                            alerts_sent += 1
                        
                        blocks_checked += 1
                        
                        # Небольшая пауза между блоками
                        time.sleep(0.1)
                    
                    self.last_checked_block = latest_block
                
                # Ждем перед следующей проверкой
                print(f"✅ Проверено блоков: {blocks_checked}, алертов: {alerts_sent}", end="\r")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Мониторинг остановлен пользователем")
        
        print(f"\n\n📊 ИТОГИ МОНИТОРИНГА:")
        print(f"   Проверено блоков: {blocks_checked}")
        print(f"   Отправлено алертов: {alerts_sent}")
        print(f"   Время работы: {duration_minutes} минут")


if __name__ == "__main__":
    # Проверяем API ключ
    if not API_KEY or API_KEY == "ваш_ключ_тут":
        print("❌ Сначала добавьте API ключ в .env файл")
        exit()
    
    # Создаем монитор
    monitor = TransactionMonitor(
        api_key=API_KEY,
        check_interval=30  # Проверяем каждые 30 секунд
    )
    
    print("🎯 МОНИТОРИНГ ON-CHAIN ТРАНЗАКЦИЙ")
    print("=" * 50)
    print("Что отслеживаем:")
    print("1. Крупные транзакции (> 100 ETH)")
    print("2. Действия китов (из data/whales.csv)")
    print("3. Переводы китам")
    print("=" * 50)
    
    # Запускаем мониторинг на 5 минут
    # В реальном проекте можно запускать на постоянной основе
    monitor.start_monitoring(duration_minutes=5)
