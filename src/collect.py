"""
📊 ЭТАП 1: Сбор данных с блокчейна
Получаем транзакции с Etherscan API
"""

import requests
import pandas as pd
import time
from datetime import datetime
import os
from dotenv import load_dotenv

# Загружаем ключ из .env файла
load_dotenv()
API_KEY = os.getenv("ETHERSCAN_API_KEY")

class EtherscanCollector:
    """Простой сборщик данных с Etherscan"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.etherscan.io/api"
        
    def get_transactions(self, address, limit=1000):
        """
        Получить транзакции для адреса
        
        Args:
            address (str): Адрес кошелька (начинается с 0x)
            limit (int): Сколько транзакций получить
            
        Returns:
            pandas.DataFrame: Таблица с транзакциями
        """
        print(f"🔍 Собираю транзакции для адреса: {address[:10]}...")
        
        # Параметры запроса
        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "sort": "desc",  # Сначала новые
            "apikey": self.api_key,
            "offset": 10000,  # Максимум за раз
            "page": 1
        }
        
        all_transactions = []
        
        try:
            # Делаем запрос к API
            response = requests.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            if data["status"] == "1":
                transactions = data["result"][:limit]
                print(f"✅ Получено {len(transactions)} транзакций")
                
                # Создаем таблицу
                df = pd.DataFrame(transactions)
                
                # Преобразуем данные
                df = self._process_transactions(df)
                
                return df
            else:
                print(f"❌ Ошибка: {data.get('message', 'Unknown error')}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Ошибка при запросе: {e}")
            return pd.DataFrame()
    
    def _process_transactions(self, df):
        """Обработка сырых данных"""
        if df.empty:
            return df
        
        # Копируем для безопасности
        df = df.copy()
        
        # 1. Конвертируем дату
        df['timestamp'] = pd.to_datetime(df['timeStamp'].astype(int), unit='s')
        df['date'] = df['timestamp'].dt.date
        
        # 2. Конвертируем значения из wei в ETH
        # 1 ETH = 10^18 wei
        df['value_eth'] = df['value'].astype(float) / 10**18
        
        # 3. Конвертируем цену газа
        df['gas_price_gwei'] = df['gasPrice'].astype(float) / 10**9  # Gwei
        
        # 4. Считаем комиссию
        df['gas_used'] = pd.to_numeric(df['gasUsed'], errors='coerce')
        df['gas_price'] = pd.to_numeric(df['gasPrice'], errors='coerce')
        df['tx_fee_eth'] = (df['gas_used'] * df['gas_price']) / 10**18
        
        # 5. Оставляем только нужные колонки
        columns_to_keep = [
            'hash', 'timestamp', 'date', 'from', 'to', 
            'value_eth', 'gas_price_gwei', 'tx_fee_eth',
            'blockNumber', 'isError'
        ]
        
        # Оставляем только существующие колонки
        existing_columns = [col for col in columns_to_keep if col in df.columns]
        df = df[existing_columns]
        
        return df
    
    def get_balance(self, address):
        """Получить текущий баланс ETH"""
        params = {
            "module": "account",
            "action": "balance",
            "address": address,
            "tag": "latest",
            "apikey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            data = response.json()
            balance_wei = int(data["result"])
            balance_eth = balance_wei / 10**18
            return balance_eth
        except:
            return 0.0
    
    def save_to_csv(self, df, filename):
        """Сохранить данные в CSV"""
        if not df.empty:
            # Создаем папку data если её нет
            os.makedirs('data', exist_ok=True)
            
            # Сохраняем
            filepath = f"data/{filename}"
            df.to_csv(filepath, index=False)
            print(f"💾 Данные сохранены в {filepath}")
            return True
        return False


# 🚀 ПРИМЕР ИСПОЛЬЗОВАНИЯ
if __name__ == "__main__":
    # Проверяем API ключ
    if not API_KEY or API_KEY == "ваш_ключ_тут":
        print("❌ Сначала получите API ключ на etherscan.io и добавьте в .env файл")
        print("   Как получить:")
        print("   1. Зайдите на https://etherscan.io/apis")
        print("   2. Нажмите 'Create Account' или войдите")
        print("   3. В My Account -> API Keys создайте ключ")
        print("   4. Скопируйте ключ в файл .env")
        exit()
    
    # Создаем сборщик
    collector = EtherscanCollector(API_KEY)
    
    # Примеры адресов для анализа
    # Можно заменить на свои
    addresses = [
        "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B",  # Vitalik Buterin
        "0xDA9dfA130Df4dE4673b89022EE50ff26f6EA73Cf",  # Крупный кошелек
        "0x742d35Cc6634C0532925a3b844Bc9e0F0d56a8b7",  # Еще один
    ]
    
    all_data = []
    
    for i, address in enumerate(addresses):
        print(f"\n{'='*50}")
        print(f"Анализ адреса {i+1}/{len(addresses)}")
        
        # Получаем транзакции
        df = collector.get_transactions(address, limit=500)
        
        if not df.empty:
            # Добавляем колонку с адресом
            df['wallet_address'] = address
            
            # Получаем баланс
            balance = collector.get_balance(address)
            print(f"💰 Баланс: {balance:.4f} ETH")
            
            # Добавляем в общий список
            all_data.append(df)
            
            # Сохраняем отдельно
            collector.save_to_csv(df, f"transactions_{address[:10]}.csv")
            
            # Небольшая пауза чтобы не заблокировали
            time.sleep(1)
    
    # Сохраняем все данные вместе
    if all_data:
        all_df = pd.concat(all_data, ignore_index=True)
        collector.save_to_csv(all_df, "all_transactions.csv")
        
        # Выводим статистику
        print("\n📊 СТАТИСТИКА:")
        print(f"Всего транзакций: {len(all_df)}")
        print(f"Уникальных кошельков: {all_df['wallet_address'].nunique()}")
        print(f"Общий объем: {all_df['value_eth'].sum():.2f} ETH")
        print(f"Даты с {all_df['date'].min()} по {all_df['date'].max()}")
