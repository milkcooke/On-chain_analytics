"""
🔧 ЭТАП 2: Предобработка данных и создание признаков
Готовим данные для машинного обучения
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Предобработка и создание признаков"""
    
    def __init__(self, transactions_df):
        self.df = transactions_df.copy()
        self.features_list = []
    
    def clean_data(self):
        """Очистка данных"""
        print("🧹 Очищаю данные...")
        
        if self.df.empty:
            return self.df
        
        # 1. Удаляем неудачные транзакции
        if 'isError' in self.df.columns:
            initial_count = len(self.df)
            self.df = self.df[self.df['isError'] == '0']
            print(f"   Удалено неудачных: {initial_count - len(self.df)}")
        
        # 2. Удаляем нулевые транзакции
        if 'value_eth' in self.df.columns:
            initial_count = len(self.df)
            self.df = self.df[self.df['value_eth'] > 0]
            print(f"   Удалено нулевых: {initial_count - len(self.df)}")
        
        # 3. Удаляем дубликаты
        if 'hash' in self.df.columns:
            initial_count = len(self.df)
            self.df = self.df.drop_duplicates(subset=['hash'])
            print(f"   Удалено дубликатов: {initial_count - len(self.df)}")
        
        return self.df
    
    def extract_wallet_features(self, wallet_address):
        """
        Извлекаем признаки для одного кошелька
        
        Признаки (фичи) - это числа, которые описывают поведение кошелька:
        - Сколько транзакций?
        - Какой средний объем?
        - Как часто совершает транзакции?
        и т.д.
        """
        
        # Фильтруем транзакции этого кошелька
        mask = (self.df['from'] == wallet_address) | (self.df['to'] == wallet_address)
        wallet_txs = self.df[mask].copy()
        
        if wallet_txs.empty:
            return None
        
        # Сортируем по времени
        if 'timestamp' in wallet_txs.columns:
            wallet_txs = wallet_txs.sort_values('timestamp')
        
        # Считаем признаки
        features = {
            'address': wallet_address,
        }
        
        # 1. БАЗОВЫЕ ПРИЗНАКИ
        # Входящие и исходящие транзакции
        incoming = wallet_txs[wallet_txs['to'] == wallet_address]
        outgoing = wallet_txs[wallet_txs['from'] == wallet_address]
        
        features['total_tx'] = len(wallet_txs)
        features['incoming_tx'] = len(incoming)
        features['outgoing_tx'] = len(outgoing)
        features['tx_ratio'] = len(incoming) / max(1, len(outgoing))
        
        # 2. ОБЪЕМНЫЕ ПРИЗНАКИ
        if 'value_eth' in wallet_txs.columns:
            features['total_volume'] = wallet_txs['value_eth'].sum()
            features['avg_tx_value'] = wallet_txs['value_eth'].mean()
            features['max_tx_value'] = wallet_txs['value_eth'].max()
            features['median_tx_value'] = wallet_txs['value_eth'].median()
            
            # Объемы входящих/исходящих
            features['incoming_volume'] = incoming['value_eth'].sum() if not incoming.empty else 0
            features['outgoing_volume'] = outgoing['value_eth'].sum() if not outgoing.empty else 0
        
        # 3. ВРЕМЕННЫЕ ПРИЗНАКИ
        if 'timestamp' in wallet_txs.columns:
            wallet_txs['timestamp'] = pd.to_datetime(wallet_txs['timestamp'])
            
            # Когда первая и последняя транзакция
            first_tx = wallet_txs['timestamp'].min()
            last_tx = wallet_txs['timestamp'].max()
            
            features['first_tx_date'] = first_tx
            features['last_tx_date'] = last_tx
            features['wallet_age_days'] = (datetime.now() - first_tx).days
            features['days_since_last_tx'] = (datetime.now() - last_tx).days
            
            # Частота транзакций
            if features['wallet_age_days'] > 0:
                features['tx_per_day'] = features['total_tx'] / features['wallet_age_days']
            else:
                features['tx_per_day'] = 0
            
            # Время между транзакциями
            if len(wallet_txs) > 1:
                time_diffs = wallet_txs['timestamp'].diff().dt.total_seconds().dropna()
                features['avg_time_between_tx_hours'] = time_diffs.mean() / 3600
        
        # 4. ПРИЗНАКИ АКТИВНОСТИ
        # Уникальные контрагенты
        features['unique_counterparties'] = len(set(list(wallet_txs['from']) + list(wallet_txs['to']))) - 1
        
        # Активность по времени суток (если есть timestamp)
        if 'timestamp' in wallet_txs.columns:
            # Ночные транзакции (0-6 утра)
            night_hours = list(range(0, 6))
            night_txs = wallet_txs[wallet_txs['timestamp'].dt.hour.isin(night_hours)]
            features['night_tx_ratio'] = len(night_txs) / max(1, len(wallet_txs))
        
        # 5. ПРИЗНАКИ ДЛЯ КЛАСТЕРИЗАЦИИ
        # Признак "кит" - много транзакций и большой объем
        features['whale_score'] = 0
        if features.get('total_volume', 0) > 100:  # Больше 100 ETH
            features['whale_score'] += 1
        if features.get('max_tx_value', 0) > 50:   # Есть крупные транзакции
            features['whale_score'] += 1
        if features.get('total_tx', 0) > 100:      # Много транзакций
            features['whale_score'] += 1
        
        # Признак "спящий" - давно не было активности
        features['is_sleeping'] = 1 if features.get('days_since_last_tx', 0) > 30 else 0
        
        # Признак "активный" - много транзакций в день
        features['is_active'] = 1 if features.get('tx_per_day', 0) > 1 else 0
        
        return features
    
    def create_feature_matrix(self):
        """Создаем матрицу признаков для всех кошельков"""
        print("🔧 Создаю матрицу признаков...")
        
        if self.df.empty:
            return pd.DataFrame()
        
        # Находим все уникальные адреса
        all_addresses = set(list(self.df['from']) + list(self.df['to']))
        print(f"   Всего уникальных адресов: {len(all_addresses)}")
        
        # Ограничим для примера
        addresses_to_process = list(all_addresses)[:50]  # Первые 50
        
        # Извлекаем признаки для каждого адреса
        features_list = []
        for i, address in enumerate(addresses_to_process):
            features = self.extract_wallet_features(address)
            if features:
                features_list.append(features)
            
            # Прогресс
            if (i + 1) % 10 == 0:
                print(f"   Обработано {i+1}/{len(addresses_to_process)} адресов")
        
        # Создаем DataFrame с признаками
        features_df = pd.DataFrame(features_list)
        
        # Сохраняем
        if not features_df.empty:
            features_df.to_csv('data/wallet_features.csv', index=False)
            print(f"💾 Матрица признаков сохранена: {len(features_df)} кошельков")
        
        return features_df


if __name__ == "__main__":
    # Загружаем данные из предыдущего этапа
    try:
        df = pd.read_csv('data/all_transactions.csv')
        print(f"📁 Загружено {len(df)} транзакций")
    except:
        print("❌ Сначала запустите collect.py для сбора данных")
        exit()
    
    # Создаем препроцессор
    preprocessor = DataPreprocessor(df)
    
    # Очищаем данные
    cleaned_df = preprocessor.clean_data()
    print(f"✅ Очищено транзакций: {len(cleaned_df)}")
    
    # Создаем матрицу признаков
    features_df = preprocessor.create_feature_matrix()
    
    if not features_df.empty:
        # Показываем пример признаков
        print("\n📋 ПРИМЕР ПРИЗНАКОВ (первые 5 кошельков):")
        print(features_df[['address', 'total_tx', 'total_volume', 'tx_per_day', 'whale_score']].head())
        
        print("\n📊 СТАТИСТИКА ПРИЗНАКОВ:")
        print(f"Китов (whale_score >= 2): {len(features_df[features_df['whale_score'] >= 2])}")
        print(f"Спящих: {features_df['is_sleeping'].sum()}")
        print(f"Активных: {features_df['is_active'].sum()}")
