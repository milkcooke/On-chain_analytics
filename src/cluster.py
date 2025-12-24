"""
🎯 ЭТАП 3: Кластеризация кошельков
Группируем кошельки по поведению с помощью ML
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Для красивого отображения
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class WalletClustering:
    """Кластеризация кошельков"""
    
    def __init__(self, features_df):
        self.df = features_df.copy()
        self.scaler = StandardScaler()
        self.cluster_labels = None
    
    def prepare_features(self):
        """Подготовка признаков для кластеризации"""
        print("🔧 Подготавливаю признаки...")
        
        # Выбираем числовые признаки
        numeric_cols = [
            'total_tx', 'total_volume', 'avg_tx_value',
            'tx_per_day', 'days_since_last_tx', 'whale_score'
        ]
        
        # Оставляем только существующие колонки
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if len(available_cols) < 2:
            print("❌ Недостаточно признаков для кластеризации")
            return None
        
        # Создаем матрицу признаков
        X = self.df[available_cols].copy()
        
        # Заполняем пропуски
        X = X.fillna(X.median())
        
        # Масштабируем (очень важно для кластеризации!)
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"✅ Используется {len(available_cols)} признаков")
        return X_scaled, available_cols
    
    def find_optimal_clusters(self, X, max_k=10):
        """Находим оптимальное число кластеров"""
        print("🔍 Ищу оптимальное число кластеров...")
        
        inertias = []
        silhouette_scores = []
        
        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            
            inertias.append(kmeans.inertia_)
            
            # Оценка качества
            if len(set(kmeans.labels_)) > 1:
                score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append(score)
            else:
                silhouette_scores.append(0)
        
        # Метод локтя: ищем "изгиб" на графике
        # Простой способ - выбираем где silhouette_score максимальный
        best_k = np.argmax(silhouette_scores) + 2  # +2 потому что начинаем с 2
        
        print(f"✅ Оптимальное число кластеров: {best_k}")
        return best_k
    
    def kmeans_clustering(self, n_clusters=None):
        """Кластеризация K-means"""
        print(f"🎯 Запускаю K-means кластеризацию...")
        
        # Подготавливаем данные
        X, feature_names = self.prepare_features()
        if X is None:
            return None
        
        # Находим оптимальное число кластеров если не задано
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(X)
        
        # Запускаем K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Добавляем метки в DataFrame
        self.df['cluster'] = labels
        self.cluster_labels = labels
        
        # Анализируем кластеры
        self.analyze_clusters()
        
        return labels
    
    def analyze_clusters(self):
        """Анализ получившихся кластеров"""
        print("\n📊 АНАЛИЗ КЛАСТЕРОВ:")
        print("=" * 50)
        
        if 'cluster' not in self.df.columns:
            print("❌ Кластеризация не выполнена")
            return
        
        # Группируем по кластерам
        cluster_stats = self.df.groupby('cluster').agg({
            'total_volume': ['count', 'mean', 'sum'],
            'total_tx': 'mean',
            'tx_per_day': 'mean',
            'days_since_last_tx': 'mean',
            'whale_score': 'mean'
        }).round(2)
        
        print(cluster_stats)
        
        # Присваиваем названия кластерам
        cluster_names = {}
        for cluster_id in self.df['cluster'].unique():
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            
            # Определяем тип кластера по статистике
            avg_volume = cluster_data['total_volume'].mean()
            avg_tx = cluster_data['total_tx'].mean()
            days_inactive = cluster_data['days_since_last_tx'].mean()
            
            if avg_volume > 100 and avg_tx > 50:
                name = "🐋 Киты"
            elif avg_tx > 20 and avg_volume < 10:
                name = "🧑‍🌾 Фермеры"
            elif days_inactive > 30:
                name = "💤 Спящие"
            elif cluster_data['whale_score'].mean() > 1.5:
                name = "🧠 Инсайдеры"
            elif avg_tx < 5:
                name = "🐢 Редкие"
            else:
                name = "📊 Обычные"
            
            cluster_names[cluster_id] = name
        
        # Добавляем названия
        self.df['cluster_name'] = self.df['cluster'].map(cluster_names)
        
        print("\n🏷️ НАЗВАНИЯ КЛАСТЕРОВ:")
        for cluster_id, name in cluster_names.items():
            count = len(self.df[self.df['cluster'] == cluster_id])
            print(f"  Кластер {cluster_id}: {name} ({count} кошельков)")
    
    def visualize_clusters(self):
        """Визуализация кластеров"""
        if 'cluster_name' not in self.df.columns:
            print("❌ Сначала выполните кластеризацию")
            return
        
        print("\n🎨 Создаю визуализации...")
        
        # 1. Распределение по кластерам
        plt.figure(figsize=(10, 6))
        cluster_counts = self.df['cluster_name'].value_counts()
        colors = plt.cm.Set3(np.arange(len(cluster_counts)))
        
        plt.pie(cluster_counts.values, labels=cluster_counts.index, 
                autopct='%1.1f%%', colors=colors, startangle=90)
        plt.title('Распределение кошельков по кластерам')
        plt.savefig('data/cluster_distribution.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        # 2. Объем транзакций по кластерам
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        sns.boxplot(data=self.df, x='cluster_name', y='total_volume')
        plt.xticks(rotation=45, ha='right')
        plt.title('Объем транзакций по кластерам')
        plt.ylabel('Объем (ETH)')
        plt.yscale('log')  # Логарифмическая шкала
        
        plt.subplot(1, 2, 2)
        sns.boxplot(data=self.df, x='cluster_name', y='total_tx')
        plt.xticks(rotation=45, ha='right')
        plt.title('Количество транзакций по кластерам')
        plt.ylabel('Количество')
        
        plt.tight_layout()
        plt.savefig('data/cluster_comparison.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        # 3. Топ-10 китов
        if '🐋 Киты' in self.df['cluster_name'].values:
            whales = self.df[self.df['cluster_name'] == '🐋 Киты']
            top_whales = whales.nlargest(10, 'total_volume')
            
            plt.figure(figsize=(12, 6))
            bars = plt.barh(range(len(top_whales)), top_whales['total_volume'])
            plt.yticks(range(len(top_whales)), [addr[:12]+'...' for addr in top_whales['address']])
            plt.xlabel('Объем транзакций (ETH)')
            plt.title('Топ-10 китов по объему')
            plt.gca().invert_yaxis()  # Самый большой сверху
            
            # Добавляем значения на столбцы
            for i, bar in enumerate(bars):
                plt.text(bar.get_width() * 0.01, bar.get_y() + bar.get_height()/2,
                        f'{bar.get_width():.1f} ETH', va='center')
            
            plt.tight_layout()
            plt.savefig('data/top_whales.png', dpi=100, bbox_inches='tight')
            plt.show()
    
    def save_results(self):
        """Сохранение результатов"""
        if not self.df.empty:
            self.df.to_csv('data/clustered_wallets.csv', index=False)
            print("💾 Результаты сохранены в data/clustered_wallets.csv")
            
            # Сохраняем отдельно китов
            if 'cluster_name' in self.df.columns:
                whales = self.df[self.df['cluster_name'] == '🐋 Киты']
                if not whales.empty:
                    whales.to_csv('data/whales.csv', index=False)
                    print(f"💾 Список китов сохранен: {len(whales)} кошельков")


if __name__ == "__main__":
    # Загружаем признаки из предыдущего этапа
    try:
        features_df = pd.read_csv('data/wallet_features.csv')
        print(f"📁 Загружено {len(features_df)} кошельков с признаками")
    except:
        print("❌ Сначала запустите preprocess.py для создания признаков")
        exit()
    
    # Создаем кластеризатор
    clustering = WalletClustering(features_df)
    
    # Запускаем кластеризацию
    labels = clustering.kmeans_clustering(n_clusters=6)
    
    if labels is not None:
        # Визуализируем результаты
        clustering.visualize_clusters()
        
        # Сохраняем
        clustering.save_results()
        
        print("\n✅ Кластеризация завершена!")
        print("📊 Созданы файлы:")
        print("   - data/clustered_wallets.csv - все кошельки с кластерами")
        print("   - data/whales.csv - список китов")
        print("   - data/cluster_*.png - графики анализа")
