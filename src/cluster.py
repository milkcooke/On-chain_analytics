"""
🎯 ЭТАП 3: Кластеризация кошельков с определением ВСЕХ типов
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
    """Кластеризация кошельков с определением всех типов"""
    
    def __init__(self, features_df):
        self.df = features_df.copy()
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.wallet_types = {}  # Будем хранить типы кошельков
    
    def prepare_features(self):
        """Подготовка признаков для кластеризации"""
        print("🔧 Подготавливаю признаки...")
        
        # Сначала добавим дополнительные признаки для лучшего определения типов
        self._add_special_features()
        
        # Выбираем числовые признаки для кластеризации
        numeric_cols = [
            'total_tx', 'total_volume', 'avg_tx_value',
            'tx_per_day', 'days_since_last_tx', 'whale_score',
            'incoming_tx', 'outgoing_tx', 'unique_counterparties',
            'avg_time_between_tx_hours', 'night_tx_ratio',
            'tx_variability_score'  # Новый признак
        ]
        
        # Оставляем только существующие колонки
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if len(available_cols) < 2:
            print("Недостаточно признаков для кластеризации")
            return None
        
        # Создаем матрицу признаков
        X = self.df[available_cols].copy()
        
        # Заполняем пропуски
        X = X.fillna(X.median())
        
        # Масштабируем (очень важно для кластеризации!)
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Используется {len(available_cols)} признаков")
        return X_scaled, available_cols
    
    def _add_special_features(self):
        """Добавляем специальные признаки для определения типов кошельков"""
        print("➕ Добавляю специальные признаки...")
        
        # 1. Признак "активность ночью" (для инсайдеров)
        if 'night_tx_ratio' not in self.df.columns:
            # Упрощенная версия - если много транзакций при небольшом объеме
            self.df['night_tx_ratio'] = np.random.random(len(self.df)) * 0.3  # Заглушка
        
        # 2. Признак "скорость реакции" (для снайперов)
        # Предполагаем, что снайперы делают транзакции быстро друг за другом
        if 'avg_time_between_tx_hours' in self.df.columns:
            self.df['reaction_speed'] = 1 / (self.df['avg_time_between_tx_hours'] + 1)
        else:
            self.df['reaction_speed'] = 0.5
        
        # 3. Признак "разнообразие взаимодействий" (для фермеров)
        if 'unique_counterparties' in self.df.columns and 'total_tx' in self.df.columns:
            self.df['interaction_diversity'] = self.df['unique_counterparties'] / (self.df['total_tx'] + 1)
        else:
            self.df['interaction_diversity'] = 0.3
        
        # 4. Признак "вариабельность транзакций" (паттерны поведения)
        if 'avg_tx_value' in self.df.columns and 'total_volume' in self.df.columns:
            # Коэффициент вариации объема транзакций
            self.df['tx_variability_score'] = np.random.random(len(self.df))  # Заглушка
        else:
            self.df['tx_variability_score'] = 0.5
        
        # 5. Признак "временной паттерн" (когда чаще всего активен)
        self.df['time_pattern_score'] = np.random.random(len(self.df))
    
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
        # Выбираем где silhouette_score максимальный
        best_k = np.argmax(silhouette_scores) + 2
        
        print(f"Оптимальное число кластеров: {best_k}")
        
        # Визуализация выбора числа кластеров
        self._plot_cluster_selection(inertias, silhouette_scores)
        
        return best_k
    
    def _plot_cluster_selection(self, inertias, silhouette_scores):
        """Визуализация выбора оптимального числа кластеров"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Метод локтя
        ax1.plot(range(2, len(inertias) + 2), inertias, 'bo-')
        ax1.set_xlabel('Количество кластеров')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Метод локтя')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette score
        ax2.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, 'ro-')
        ax2.set_xlabel('Количество кластеров')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Метод силуэта')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/cluster_selection.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def kmeans_clustering(self, n_clusters=None):
        """Кластеризация K-means"""
        print(f"Запускаю K-means кластеризацию...")
        
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
        self.kmeans_model = kmeans
        
        # Анализируем кластеры и определяем типы
        self.analyze_and_label_clusters()
        
        return labels
    
    def analyze_and_label_clusters(self):
        """Анализ кластеров и определение типов кошельков"""
        print("\n📊 АНАЛИЗ КЛАСТЕРОВ:")
        print("=" * 50)
        
        if 'cluster' not in self.df.columns:
            print("Кластеризация не выполнена")
            return
        
        # Группируем по кластерам
        cluster_stats = self.df.groupby('cluster').agg({
            'total_volume': ['count', 'mean', 'sum'],
            'total_tx': 'mean',
            'avg_tx_value': 'mean',
            'tx_per_day': 'mean',
            'days_since_last_tx': 'mean',
            'whale_score': 'mean',
            'reaction_speed': 'mean',
            'interaction_diversity': 'mean',
            'night_tx_ratio': 'mean'
        }).round(2)
        
        print(cluster_stats)
        
        # Определяем тип для каждого кластера
        cluster_names = {}
        cluster_descriptions = {}
        
        for cluster_id in self.df['cluster'].unique():
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            
            # Статистика кластера
            avg_volume = cluster_data['total_volume'].mean()
            avg_tx = cluster_data['total_tx'].mean()
            avg_tx_value = cluster_data['avg_tx_value'].mean()
            days_inactive = cluster_data['days_since_last_tx'].mean()
            whale_score = cluster_data['whale_score'].mean()
            reaction_speed = cluster_data['reaction_speed'].mean() if 'reaction_speed' in cluster_data.columns else 0.5
            interaction_diversity = cluster_data['interaction_diversity'].mean() if 'interaction_diversity' in cluster_data.columns else 0.3
            night_activity = cluster_data['night_tx_ratio'].mean() if 'night_tx_ratio' in cluster_data.columns else 0.1
            
            # Определяем тип кошелька
            wallet_type, description = self._determine_wallet_type(
                avg_volume, avg_tx, avg_tx_value, days_inactive,
                whale_score, reaction_speed, interaction_diversity, night_activity
            )
            
            cluster_names[cluster_id] = wallet_type
            cluster_descriptions[cluster_id] = description
            
            # Сохраняем в словарь типов
            self.wallet_types[cluster_id] = {
                'type': wallet_type,
                'description': description,
                'count': len(cluster_data),
                'avg_volume': avg_volume,
                'avg_tx': avg_tx
            }
        
        # Добавляем названия и описания в DataFrame
        self.df['cluster_name'] = self.df['cluster'].map(cluster_names)
        self.df['cluster_description'] = self.df['cluster'].map(cluster_descriptions)
        
        print("\n🏷️ ОПРЕДЕЛЕННЫЕ ТИПЫ КОШЕЛЬКОВ:")
        print("=" * 50)
        for cluster_id in sorted(cluster_names.keys()):
            info = self.wallet_types[cluster_id]
            print(f"\nКластер {cluster_id}: {info['type']}")
            print(f"   Количество: {info['count']} кошельков")
            print(f"   Средний объем: {info['avg_volume']:.2f} ETH")
            print(f"   Среднее количество TX: {info['avg_tx']:.1f}")
            print(f"   Описание: {info['description']}")
    
    def _determine_wallet_type(self, avg_volume, avg_tx, avg_tx_value, days_inactive,
                              whale_score, reaction_speed, interaction_diversity, night_activity):
        """Определение типа кошелька на основе характеристик"""
        
        # 🐋 Киты - крупные игроки
        if avg_volume > 500 or avg_tx_value > 100 or whale_score >= 2.5:
            return "🐋 Киты", "Крупные игроки с большим объемом транзакций"
        
        # 🧠 Инсайдеры - покупают перед пампами
        # Характеристики: высокая ночная активность, средний объем, хороший timing
        elif night_activity > 0.4 and avg_tx_value > 10 and reaction_speed > 0.7:
            return "🧠 Инсайдеры", "Покупают перед ростом, активны в нерабочее время"
        
        # ⚡ Снайперы - ловят токены при запуске
        # Характеристики: быстрая реакция, небольшой объем, много транзакций
        elif reaction_speed > 0.8 and avg_tx > 30 and avg_volume < 50:
            return "⚡ Снайперы", "Быстро реагируют на новые возможности, много мелких транзакций"
        
        # 🧑‍🌾 Фермеры - взаимодействуют с большим числом контрактов
        # Характеристики: высокое разнообразие взаимодействий
        elif interaction_diversity > 0.6 and avg_tx > 20:
            return "🧑‍🌾 Фермеры", "Взаимодействуют со многими контрактами, активно в DeFi"
        
        # 💤 Спящие - давно неактивные кошельки
        elif days_inactive > 90 or avg_tx < 2:
            return "💤 Спящие", "Давно неактивные или малоактивные кошельки"
        
        # 📊 Обычные пользователи
        elif avg_tx > 5 and avg_volume < 20:
            return "📊 Обычные", "Стандартные пользователи, умеренная активность"
        
        # 🎯 Трейдеры
        elif avg_tx > 15 and 10 < avg_tx_value < 100:
            return "🎯 Трейдеры", "Активные трейдеры, средний объем операций"
        
        # 🔍 Исследователи
        elif interaction_diversity > 0.4 and avg_tx > 10:
            return "🔍 Исследователи", "Тестируют разные протоколы и контракты"
        
        # По умолчанию
        else:
            return "Неизвестный", "Не удалось определить четкий тип"
    
    def visualize_clusters(self):
        """Визуализация кластеров"""
        if 'cluster_name' not in self.df.columns:
            print("Сначала выполните кластеризацию")
            return
        
        print("\nСоздаю визуализации...")
        
        # 1. Распределение по типам кошельков
        self._plot_wallet_types_distribution()
        
        # 2. Характеристики кластеров
        self._plot_cluster_characteristics()
        
        # 3. Топ-10 по каждому типу
        self._plot_top_wallets_by_type()
        
        # 4. Матрица характеристик
        self._plot_characteristics_matrix()
    
    def _plot_wallet_types_distribution(self):
        """Распределение кошельков по типам"""
        plt.figure(figsize=(14, 8))
        
        # Распределение по типам
        type_counts = self.df['cluster_name'].value_counts()
        
        # Сортируем по убыванию
        type_counts = type_counts.sort_values(ascending=True)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
        
        plt.subplot(1, 2, 1)
        bars = plt.barh(range(len(type_counts)), type_counts.values, color=colors)
        plt.yticks(range(len(type_counts)), type_counts.index)
        plt.xlabel('Количество кошельков')
        plt.title('Распределение кошельков по типам')
        
        # Добавляем значения на столбцы
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + bar.get_width()*0.01, bar.get_y() + bar.get_height()/2,
                    f'{int(bar.get_width())}', va='center')
        
        # Круговая диаграмма
        plt.subplot(1, 2, 2)
        plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        plt.title('Процентное распределение')
        
        plt.tight_layout()
        plt.savefig('data/wallet_types_distribution.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def _plot_cluster_characteristics(self):
        """Визуализация характеристик кластеров"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Средний объем по типам
        volume_by_type = self.df.groupby('cluster_name')['total_volume'].mean().sort_values(ascending=False)
        axes[0, 0].bar(range(len(volume_by_type)), volume_by_type.values)
        axes[0, 0].set_xticks(range(len(volume_by_type)))
        axes[0, 0].set_xticklabels(volume_by_type.index, rotation=45, ha='right')
        axes[0, 0].set_title('Средний объем по типам')
        axes[0, 0].set_ylabel('Объем (ETH)')
        axes[0, 0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        
        # 2. Среднее количество транзакций
        tx_by_type = self.df.groupby('cluster_name')['total_tx'].mean().sort_values(ascending=False)
        axes[0, 1].bar(range(len(tx_by_type)), tx_by_type.values, color='orange')
        axes[0, 1].set_xticks(range(len(tx_by_type)))
        axes[0, 1].set_xticklabels(tx_by_type.index, rotation=45, ha='right')
        axes[0, 1].set_title('Среднее количество транзакций')
        axes[0, 1].set_ylabel('Количество транзакций')
        
        # 3. Дни с последней транзакции
        if 'days_since_last_tx' in self.df.columns:
            inactive_by_type = self.df.groupby('cluster_name')['days_since_last_tx'].mean().sort_values(ascending=False)
            axes[1, 0].bar(range(len(inactive_by_type)), inactive_by_type.values, color='green')
            axes[1, 0].set_xticks(range(len(inactive_by_type)))
            axes[1, 0].set_xticklabels(inactive_by_type.index, rotation=45, ha='right')
            axes[1, 0].set_title('Дней с последней транзакции')
            axes[1, 0].set_ylabel('Дни')
        
        # 4. Whale score по типам
        if 'whale_score' in self.df.columns:
            whale_by_type = self.df.groupby('cluster_name')['whale_score'].mean().sort_values(ascending=False)
            axes[1, 1].bar(range(len(whale_by_type)), whale_by_type.values, color='red')
            axes[1, 1].set_xticks(range(len(whale_by_type)))
            axes[1, 1].set_xticklabels(whale_by_type.index, rotation=45, ha='right')
            axes[1, 1].set_title('Whale Score по типам')
            axes[1, 1].set_ylabel('Whale Score (0-3)')
        
        plt.tight_layout()
        plt.savefig('data/cluster_characteristics.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def _plot_top_wallets_by_type(self):
        """Топ кошельков по каждому типу"""
        wallet_types = self.df['cluster_name'].unique()
        
        for wallet_type in wallet_types:
            if wallet_type == "💤 Спящие":
                continue  # Пропускаем спящих
            
            wallets_of_type = self.df[self.df['cluster_name'] == wallet_type]
            
            if len(wallets_of_type) > 0:
                # Топ-5 по объему
                top_wallets = wallets_of_type.nlargest(5, 'total_volume')
                
                plt.figure(figsize=(12, 6))
                bars = plt.barh(range(len(top_wallets)), top_wallets['total_volume'])
                plt.yticks(range(len(top_wallets)), [f"{addr[:10]}..." for addr in top_wallets['address']])
                plt.xlabel('Объем транзакций (ETH)')
                plt.title(f'Топ-5 {wallet_type} по объему')
                plt.gca().invert_yaxis()
                
                # Добавляем значения
                for i, bar in enumerate(bars):
                    plt.text(bar.get_width() * 0.01, bar.get_y() + bar.get_height()/2,
                            f'{bar.get_width():.1f} ETH', va='center')
                
                plt.tight_layout()
                filename = f"data/top_{wallet_type.replace(' ', '_').replace('🐋', 'whale').replace('🧠', 'insider').replace('⚡', 'sniper').replace('🧑‍🌾', 'farmer')}.png"
                plt.savefig(filename, dpi=100, bbox_inches='tight')
                plt.show()
    
    def _plot_characteristics_matrix(self):
        """Матрица характеристик типов кошельков"""
        # Создаем сводную таблицу характеристик
        characteristics = ['total_volume', 'total_tx', 'days_since_last_tx', 'whale_score']
        available_chars = [c for c in characteristics if c in self.df.columns]
        
        if len(available_chars) >= 2:
            # Берем две характеристики для визуализации
            plt.figure(figsize=(10, 8))
            
            scatter = plt.scatter(self.df[available_chars[0]], 
                                 self.df[available_chars[1]],
                                 c=self.df['cluster'].astype('category').cat.codes,
                                 cmap='tab20', alpha=0.6, s=50)
            
            plt.xlabel(available_chars[0])
            plt.ylabel(available_chars[1])
            plt.title(f'Матрица характеристик: {available_chars[0]} vs {available_chars[1]}')
            plt.xscale('log')
            plt.yscale('log')
            
            # Добавляем легенду с типами
            unique_clusters = self.df['cluster_name'].unique()
            for i, cluster_type in enumerate(unique_clusters):
                plt.scatter([], [], color=plt.cm.tab20(i/len(unique_clusters)), 
                           label=cluster_type, alpha=0.6)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('data/characteristics_matrix.png', dpi=100, bbox_inches='tight')
            plt.show()
    
    def save_results(self):
        """Сохранение результатов"""
        if not self.df.empty:
            # Сохраняем все кошельки с кластерами
            self.df.to_csv('data/clustered_wallets.csv', index=False)
            print("💾 Все кошельки сохранены: data/clustered_wallets.csv")
            
            # Сохраняем отдельно по типам
            wallet_types = self.df['cluster_name'].unique()
            
            for wallet_type in wallet_types:
                wallets_of_type = self.df[self.df['cluster_name'] == wallet_type]
                if not wallets_of_type.empty:
                    filename = wallet_type.replace(' ', '_').replace('🐋', 'whales').replace('🧠', 'insiders').replace('⚡', 'snipers').replace('🧑‍🌾', 'farmers').replace('💤', 'sleepers').replace('📊', 'regular').replace('🎯', 'traders').replace('🔍', 'explorers').replace('❓', 'unknown')
                    wallets_of_type.to_csv(f'data/{filename}.csv', index=False)
                    print(f"💾 {wallet_type}: {len(wallets_of_type)} кошельков -> data/{filename}.csv")
            
            # Сохраняем статистику по типам
            type_stats = self.df.groupby('cluster_name').agg({
                'total_volume': ['count', 'mean', 'sum', 'min', 'max'],
                'total_tx': 'mean',
                'days_since_last_tx': 'mean'
            }).round(2)
            
            type_stats.to_csv('data/wallet_types_statistics.csv')
            print("💾 Статистика по типам: data/wallet_types_statistics.csv")
            
            # Сохраняем подробную информацию о кластерах
            clusters_info = pd.DataFrame.from_dict(self.wallet_types, orient='index')
            clusters_info.to_csv('data/clusters_detailed_info.csv')
            print("💾 Детальная информация о кластерах: data/clusters_detailed_info.csv")


# 🚀 ПРИМЕР ИСПОЛЬЗОВАНИЯ
if __name__ == "__main__":
    print("=" * 60)
    print("КЛАСТЕРИЗАЦИЯ КОШЕЛЬКОВ ПО ТИПАМ")
    print("=" * 60)
    
    # Загружаем признаки из предыдущего этапа
    try:
        features_df = pd.read_csv('data/wallet_features.csv')
        print(f"Загружено {len(features_df)} кошельков с признаками")
    except FileNotFoundError:
        print("Файл wallet_features.csv не найден!")
        print("\nЧто делать:")
        print("1. Сначала запустите collect.py для сбора данных")
        print("2. Затем запустите preprocess.py для создания признаков")
        print("3. После этого запустите cluster.py")
        exit()
    except Exception as e:
        print(f"Ошибка загрузки файла: {e}")
        exit()
    
    # Проверяем что есть нужные колонки
    required_cols = ['total_tx', 'total_volume', 'days_since_last_tx']
    missing_cols = [col for col in required_cols if col not in features_df.columns]
    
    if missing_cols:
        print(f"❌ Отсутствуют колонки: {missing_cols}")
        print("   Запустите preprocess.py для создания признаков")
        exit()
    
    # Создаем кластеризатор
    clustering = WalletClustering(features_df)
    
    # Запускаем кластеризацию
    print("\n🚀 Запускаю кластеризацию...")
    labels = clustering.kmeans_clustering(n_clusters=8)  # 8 кластеров для лучшего разделения
    
    if labels is not None:
        print("\nКластеризация выполнена успешно!")
        
        # Визуализируем результаты
        clustering.visualize_clusters()
        
        # Сохраняем результаты
        clustering.save_results()
        
        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ:")
        print("=" * 60)
        
        # Показываем сводку
        wallet_types_summary = features_df['cluster_name'].value_counts() if 'cluster_name' in features_df.columns else pd.Series()
        
        for wallet_type in ["🐋 Киты", "🧠 Инсайдеры", "⚡ Снайперы", "🧑‍🌾 Фермеры", "💤 Спящие"]:
            if wallet_type in wallet_types_summary:
                count = wallet_types_summary[wallet_type]
                percentage = (count / len(features_df)) * 100
                print(f"{wallet_type}: {count} кошельков ({percentage:.1f}%)")
        
        print("\nСозданные файлы:")
        print("   • clustered_wallets.csv - все кошельки с типами")
        print("   • whales.csv, insiders.csv и т.д. - кошельки по типам")
        print("   • wallet_types_statistics.csv - статистика")
        print("   • *.png - графики и визуализации")"""
🎯 ЭТАП 3: Кластеризация кошельков с определением ВСЕХ типов
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
    """Кластеризация кошельков с определением всех типов"""
    
    def __init__(self, features_df):
        self.df = features_df.copy()
        self.scaler = StandardScaler()
        self.cluster_labels = None
        self.wallet_types = {}  # Будем хранить типы кошельков
    
    def prepare_features(self):
        """Подготовка признаков для кластеризации"""
        print("🔧 Подготавливаю признаки...")
        
        # Сначала добавим дополнительные признаки для лучшего определения типов
        self._add_special_features()
        
        # Выбираем числовые признаки для кластеризации
        numeric_cols = [
            'total_tx', 'total_volume', 'avg_tx_value',
            'tx_per_day', 'days_since_last_tx', 'whale_score',
            'incoming_tx', 'outgoing_tx', 'unique_counterparties',
            'avg_time_between_tx_hours', 'night_tx_ratio',
            'tx_variability_score'  # Новый признак
        ]
        
        # Оставляем только существующие колонки
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        
        if len(available_cols) < 2:
            print("Недостаточно признаков для кластеризации")
            return None
        
        # Создаем матрицу признаков
        X = self.df[available_cols].copy()
        
        # Заполняем пропуски
        X = X.fillna(X.median())
        
        # Масштабируем (очень важно для кластеризации!)
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Используется {len(available_cols)} признаков")
        return X_scaled, available_cols
    
    def _add_special_features(self):
        """Добавляем специальные признаки для определения типов кошельков"""
        print("➕ Добавляю специальные признаки...")
        
        # 1. Признак "активность ночью" (для инсайдеров)
        if 'night_tx_ratio' not in self.df.columns:
            # Упрощенная версия - если много транзакций при небольшом объеме
            self.df['night_tx_ratio'] = np.random.random(len(self.df)) * 0.3  # Заглушка
        
        # 2. Признак "скорость реакции" (для снайперов)
        # Предполагаем, что снайперы делают транзакции быстро друг за другом
        if 'avg_time_between_tx_hours' in self.df.columns:
            self.df['reaction_speed'] = 1 / (self.df['avg_time_between_tx_hours'] + 1)
        else:
            self.df['reaction_speed'] = 0.5
        
        # 3. Признак "разнообразие взаимодействий" (для фермеров)
        if 'unique_counterparties' in self.df.columns and 'total_tx' in self.df.columns:
            self.df['interaction_diversity'] = self.df['unique_counterparties'] / (self.df['total_tx'] + 1)
        else:
            self.df['interaction_diversity'] = 0.3
        
        # 4. Признак "вариабельность транзакций" (паттерны поведения)
        if 'avg_tx_value' in self.df.columns and 'total_volume' in self.df.columns:
            # Коэффициент вариации объема транзакций
            self.df['tx_variability_score'] = np.random.random(len(self.df))  # Заглушка
        else:
            self.df['tx_variability_score'] = 0.5
        
        # 5. Признак "временной паттерн" (когда чаще всего активен)
        self.df['time_pattern_score'] = np.random.random(len(self.df))
    
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
        # Выбираем где silhouette_score максимальный
        best_k = np.argmax(silhouette_scores) + 2
        
        print(f"Оптимальное число кластеров: {best_k}")
        
        # Визуализация выбора числа кластеров
        self._plot_cluster_selection(inertias, silhouette_scores)
        
        return best_k
    
    def _plot_cluster_selection(self, inertias, silhouette_scores):
        """Визуализация выбора оптимального числа кластеров"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Метод локтя
        ax1.plot(range(2, len(inertias) + 2), inertias, 'bo-')
        ax1.set_xlabel('Количество кластеров')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Метод локтя')
        ax1.grid(True, alpha=0.3)
        
        # Silhouette score
        ax2.plot(range(2, len(silhouette_scores) + 2), silhouette_scores, 'ro-')
        ax2.set_xlabel('Количество кластеров')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Метод силуэта')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data/cluster_selection.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def kmeans_clustering(self, n_clusters=None):
        """Кластеризация K-means"""
        print(f"Запускаю K-means кластеризацию...")
        
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
        self.kmeans_model = kmeans
        
        # Анализируем кластеры и определяем типы
        self.analyze_and_label_clusters()
        
        return labels
    
    def analyze_and_label_clusters(self):
        """Анализ кластеров и определение типов кошельков"""
        print("\n📊 АНАЛИЗ КЛАСТЕРОВ:")
        print("=" * 50)
        
        if 'cluster' not in self.df.columns:
            print("Кластеризация не выполнена")
            return
        
        # Группируем по кластерам
        cluster_stats = self.df.groupby('cluster').agg({
            'total_volume': ['count', 'mean', 'sum'],
            'total_tx': 'mean',
            'avg_tx_value': 'mean',
            'tx_per_day': 'mean',
            'days_since_last_tx': 'mean',
            'whale_score': 'mean',
            'reaction_speed': 'mean',
            'interaction_diversity': 'mean',
            'night_tx_ratio': 'mean'
        }).round(2)
        
        print(cluster_stats)
        
        # Определяем тип для каждого кластера
        cluster_names = {}
        cluster_descriptions = {}
        
        for cluster_id in self.df['cluster'].unique():
            cluster_data = self.df[self.df['cluster'] == cluster_id]
            
            # Статистика кластера
            avg_volume = cluster_data['total_volume'].mean()
            avg_tx = cluster_data['total_tx'].mean()
            avg_tx_value = cluster_data['avg_tx_value'].mean()
            days_inactive = cluster_data['days_since_last_tx'].mean()
            whale_score = cluster_data['whale_score'].mean()
            reaction_speed = cluster_data['reaction_speed'].mean() if 'reaction_speed' in cluster_data.columns else 0.5
            interaction_diversity = cluster_data['interaction_diversity'].mean() if 'interaction_diversity' in cluster_data.columns else 0.3
            night_activity = cluster_data['night_tx_ratio'].mean() if 'night_tx_ratio' in cluster_data.columns else 0.1
            
            # Определяем тип кошелька
            wallet_type, description = self._determine_wallet_type(
                avg_volume, avg_tx, avg_tx_value, days_inactive,
                whale_score, reaction_speed, interaction_diversity, night_activity
            )
            
            cluster_names[cluster_id] = wallet_type
            cluster_descriptions[cluster_id] = description
            
            # Сохраняем в словарь типов
            self.wallet_types[cluster_id] = {
                'type': wallet_type,
                'description': description,
                'count': len(cluster_data),
                'avg_volume': avg_volume,
                'avg_tx': avg_tx
            }
        
        # Добавляем названия и описания в DataFrame
        self.df['cluster_name'] = self.df['cluster'].map(cluster_names)
        self.df['cluster_description'] = self.df['cluster'].map(cluster_descriptions)
        
        print("\nОПРЕДЕЛЕННЫЕ ТИПЫ КОШЕЛЬКОВ:")
        print("=" * 50)
        for cluster_id in sorted(cluster_names.keys()):
            info = self.wallet_types[cluster_id]
            print(f"\nКластер {cluster_id}: {info['type']}")
            print(f"   Количество: {info['count']} кошельков")
            print(f"   Средний объем: {info['avg_volume']:.2f} ETH")
            print(f"   Среднее количество TX: {info['avg_tx']:.1f}")
            print(f"   Описание: {info['description']}")
    
    def _determine_wallet_type(self, avg_volume, avg_tx, avg_tx_value, days_inactive,
                              whale_score, reaction_speed, interaction_diversity, night_activity):
        """Определение типа кошелька на основе характеристик"""
        
        # 🐋 Киты - крупные игроки
        if avg_volume > 500 or avg_tx_value > 100 or whale_score >= 2.5:
            return "🐋 Киты", "Крупные игроки с большим объемом транзакций"
        
        # 🧠 Инсайдеры - покупают перед пампами
        # Характеристики: высокая ночная активность, средний объем, хороший timing
        elif night_activity > 0.4 and avg_tx_value > 10 and reaction_speed > 0.7:
            return "🧠 Инсайдеры", "Покупают перед ростом, активны в нерабочее время"
        
        # ⚡ Снайперы - ловят токены при запуске
        # Характеристики: быстрая реакция, небольшой объем, много транзакций
        elif reaction_speed > 0.8 and avg_tx > 30 and avg_volume < 50:
            return "⚡ Снайперы", "Быстро реагируют на новые возможности, много мелких транзакций"
        
        # 🧑‍🌾 Фермеры - взаимодействуют с большим числом контрактов
        # Характеристики: высокое разнообразие взаимодействий
        elif interaction_diversity > 0.6 and avg_tx > 20:
            return "🧑‍🌾 Фермеры", "Взаимодействуют со многими контрактами, активно в DeFi"
        
        # 💤 Спящие - давно неактивные кошельки
        elif days_inactive > 90 or avg_tx < 2:
            return "💤 Спящие", "Давно неактивные или малоактивные кошельки"
        
        # 📊 Обычные пользователи
        elif avg_tx > 5 and avg_volume < 20:
            return "📊 Обычные", "Стандартные пользователи, умеренная активность"
        
        # 🎯 Трейдеры
        elif avg_tx > 15 and 10 < avg_tx_value < 100:
            return "🎯 Трейдеры", "Активные трейдеры, средний объем операций"
        
        # 🔍 Исследователи
        elif interaction_diversity > 0.4 and avg_tx > 10:
            return "🔍 Исследователи", "Тестируют разные протоколы и контракты"
        
        # По умолчанию
        else:
            return "❓ Неизвестный", "Не удалось определить четкий тип"
    
    def visualize_clusters(self):
        """Визуализация кластеров"""
        if 'cluster_name' not in self.df.columns:
            print("Сначала выполните кластеризацию")
            return
        
        print("\nСоздаю визуализации...")
        
        # 1. Распределение по типам кошельков
        self._plot_wallet_types_distribution()
        
        # 2. Характеристики кластеров
        self._plot_cluster_characteristics()
        
        # 3. Топ-10 по каждому типу
        self._plot_top_wallets_by_type()
        
        # 4. Матрица характеристик
        self._plot_characteristics_matrix()
    
    def _plot_wallet_types_distribution(self):
        """Распределение кошельков по типам"""
        plt.figure(figsize=(14, 8))
        
        # Распределение по типам
        type_counts = self.df['cluster_name'].value_counts()
        
        # Сортируем по убыванию
        type_counts = type_counts.sort_values(ascending=True)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(type_counts)))
        
        plt.subplot(1, 2, 1)
        bars = plt.barh(range(len(type_counts)), type_counts.values, color=colors)
        plt.yticks(range(len(type_counts)), type_counts.index)
        plt.xlabel('Количество кошельков')
        plt.title('Распределение кошельков по типам')
        
        # Добавляем значения на столбцы
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + bar.get_width()*0.01, bar.get_y() + bar.get_height()/2,
                    f'{int(bar.get_width())}', va='center')
        
        # Круговая диаграмма
        plt.subplot(1, 2, 2)
        plt.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        plt.title('Процентное распределение')
        
        plt.tight_layout()
        plt.savefig('data/wallet_types_distribution.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def _plot_cluster_characteristics(self):
        """Визуализация характеристик кластеров"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Средний объем по типам
        volume_by_type = self.df.groupby('cluster_name')['total_volume'].mean().sort_values(ascending=False)
        axes[0, 0].bar(range(len(volume_by_type)), volume_by_type.values)
        axes[0, 0].set_xticks(range(len(volume_by_type)))
        axes[0, 0].set_xticklabels(volume_by_type.index, rotation=45, ha='right')
        axes[0, 0].set_title('Средний объем по типам')
        axes[0, 0].set_ylabel('Объем (ETH)')
        axes[0, 0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
        
        # 2. Среднее количество транзакций
        tx_by_type = self.df.groupby('cluster_name')['total_tx'].mean().sort_values(ascending=False)
        axes[0, 1].bar(range(len(tx_by_type)), tx_by_type.values, color='orange')
        axes[0, 1].set_xticks(range(len(tx_by_type)))
        axes[0, 1].set_xticklabels(tx_by_type.index, rotation=45, ha='right')
        axes[0, 1].set_title('Среднее количество транзакций')
        axes[0, 1].set_ylabel('Количество транзакций')
        
        # 3. Дни с последней транзакции
        if 'days_since_last_tx' in self.df.columns:
            inactive_by_type = self.df.groupby('cluster_name')['days_since_last_tx'].mean().sort_values(ascending=False)
            axes[1, 0].bar(range(len(inactive_by_type)), inactive_by_type.values, color='green')
            axes[1, 0].set_xticks(range(len(inactive_by_type)))
            axes[1, 0].set_xticklabels(inactive_by_type.index, rotation=45, ha='right')
            axes[1, 0].set_title('Дней с последней транзакции')
            axes[1, 0].set_ylabel('Дни')
        
        # 4. Whale score по типам
        if 'whale_score' in self.df.columns:
            whale_by_type = self.df.groupby('cluster_name')['whale_score'].mean().sort_values(ascending=False)
            axes[1, 1].bar(range(len(whale_by_type)), whale_by_type.values, color='red')
            axes[1, 1].set_xticks(range(len(whale_by_type)))
            axes[1, 1].set_xticklabels(whale_by_type.index, rotation=45, ha='right')
            axes[1, 1].set_title('Whale Score по типам')
            axes[1, 1].set_ylabel('Whale Score (0-3)')
        
        plt.tight_layout()
        plt.savefig('data/cluster_characteristics.png', dpi=100, bbox_inches='tight')
        plt.show()
    
    def _plot_top_wallets_by_type(self):
        """Топ кошельков по каждому типу"""
        wallet_types = self.df['cluster_name'].unique()
        
        for wallet_type in wallet_types:
            if wallet_type == "💤 Спящие":
                continue  # Пропускаем спящих
            
            wallets_of_type = self.df[self.df['cluster_name'] == wallet_type]
            
            if len(wallets_of_type) > 0:
                # Топ-5 по объему
                top_wallets = wallets_of_type.nlargest(5, 'total_volume')
                
                plt.figure(figsize=(12, 6))
                bars = plt.barh(range(len(top_wallets)), top_wallets['total_volume'])
                plt.yticks(range(len(top_wallets)), [f"{addr[:10]}..." for addr in top_wallets['address']])
                plt.xlabel('Объем транзакций (ETH)')
                plt.title(f'Топ-5 {wallet_type} по объему')
                plt.gca().invert_yaxis()
                
                # Добавляем значения
                for i, bar in enumerate(bars):
                    plt.text(bar.get_width() * 0.01, bar.get_y() + bar.get_height()/2,
                            f'{bar.get_width():.1f} ETH', va='center')
                
                plt.tight_layout()
                filename = f"data/top_{wallet_type.replace(' ', '_').replace('🐋', 'whale').replace('🧠', 'insider').replace('⚡', 'sniper').replace('🧑‍🌾', 'farmer')}.png"
                plt.savefig(filename, dpi=100, bbox_inches='tight')
                plt.show()
    
    def _plot_characteristics_matrix(self):
        """Матрица характеристик типов кошельков"""
        # Создаем сводную таблицу характеристик
        characteristics = ['total_volume', 'total_tx', 'days_since_last_tx', 'whale_score']
        available_chars = [c for c in characteristics if c in self.df.columns]
        
        if len(available_chars) >= 2:
            # Берем две характеристики для визуализации
            plt.figure(figsize=(10, 8))
            
            scatter = plt.scatter(self.df[available_chars[0]], 
                                 self.df[available_chars[1]],
                                 c=self.df['cluster'].astype('category').cat.codes,
                                 cmap='tab20', alpha=0.6, s=50)
            
            plt.xlabel(available_chars[0])
            plt.ylabel(available_chars[1])
            plt.title(f'Матрица характеристик: {available_chars[0]} vs {available_chars[1]}')
            plt.xscale('log')
            plt.yscale('log')
            
            # Добавляем легенду с типами
            unique_clusters = self.df['cluster_name'].unique()
            for i, cluster_type in enumerate(unique_clusters):
                plt.scatter([], [], color=plt.cm.tab20(i/len(unique_clusters)), 
                           label=cluster_type, alpha=0.6)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig('data/characteristics_matrix.png', dpi=100, bbox_inches='tight')
            plt.show()
    
    def save_results(self):
        """Сохранение результатов"""
        if not self.df.empty:
            # Сохраняем все кошельки с кластерами
            self.df.to_csv('data/clustered_wallets.csv', index=False)
            print("💾 Все кошельки сохранены: data/clustered_wallets.csv")
            
            # Сохраняем отдельно по типам
            wallet_types = self.df['cluster_name'].unique()
            
            for wallet_type in wallet_types:
                wallets_of_type = self.df[self.df['cluster_name'] == wallet_type]
                if not wallets_of_type.empty:
                    filename = wallet_type.replace(' ', '_').replace('🐋', 'whales').replace('🧠', 'insiders').replace('⚡', 'snipers').replace('🧑‍🌾', 'farmers').replace('💤', 'sleepers').replace('📊', 'regular').replace('🎯', 'traders').replace('🔍', 'explorers').replace('❓', 'unknown')
                    wallets_of_type.to_csv(f'data/{filename}.csv', index=False)
                    print(f"💾 {wallet_type}: {len(wallets_of_type)} кошельков -> data/{filename}.csv")
            
            # Сохраняем статистику по типам
            type_stats = self.df.groupby('cluster_name').agg({
                'total_volume': ['count', 'mean', 'sum', 'min', 'max'],
                'total_tx': 'mean',
                'days_since_last_tx': 'mean'
            }).round(2)
            
            type_stats.to_csv('data/wallet_types_statistics.csv')
            print("💾 Статистика по типам: data/wallet_types_statistics.csv")
            
            # Сохраняем подробную информацию о кластерах
            clusters_info = pd.DataFrame.from_dict(self.wallet_types, orient='index')
            clusters_info.to_csv('data/clusters_detailed_info.csv')
            print("💾 Детальная информация о кластерах: data/clusters_detailed_info.csv")


# 🚀 ПРИМЕР ИСПОЛЬЗОВАНИЯ
if __name__ == "__main__":
    print("=" * 60)
    print("КЛАСТЕРИЗАЦИЯ КОШЕЛЬКОВ ПО ТИПАМ")
    print("=" * 60)
    
    # Загружаем признаки из предыдущего этапа
    try:
        features_df = pd.read_csv('data/wallet_features.csv')
        print(f"Загружено {len(features_df)} кошельков с признаками")
    except FileNotFoundError:
        print("Файл wallet_features.csv не найден!")
        print("\nЧто делать:")
        print("1. Сначала запустите collect.py для сбора данных")
        print("2. Затем запустите preprocess.py для создания признаков")
        print("3. После этого запустите cluster.py")
        exit()
    except Exception as e:
        print(f"Ошибка загрузки файла: {e}")
        exit()
    
    # Проверяем что есть нужные колонки
    required_cols = ['total_tx', 'total_volume', 'days_since_last_tx']
    missing_cols = [col for col in required_cols if col not in features_df.columns]
    
    if missing_cols:
        print(f"Отсутствуют колонки: {missing_cols}")
        print("   Запустите preprocess.py для создания признаков")
        exit()
    
    # Создаем кластеризатор
    clustering = WalletClustering(features_df)
    
    # Запускаем кластеризацию
    print("\nЗапускаю кластеризацию...")
    labels = clustering.kmeans_clustering(n_clusters=8)  # 8 кластеров для лучшего разделения
    
    if labels is not None:
        print("\nКластеризация выполнена успешно!")
        
        # Визуализируем результаты
        clustering.visualize_clusters()
        
        # Сохраняем результаты
        clustering.save_results()
        
        print("\n" + "=" * 60)
        print("📊 РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ:")
        print("=" * 60)
        
        # Показываем сводку
        wallet_types_summary = features_df['cluster_name'].value_counts() if 'cluster_name' in features_df.columns else pd.Series()
        
        for wallet_type in ["🐋 Киты", "🧠 Инсайдеры", "⚡ Снайперы", "🧑‍🌾 Фермеры", "💤 Спящие"]:
            if wallet_type in wallet_types_summary:
                count = wallet_types_summary[wallet_type]
                percentage = (count / len(features_df)) * 100
                print(f"{wallet_type}: {count} кошельков ({percentage:.1f}%)")
        
        print("\nСозданные файлы:")
        print("   • clustered_wallets.csv - все кошельки с типами")
        print("   • whales.csv, insiders.csv и т.д. - кошельки по типам")
        print("   • wallet_types_statistics.csv - статистика")
        print("   • *.png - графики и визуализации")
