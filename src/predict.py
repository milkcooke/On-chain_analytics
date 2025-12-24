"""
🤖 ЭТАП 4: Предиктивная аналитика
ML модели для предсказания активности
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class PredictiveModels:
    """Модели машинного обучения"""
    
    def __init__(self):
        self.models = {}
        
    def prepare_classification_data(self, df):
        """
        Подготовка данных для классификации
        Предсказываем: будет ли крупная транзакция?
        """
        print("🔧 Подготавливаю данные для классификации...")
        
        # Создаем целевую переменную
        # Крупная транзакция = больше 10 ETH
        df = df.copy()
        
        # Для классификации нам нужно предсказать будущее
        # В реальном проекте нужно данные по времени
        # Здесь сделаем упрощенный вариант
        
        # Признаки для модели
        feature_cols = [
            'total_tx', 'total_volume', 'avg_tx_value',
            'tx_per_day', 'days_since_last_tx', 'whale_score',
            'incoming_tx', 'outgoing_tx'
        ]
        
        # Целевая переменная: является ли кит?
        df['is_whale'] = (df['whale_score'] >= 2).astype(int)
        
        # Оставляем только существующие колонки
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].fillna(0)
        y = df['is_whale']
        
        print(f"✅ Используется {len(available_features)} признаков")
        print(f"✅ Китов в данных: {y.sum()} ({y.mean()*100:.1f}%)")
        
        return X, y, available_features
    
    def train_whale_classifier(self, df):
        """Обучаем модель для определения китов"""
        print("\n🤖 Обучаю модель классификации китов...")
        
        # Подготавливаем данные
        X, y, feature_names = self.prepare_classification_data(df)
        
        # Разделяем на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"📊 Размер выборок:")
        print(f"   Обучающая: {len(X_train)} кошельков")
        print(f"   Тестовая: {len(X_test)} кошельков")
        
        # Обучаем Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'  # Важно для несбалансированных данных
        )
        
        model.fit(X_train, y_train)
        
        # Оценка модели
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        print("\n📈 РЕЗУЛЬТАТЫ МОДЕЛИ:")
        print(f"Точность: {accuracy_score(y_test, y_pred):.3f}")
        print("\nОтчет классификации:")
        print(classification_report(y_test, y_pred))
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Не кит', 'Кит'],
                   yticklabels=['Не кит', 'Кит'])
        plt.xlabel('Предсказание')
        plt.ylabel('Реальность')
        plt.title('Матрица ошибок для классификации китов')
        plt.savefig('data/confusion_matrix.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        # Важность признаков
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_importance)), feature_importance['importance'])
        plt.yticks(range(len(feature_importance)), feature_importance['feature'])
        plt.xlabel('Важность признака')
        plt.title('Важность признаков для определения китов')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('data/feature_importance.png', dpi=100, bbox_inches='tight')
        plt.show()
        
        # Сохраняем модель
        self.models['whale_classifier'] = model
        joblib.dump(model, 'models/whale_classifier.joblib')
        print("💾 Модель сохранена: models/whale_classifier.joblib")
        
        return model
    
    def predict_new_wallet(self, wallet_features):
        """Предсказание для нового кошелька"""
        if 'whale_classifier' not in self.models:
            print("❌ Модель не обучена")
            return None
        
        model = self.models['whale_classifier']
        prediction = model.predict([wallet_features])[0]
        probability = model.predict_proba([wallet_features])[0][1]
        
        return {
            'is_whale': bool(prediction),
            'probability': probability,
            'whale_score': probability * 3  # Преобразуем в нашу шкалу 0-3
        }
    
    def create_activity_forecast(self, df):
        """Прогноз активности на основе истории"""
        print("\n📊 Создаю прогноз активности...")
        
        # Для прогноза нужны временные ряды
        # В упрощенном варианте просто посчитаем статистику
        
        if 'last_tx_date' in df.columns:
            # Преобразуем даты
            df['last_tx_date'] = pd.to_datetime(df['last_tx_date'])
            
            # Кошельки с недавней активностью
            recent_active = df[df['days_since_last_tx'] < 7]
            
            print(f"📈 АКТИВНОСТЬ ЗА ПОСЛЕДНЮЮ НЕДЕЛЮ:")
            print(f"   Активных кошельков: {len(recent_active)}")
            print(f"   Из них китов: {len(recent_active[recent_active['whale_score'] >= 2])}")
            
            if len(recent_active) > 0:
                # Прогноз: если много китов активно, ждем роста
                whale_activity = len(recent_active[recent_active['whale_score'] >= 2])
                total_whales = len(df[df['whale_score'] >= 2])
                
                if total_whales > 0:
                    whale_activity_ratio = whale_activity / total_whales
                    
                    print(f"\n🎯 ПРОГНОЗ:")
                    print(f"   Активность китов: {whale_activity_ratio*100:.1f}%")
                    
                    if whale_activity_ratio > 0.3:
                        print("   🚀 ВЫСОКАЯ активность китов - возможен рост!")
                    elif whale_activity_ratio > 0.1:
                        print("   📈 УМЕРЕННАЯ активность китов")
                    else:
                        print("   📉 НИЗКАЯ активность китов")


if __name__ == "__main__":
    # Загружаем данные с кластерами
    try:
        clustered_df = pd.read_csv('data/clustered_wallets.csv')
        print(f"📁 Загружено {len(clustered_df)} кошельков с кластерами")
    except:
        print("❌ Сначала запустите cluster.py для кластеризации")
        exit()
    
    # Создаем папку для моделей
    import os
    os.makedirs('models', exist_ok=True)
    
    # Создаем и обучаем модели
    predictor = PredictiveModels()
    
    # 1. Модель определения китов
    model = predictor.train_whale_classifier(clustered_df)
    
    # 2. Прогноз активности
    predictor.create_activity_forecast(clustered_df)
    
    # 3. Пример предсказания для нового кошелька
    print("\n🧪 ПРИМЕР ПРЕДСКАЗАНИЯ:")
    example_features = [50, 200, 4, 0.5, 2, 1, 30, 20]  # Пример признаков
    prediction = predictor.predict_new_wallet(example_features)
    
    if prediction:
        print(f"   Предсказание: {'КИТ 🐋' if prediction['is_whale'] else 'Не кит'}")
        print(f"   Вероятность: {prediction['probability']*100:.1f}%")
        print(f"   Whale score: {prediction['whale_score']:.1f}/3")
    
    print("\n✅ ML модели обучены и готовы к использованию!")
