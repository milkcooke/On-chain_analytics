"""
📊 ЭТАП 6: Дашборд и визуализация
Интерактивная панель для анализа результатов
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Настройки страницы
st.set_page_config(
    page_title="On-Chain Analytics",
    page_icon="📊",
    layout="wide"
)

class AnalyticsDashboard:
    """Дашборд для анализа on-chain данных"""
    
    def __init__(self):
        self.load_data()
    
    def load_data(self):
        """Загрузка всех данных"""
        self.transactions = pd.DataFrame()
        self.features = pd.DataFrame()
        self.clusters = pd.DataFrame()
        self.whales = pd.DataFrame()
        
        # Пробуем загрузить данные
        data_files = {
            'transactions': 'data/all_transactions.csv',
            'features': 'data/wallet_features.csv',
            'clusters': 'data/clustered_wallets.csv',
            'whales': 'data/whales.csv'
        }
        
        for name, filepath in data_files.items():
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    if name == 'transactions' and 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                    if name == 'features' and 'first_tx_date' in df.columns:
                        df['first_tx_date'] = pd.to_datetime(df['first_tx_date'])
                    if name == 'clusters' and 'first_tx_date' in df.columns:
                        df['first_tx_date'] = pd.to_datetime(df['first_tx_date'])
                    
                    setattr(self, name, df)
                    st.sidebar.success(f"✅ {name} загружен")
                except:
                    st.sidebar.warning(f"⚠️  {name} не загружен")
    
    def show_header(self):
        """Заголовок дашборда"""
        st.title("📊 On-Chain Analytics Dashboard")
        st.markdown("""
        Анализ блокчейн-транзакций и кластеризация кошельков
        """)
        st.markdown("---")
    
    def show_overview(self):
        """Общая статистика"""
        st.header("📈 Общая статистика")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if not self.transactions.empty:
                total_tx = len(self.transactions)
                st.metric("Всего транзакций", f"{total_tx:,}")
        
        with col2:
            if not self.transactions.empty:
                total_volume = self.transactions['value_eth'].sum()
                st.metric("Общий объем", f"{total_volume:,.1f} ETH")
        
        with col3:
            if not self.clusters.empty:
                unique_wallets = self.clusters['address'].nunique()
                st.metric("Уникальных кошельков", f"{unique_wallets:,}")
        
        with col4:
            if not self.whales.empty:
                whale_count = len(self.whales)
                st.metric("Обнаружено китов", f"{whale_count}")
        
        # График объема по дням
        if not self.transactions.empty and 'date' in self.transactions.columns:
            st.subheader("📅 Объем транзакций по дням")
            
            daily_volume = self.transactions.groupby('date')['value_eth'].sum().reset_index()
            daily_volume['date'] = pd.to_datetime(daily_volume['date'])
            
            fig = px.line(daily_volume, x='date', y='value_eth',
                         title="Динамика объема транзакций")
            fig.update_xaxes(title="Дата")
            fig.update_yaxes(title="Объем (ETH)")
            st.plotly_chart(fig, use_container_width=True)
    
    def show_clustering_results(self):
        """Результаты кластеризации"""
        st.header("🎯 Результаты кластеризации")
        
        if self.clusters.empty:
            st.warning("Сначала выполните кластеризацию (запустите cluster.py)")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Распределение по кластерам
            if 'cluster_name' in self.clusters.columns:
                cluster_dist = self.clusters['cluster_name'].value_counts().reset_index()
                cluster_dist.columns = ['Кластер', 'Количество']
                
                fig = px.pie(cluster_dist, values='Количество', names='Кластер',
                            title="Распределение кошельков по кластерам")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Средний объем по кластерам
            if 'cluster_name' in self.clusters.columns and 'total_volume' in self.clusters.columns:
                cluster_stats = self.clusters.groupby('cluster_name')['total_volume'].mean().reset_index()
                cluster_stats = cluster_stats.sort_values('total_volume', ascending=False)
                
                fig = px.bar(cluster_stats, x='cluster_name', y='total_volume',
                            title="Средний объем по кластерам")
                fig.update_xaxes(title="Кластер")
                fig.update_yaxes(title="Средний объем (ETH)")
                st.plotly_chart(fig, use_container_width=True)
        
        # Диаграмма рассеяния
        st.subheader("📊 Диаграмма рассеяния: Объем vs Количество транзакций")
        
        if all(col in self.clusters.columns for col in ['total_volume', 'total_tx', 'cluster_name']):
            fig = px.scatter(self.clusters, x='total_tx', y='total_volume',
                            color='cluster_name', size='whale_score',
                            hover_data=['address'],
                            title="Кластеризация кошельков",
                            log_x=True, log_y=True)
            fig.update_xaxes(title="Количество транзакций (лог)")
            fig.update_yaxes(title="Общий объем (ETH, лог)")
            st.plotly_chart(fig, use_container_width=True)
    
    def show_whales_analysis(self):
        """Анализ китов"""
        st.header("🐋 Анализ китов")
        
        if self.whales.empty:
            st.warning("Киты не обнаружены")
            return
        
        # Топ китов
        st.subheader("Топ-10 китов по объему")
        
        top_whales = self.whales.nlargest(10, 'total_volume')[['address', 'total_volume', 'total_tx', 'tx_per_day']]
        top_whales['address_short'] = top_whales['address'].apply(lambda x: x[:12] + '...')
        
        fig = go.Figure(data=[
            go.Bar(name='Объем', x=top_whales['address_short'], y=top_whales['total_volume']),
            go.Bar(name='Транзакции', x=top_whales['address_short'], y=top_whales['total_tx'])
        ])
        
        fig.update_layout(barmode='group', title="Топ-10 китов")
        st.plotly_chart(fig, use_container_width=True)
        
        # Таблица с деталями
        st.subheader("Детальная информация о китах")
        
        display_cols = ['address', 'total_volume', 'total_tx', 'tx_per_day', 
                       'avg_tx_value', 'days_since_last_tx']
        available_cols = [col for col in display_cols if col in self.whales.columns]
        
        if available_cols:
            whales_display = self.whales[available_cols].copy()
            whales_display['address'] = whales_display['address'].apply(lambda x: x[:20] + '...')
            st.dataframe(whales_display.style.format({
                'total_volume': '{:,.2f}',
                'avg_tx_value': '{:,.2f}',
                'tx_per_day': '{:.3f}'
            }), use_container_width=True)
    
    def show_transaction_analysis(self):
        """Анализ транзакций"""
        st.header("💸 Анализ транзакций")
        
        if self.transactions.empty:
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Распределение по размеру транзакций
            st.subheader("Распределение по размеру")
            
            fig = px.histogram(self.transactions, x='value_eth',
                              nbins=50, title="Размер транзакций",
                              log_x=True)
            fig.update_xaxes(title="Размер (ETH, лог)")
            fig.update_yaxes(title="Количество")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Топ транзакций
            st.subheader("Самые крупные транзакции")
            
            top_tx = self.transactions.nlargest(10, 'value_eth')[['timestamp', 'from', 'to', 'value_eth']]
            top_tx['timestamp'] = pd.to_datetime(top_tx['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            top_tx['from'] = top_tx['from'].apply(lambda x: x[:12] + '...')
            top_tx['to'] = top_tx['to'].apply(lambda x: x[:12] + '...')
            
            st.dataframe(top_tx.style.format({'value_eth': '{:,.2f}'}), 
                        use_container_width=True)
    
    def show_wallet_search(self):
        """Поиск по кошелькам"""
        st.header("🔍 Поиск кошелька")
        
        if self.clusters.empty:
            return
        
        # Поле поиска
        search_address = st.text_input("Введите адрес кошелька (начинается с 0x):", 
                                      placeholder="0x...")
        
        if search_address:
            # Ищем кошелек
            wallet_data = self.clusters[self.clusters['address'].str.contains(search_address, case=False)]
            
            if not wallet_data.empty:
                st.success("✅ Кошелек найден!")
                
                # Показываем информацию
                cols = st.columns(3)
                
                with cols[0]:
                    st.metric("Кластер", wallet_data.iloc[0].get('cluster_name', 'Неизвестно'))
                
                with cols[1]:
                    st.metric("Общий объем", f"{wallet_data.iloc[0].get('total_volume', 0):,.2f} ETH")
                
                with cols[2]:
                    st.metric("Количество TX", f"{wallet_data.iloc[0].get('total_tx', 0):,}")
                
                # Подробная информация
                st.subheader("Подробная информация:")
                
                info_cols = ['address', 'total_tx', 'total_volume', 'avg_tx_value',
                            'tx_per_day', 'days_since_last_tx', 'whale_score']
                
                available_info = {col: wallet_data.iloc[0].get(col, 'N/A') 
                                 for col in info_cols if col in wallet_data.columns}
                
                for key, value in available_info.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.warning("Кошелек не найден в данных")
    
    def run(self):
        """Запуск дашборда"""
        self.show_header()
        
        # Сайдбар с навигацией
        st.sidebar.title("Навигация")
        page = st.sidebar.radio(
            "Выберите раздел:",
            ["📊 Обзор", "🎯 Кластеризация", "🐋 Киты", "💸 Транзакции", "🔍 Поиск"]
        )
        
        # Загрузка данных
        with st.sidebar.expander("📁 Загруженные данные"):
            data_status = {
                "Транзакции": not self.transactions.empty,
                "Признаки": not self.features.empty,
                "Кластеры": not self.clusters.empty,
                "Киты": not self.whales.empty
            }
            
            for name, loaded in data_status.items():
                if loaded:
                    st.success(f"✅ {name}")
                else:
                    st.error(f"❌ {name}")
        
        # Инструкция
        with st.sidebar.expander("ℹ️ Инструкция"):
            st.markdown("""
            1. Запустите `collect.py` - сбор данных
            2. Запустите `preprocess.py` - обработка
            3. Запустите `cluster.py` - кластеризация
            4. Обновите страницу
            
            Все файлы сохраняются в папку `data/`
            """)
        
        # Показываем выбранную страницу
        if page == "📊 Обзор":
            self.show_overview()
        elif page == "🎯 Кластеризация":
            self.show_clustering_results()
        elif page == "🐋 Киты":
            self.show_whales_analysis()
        elif page == "💸 Транзакции":
            self.show_transaction_analysis()
        elif page == "🔍 Поиск":
            self.show_wallet_search()


if __name__ == "__main__":
    # Проверяем установлен ли streamlit
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit не установлен")
        print("   Установите: pip install streamlit")
        print("   Запустите: streamlit run src/dashboard.py")
        exit()
    
    print("🚀 Запускаю дашборд...")
    print("   Откройте http://localhost:8501 в браузере")
    
    # В реальном коде здесь будет запуск Streamlit
    # Но в файле мы просто создаем класс
    
    dashboard = AnalyticsDashboard()
    
    # Для запуска через Streamlit нужно:
    # 1. Сохранить этот файл
    # 2. В терминале запустить: streamlit run src/dashboard.py
    
    # Здесь мы просто показываем что будет в дашборде
    print("\n📊 ДАШБОРД БУДЕТ СОДЕРЖАТЬ:")
    print("1. 📊 Общая статистика")
    print("2. 🎯 Результаты кластеризации")
    print("3. 🐋 Анализ китов")
    print("4. 💸 Анализ транзакций")
    print("5. 🔍 Поиск по кошелькам")
    
    print("\n💡 Для запуска дашборда выполните:")
    print("   streamlit run src/dashboard.py")
