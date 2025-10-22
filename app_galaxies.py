# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io

# Configuração da página
st.set_page_config(
    page_title="Segmentação de Clientes - Galaxies",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado com Bootstrap
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        border-bottom: 2px solid #2e86ab;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .cluster-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #2e86ab;
    }
    .metric-card {
        background-color: #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🚀 Segmentação de Clientes - Galaxies</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configurações")
uploaded_file = st.sidebar.file_uploader("Carregar arquivo CSV", type="csv")

# Parâmetros ajustáveis
st.sidebar.markdown("### Parâmetros do Modelo")
n_clusters = st.sidebar.slider("Número de Clusters", min_value=2, max_value=8, value=4)
pca_variance = st.sidebar.slider("Variância mínima do PCA", min_value=0.7, max_value=0.95, value=0.85)

def load_data(uploaded_file):
    """Carrega os dados do arquivo CSV"""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        # Carregar dados de exemplo ou mostrar mensagem
        st.info("📊 Por favor, carregue um arquivo CSV na sidebar para começar a análise.")
        return None

def preprocess_data(df):
    """Pré-processamento dos dados"""
    
    # Limpeza de dados categóricos
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        df[col] = (df[col]
                     .astype(str)
                     .str.lower()
                     .str.strip()
                  )
    
    # Cálculo da idade
    hoje = datetime.now()
    df['idade'] = hoje.year - pd.to_datetime(df['data_nascimento'], format='%d/%m/%Y').dt.year
    
    # Faixas etárias
    df['faixa_etaria'] = pd.qcut(df['idade'], q=4)
    df['faixa_etaria'] = df['faixa_etaria'].astype(str)
    
    # Frequência de compra ordinal
    freq_map = {'semanal':0, 'quinzenal':1, 'mensal':2, 'bimestral':3, 'trimestral':4}
    df['frequencia_compra_ord'] = df['frequencia_compra'].map(freq_map)
    
    # Frequency encoding para influenciador
    df['influenciador'] = df['influenciador'].map(df['influenciador'].value_counts(normalize=True))
    df = df.drop('influenciador', axis=1)
    
    return df

def prepare_features(df):
    """Prepara as features para o modelo"""
    categorical_cols = ['canal_preferido','categoria_favorita','regiao','pagamento','genero','faixa_etaria']
    numeric_cols = ['ticket_medio','qtd_itens','idade','frequencia_compra_ord']
    
    df_ready = pd.get_dummies(df[categorical_cols + numeric_cols], 
                             columns=categorical_cols, drop_first=False)
    
    return df_ready.fillna(0)

def plot_distributions(df):
    """Plota distribuições das variáveis principais"""
    st.markdown('<h2 class="section-header">📊 Análise Exploratória</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuição de frequência de compra
        fig, ax = plt.subplots(figsize=(8, 5))
        df['frequencia_compra'].value_counts().plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title('Distribuição de Frequência de Compra')
        ax.set_xlabel('Frequência')
        ax.set_ylabel('Contagem')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Boxplot do ticket médio
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=df['ticket_medio'], ax=ax, color='lightgreen')
        ax.set_title('Boxplot do Ticket Médio')
        st.pyplot(fig)
    
    with col2:
        # Distribuição de idade
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df['idade'], bins=15, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.set_title('Distribuição de Idade dos Clientes')
        ax.set_xlabel('Idade (anos)')
        ax.set_ylabel('Contagem')
        st.pyplot(fig)
        
        # Contagem por gênero
        fig, ax = plt.subplots(figsize=(8, 5))
        df['genero'].value_counts().plot(kind='bar', ax=ax, color='lightblue')
        ax.set_title('Contagem por Gênero')
        ax.set_xlabel('Gênero')
        ax.set_ylabel('Contagem')
        st.pyplot(fig)

def perform_clustering(df_ready, n_clusters, pca_variance):
    """Executa a clusterização"""
    
    # Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_ready)
    
    # PCA
    pca = PCA(n_components=pca_variance, svd_solver='full', random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Métricas para escolha do número de clusters
    sil_scores = []
    inertias = []
    K_range = range(2, 9)
    
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        sil_scores.append(silhouette_score(X_pca, labels))
        inertias.append(km.inertia_)
    
    # Clusterização final
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    
    return X_pca, labels, sil_scores, inertias, K_range, pca

def plot_clustering_results(X_pca, labels, sil_scores, inertias, K_range, n_clusters):
    """Plota os resultados da clusterização"""
    
    st.markdown('<h2 class="section-header">🔍 Resultados da Clusterização</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot Silhouette Score
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(K_range, sil_scores, marker='o', linewidth=2, markersize=8)
        ax.axvline(x=n_clusters, color='red', linestyle='--', alpha=0.7)
        ax.set_title("Silhouette Score por Número de Clusters")
        ax.set_xlabel("Número de Clusters (K)")
        ax.set_ylabel("Silhouette Score")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Plot Método do Cotovelo
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(K_range, inertias, marker='o', linewidth=2, markersize=8)
        ax.axvline(x=n_clusters, color='red', linestyle='--', alpha=0.7)
        ax.set_title("Método do Cotovelo (Inércia)")
        ax.set_xlabel("Número de Clusters (K)")
        ax.set_ylabel("Inércia")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Plot dos clusters
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7, s=50)
    ax.set_title(f'Visualização dos Clusters (K={n_clusters})')
    ax.set_xlabel('Componente Principal 1')
    ax.set_ylabel('Componente Principal 2')
    plt.colorbar(scatter, ax=ax, label='Cluster')
    st.pyplot(fig)

def create_cluster_profiles(df, labels):
    """Cria perfis detalhados para cada cluster"""
    
    st.markdown('<h2 class="section-header">👥 Perfis dos Clusters</h2>', unsafe_allow_html=True)
    
    df_used = df.copy()
    df_used['cluster'] = labels
    
    cat_cols_profile = ['canal_preferido','categoria_favorita','regiao','pagamento','genero','faixa_etaria']
    num_cols_profile = ['idade','ticket_medio','qtd_itens']
    
    # Estatísticas gerais
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Clientes", len(df_used))
    with col2:
        st.metric("Número de Clusters", len(np.unique(labels)))
    with col3:
        st.metric("Silhouette Score", f"{silhouette_score(df_used[['idade', 'ticket_medio', 'qtd_itens']], labels):.3f}")
    
    # Perfil de cada cluster
    for c_id in sorted(df_used['cluster'].unique()):
        cluster_data = df_used[df_used['cluster'] == c_id]
        
        with st.expander(f"📊 Cluster {c_id} (Tamanho: {len(cluster_data)} clientes - {len(cluster_data)/len(df_used)*100:.1f}%)", expanded=True):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 Estatísticas Numéricas:**")
                for col in num_cols_profile:
                    if col in cluster_data.columns:
                        avg_val = cluster_data[col].mean()
                        st.write(f"• {col}: {avg_val:.2f}")
            
            with col2:
                st.markdown("**📋 Características Principais:**")
                for var in cat_cols_profile:
                    if var in cluster_data.columns:
                        moda = cluster_data[var].mode()
                        if not moda.empty:
                            val = moda.iloc[0]
                            pct = (cluster_data[var].eq(val).mean()*100)
                            st.write(f"• {var}: **{val}** ({pct:.1f}%)")
            
            # Gráfico de barras para características categóricas
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.ravel()
            
            for i, var in enumerate(cat_cols_profile[:6]):
                if var in cluster_data.columns:
                    cluster_data[var].value_counts().head(5).plot(kind='bar', ax=axes[i], color='lightblue')
                    axes[i].set_title(f'{var} - Cluster {c_id}')
                    axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig)

# Main application logic
def main():
    if uploaded_file is not None:
        # Carregar dados
        with st.spinner('Carregando dados...'):
            df = load_data(uploaded_file)
        
        if df is not None:
            # Pré-processamento
            with st.spinner('Pré-processando dados...'):
                df_processed = preprocess_data(df)
            
            # Análise exploratória
            plot_distributions(df_processed)
            
            # Preparação de features
            with st.spinner('Preparando features para o modelo...'):
                df_ready = prepare_features(df_processed)
            
            # Clusterização
            with st.spinner('Executando clusterização...'):
                X_pca, labels, sil_scores, inertias, K_range, pca = perform_clustering(
                    df_ready, n_clusters, pca_variance
                )
            
            # Resultados
            plot_clustering_results(X_pca, labels, sil_scores, inertias, K_range, n_clusters)
            
            # Perfis dos clusters
            create_cluster_profiles(df_processed, labels)
            
            # Download dos resultados
            st.markdown('<h2 class="section-header">💾 Exportar Resultados</h2>', unsafe_allow_html=True)
            
            df_result = df_processed.copy()
            df_result['cluster'] = labels
            
            # Converter para CSV
            csv = df_result.to_csv(index=False)
            st.download_button(
                label="📥 Download dos dados com clusters",
                data=csv,
                file_name="clientes_clusterizados.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()