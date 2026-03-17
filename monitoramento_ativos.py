import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

# Bloco de segurança para garantir a importação do Scikit-Learn
try:
    from sklearn.ensemble import IsolationForest
except ModuleNotFoundError:
    st.error("A biblioteca 'scikit-learn' não foi encontrada. Certifique-se de que o arquivo requirements.txt existe ou execute 'pip install scikit-learn' no terminal.")
    st.stop()

# Configuração da página para o padrão de dashboard industrial
st.set_page_config(page_title="Yokogawa - Monitoramento IA", layout="wide")

st.title("🛡️ Monitoramento de Ativos com IA")
st.markdown("Interface para detecção de anomalias em tempo real, focada em ativos da América do Sul.")

# Barra lateral para ajustes do engenheiro
st.sidebar.header("Parâmetros Técnicos")
contaminacao = st.sidebar.slider("Sensibilidade da IA", 0.01, 0.20, 0.05)
ruido = st.sidebar.slider("Nível de Ruído dos Sensores", 1.0, 10.0, 2.5)

# Função para simular telemetria (Vibração e Temperatura)
def capturar_dados_sensores():
    # Gerando dados de operação estável
    estavel = np.random.normal(loc=[22, 48], scale=ruido, size=(100, 2))
    # Gerando picos aleatórios que representam falhas
    falhas = np.random.uniform(low=[40, 70], high=[60, 90], size=(5, 2))
    
    dados = np.vstack([estavel, falhas])
    return pd.DataFrame(dados, columns=['Vibracao', 'Temperatura'])

# Espaço reservado para atualização do dashboard
container_principal = st.empty()

# Loop de monitoramento (simulando 10 ciclos de leitura)
for ciclo in range(10):
    df_atual = capturar_dados_sensores()
    
    # Aplicação do algoritmo de IA (Isolation Forest corrigido)
    modelo = IsolationForest(contamination=contaminacao, random_state=42)
    df_atual['resultado'] = modelo.fit_predict(df_atual[['Vibracao', 'Temperatura']])
    
    # Tradução dos resultados para o operador
    df_atual['Diagnóstico'] = df_atual['resultado'].map({1: 'Normal', -1: 'Anomalia'})

    with container_principal.container():
        col_grafico, col_metricas = st.columns([2, 1])

        with col_grafico:
            fig = go.Figure()
            
            # Plotando pontos normais (Azul Yokogawa)
            df_normal = df_atual[df_atual['resultado'] == 1]
            fig.add_trace(go.Scatter(
                x=df_normal['Vibracao'], y=df_normal['Temperatura'],
                mode='markers', name='Operação Normal',
                marker=dict(color='#005596', size=7, opacity=0.6)
            ))

            # Plotando anomalias (Vermelho de Alerta)
            df_falha = df_atual[df_atual['resultado'] == -1]
            fig.add_trace(go.Scatter(
                x=df_falha['Vibracao'], y=df_falha['Temperatura'],
                mode='markers', name='ALERTA CRÍTICO',
                marker=dict(color='#D32F2F', size=12, symbol='diamond')
            ))

            fig.update_layout(
                title=f"Mapa de Estados do Ativo - Ciclo {ciclo + 1}",
                xaxis_title="Vibração (mm/s)",
                yaxis_title="Temperatura (°C)",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_metricas:
            qtd_anomalias = len(df_falha)
            st.metric("Desvios Detectados", qtd_anomalias, delta_color="inverse")
            
            if qtd_anomalias > 0:
                st.warning(f"O sistema identificou {qtd_anomalias} pontos fora do padrão.")
            else:
                st.success("Equipamento operando dentro da normalidade.")

            st.write("Tabela de Dados Recentes:")
            st.dataframe(df_atual.tail(5), use_container_width=True)

    time.sleep(1.5)
