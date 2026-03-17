import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import time

# Configuração da página para aproveitar o espaço lateral
st.set_page_config(page_title="Yokogawa - IA de Ativos", layout="wide")

st.title("🛡️ Monitoramento de Ativos Industriais com IA")
st.markdown("""
Esta aplicação utiliza o algoritmo **Isolation Forest** para detectar desvios operacionais 
em sensores de vibração e temperatura, auxiliando na manutenção preditiva.
""")

# Barra lateral para ajustes técnicos
st.sidebar.header("Configurações do Modelo")
contamination = st.sidebar.slider("Sensibilidade (Contaminação)", 0.01, 0.20, 0.05)
sensor_noise = st.sidebar.slider("Ruído do Sensor", 0.5, 5.0, 2.0)

# Função para simular a leitura de sensores (OT Data)
def get_sensor_data():
    # Gerando dados normais (Cluster central)
    normal = np.random.normal(loc=[20, 45], scale=sensor_noise, size=(100, 2))
    # Gerando anomalias propositais (Outliers)
    anomalies = np.random.uniform(low=[35, 60], high=[50, 80], size=(5, 2))
    
    data = np.vstack([normal, anomalies])
    df = pd.DataFrame(data, columns=['Vibracao', 'Temperatura'])
    return df

# Container que será atualizado no loop
placeholder = st.empty()

# Simulação de monitoramento contínuo
for _ in range(5): 
    df_atual = get_sensor_data()
    
    # Instanciando e treinando o modelo corrigido
    model = IsolationForest(contamination=contamination, random_state=42)
    df_atual['status_code'] = model.fit_predict(df_atual[['Vibracao', 'Temperatura']])
    
    # Mapeando os resultados para linguagem humanizada
    df_atual['Status'] = df_atual['status_code'].map({1: 'Operação Normal', -1: 'Anomalia'})

    with placeholder.container():
        col1, col2 = st.columns([2, 1])

        with col1:
            # Gráfico de Dispersão com Plotly
            fig = go.Figure()
            
            # Plot dos dados normais
            normais = df_atual[df_atual['status_code'] == 1]
            fig.add_trace(go.Scatter(
                x=normais['Vibracao'], y=normais['Temperatura'],
                mode='markers', name='Normal', 
                marker=dict(color='#005596', size=8, opacity=0.7)
            ))

            # Plot das anomalias detectadas
            anomalias = df_atual[df_atual['status_code'] == -1]
            fig.add_trace(go.Scatter(
                x=anomalias['Vibracao'], y=anomalias['Temperatura'],
                mode='markers', name='Anomalia', 
                marker=dict(color='#FF0000', size=12, symbol='x')
            ))

            fig.update_layout(
                title="Espaço de Estados do Ativo",
                xaxis_title="Vibração (mm/s)",
                yaxis_title="Temperatura (°C)",
                template="plotly_white",
                legend_title="Diagnóstico"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Painel de métricas
            total_anomalies = len(anomalias)
            st.metric("Alertas de Anomalia", total_anomalies)
            
            if total_anomalies > 0:
                st.error(f"Sistema detectou {total_anomalies} desvios críticos.")
            else:
                st.success("Estabilidade operacional confirmada.")

            st.write("Dados Recentes (Sensores):")
            st.dataframe(df_atual.tail(8), use_container_width=True)

    time.sleep(1) # Intervalo entre "leituras"
