import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from sklearn.ensemble import IsolationForest

# Configurações de interface
st.set_page_config(page_title="Yokogawa IA - Ativos", layout="wide")

st.title("🛡️ Diagnóstico de Ativos Industriais")
st.write("Monitoramento baseado em normas técnicas para vibração (mm/s) e temperatura (°C).")

# Parâmetros Técnicos na barra lateral
st.sidebar.header("⚙️ Configuração do Modelo")
sensibilidade = st.sidebar.slider("Sensibilidade da IA (Contaminação)", 0.01, 0.15, 0.05)

st.sidebar.subheader("📍 Limites de Referência")
st.sidebar.info("""
**ISO 10816 (Vibração):**
* Normal: < 2.8 mm/s
* Alerta: > 4.5 mm/s
* Crítico: > 7.1 mm/s
""")

# Função para gerar dados simulando um motor real
def gerar_dados_reais():
    # Operação estável: 1.5 a 2.5 mm/s e 40 a 50°C
    base_normal = np.random.normal(loc=[2.0, 45.0], scale=[0.4, 2.0], size=(100, 2))
    
    # Simulação de anomalia: Aumento de vibração por desalinhamento ou aquecimento
    base_falha = np.random.uniform(low=[5.0, 65.0], high=[9.0, 85.0], size=(5, 2))
    
    uniao = np.vstack([base_normal, base_falha])
    return pd.DataFrame(uniao, columns=['Vibração (mm/s)', 'Temperatura (°C)'])

# Espaço de exibição dinâmico
painel = st.empty()

# Loop de monitoramento
for i in range(20):
    df = gerar_dados_reais()
    
    # IA: Treinamento e Predição
    ia_modelo = IsolationForest(contamination=sensibilidade, random_state=42)
    df['previsao'] = ia_modelo.fit_predict(df[['Vibração (mm/s)', 'Temperatura (°C)']])
    
    with painel.container():
        m1, m2, m3 = st.columns(3)
        
        # Última leitura capturada
        ultima_vibracao = df['Vibração (mm/s)'].iloc[-1]
        ultima_temp = df['Temperatura (°C)'].iloc[-1]
        anomalias_total = len(df[df['previsao'] == -1])

        m1.metric("Vibração Atual", f"{ultima_vibracao:.2f} mm/s")
        m2.metric("Temperatura Atual", f"{ultima_temp:.1f} °C")
        m3.metric("Pontos Fora de Padrão", anomalias_total)

        st.divider()

        col_esq, col_dir = st.columns([2, 1])

        with col_esq:
            fig = go.Figure()
            
            # Dados estáveis
            df_n = df[df['previsao'] == 1]
            fig.add_trace(go.Scatter(x=df_n['Vibração (mm/s)'], y=df_n['Temperatura (°C)'], 
                                     mode='markers', name='Estável',
                                     marker=dict(color='#005596', opacity=0.6, size=8)))
            
            # Anomalias (Alertas)
            df_a = df[df['previsao'] == -1]
            fig.add_trace(go.Scatter(x=df_a['Vibração (mm/s)'], y=df_a['Temperatura (°C)'], 
                                     mode='markers', name='ANOMALIA',
                                     marker=dict(color='#D32F2F', size=12, symbol='diamond')))

            fig.update_layout(title="Correlação Vibração x Temperatura", 
                              xaxis_title="Vibração (mm/s)", yaxis_title="Temperatura (°C)",
                              template="plotly_white", height=450)
            st.plotly_chart(fig, use_container_width=True)

        with col_dir:
            if anomalias_total > 0:
                st.error(f"🚨 Atenção: Detectados {anomalias_total} desvios de performance.")
                st.write("Verificar possível desalinhamento ou lubrificação.")
            else:
                st.success("✅ Ativo operando em conformidade técnica.")

            st.write("Histórico de Sinais:")
            st.dataframe(df.tail(8), use_container_width=True)

    time.sleep(2)
