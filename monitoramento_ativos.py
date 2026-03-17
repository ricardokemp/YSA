import pandas as pd
import numpy as np
from sklearn.ensemble import Isolation Forest
import plotly.graph_objects as go

# 1. Simulação de Dados de Sensores (Vibração e Temperatura)
np.random.seed(42)
normal_data = np.random.normal(loc=20, scale=2, size=(500, 2)) # Operação Normal
anomalies = np.random.uniform(low=30, high=50, size=(20, 2))    # Picos de falha

# Criando o DataFrame
df = pd.DataFrame(np.vstack([normal_data, anomalies]), columns=['Vibracao', 'Temperatura'])

# 2. Configuração do Modelo Isolation Forest
# contamination: proporção esperada de anomalias no conjunto de dados
model = IsolationForest(contamination=0.05, random_state=42)

# Treinamento e Predição
# -1 indica anomalia, 1 indica dado normal
df['anomaly_score'] = model.fit_predict(df[['Vibracao', 'Temperatura']])

# 3. Visualização dos Resultados
fig = go.Figure()

# Dados Normais
fig.add_trace(go.Scatter(
    x=df[df['anomaly_score'] == 1]['Vibracao'],
    y=df[df['anomaly_score'] == 1]['Temperatura'],
    mode='markers', name='Operação Normal', marker=dict(color='blue')
))

# Anomalias Detectadas
fig.add_trace(go.Scatter(
    x=df[df['anomaly_score'] == -1]['Vibracao'],
    y=df[df['anomaly_score'] == -1]['Temperatura'],
    mode='markers', name='Anomalia Detectada', marker=dict(color='red', symbol='x')
))

fig.update_layout(title='Detecção de Anomalias em Ativos Industriais',
                  xaxis_title='Nível de Vibração', yaxis_title='Temperatura (°C)')
fig.show()