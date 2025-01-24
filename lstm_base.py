import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from math import sqrt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Chargement des données
data = pd.read_csv("Traffic.csv")

# Encodage des variables catégorielles avec un ordre spécifié
order_traffic = ['low', 'normal', 'high', 'heavy']
order_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

encoder_traffic = OrdinalEncoder(categories=[order_traffic])
encoder_days = OrdinalEncoder(categories=[order_days])

data["Traffic_Situation"] = encoder_traffic.fit_transform(data[["Traffic Situation"]])
data["Day of the week"] = encoder_days.fit_transform(data[["Day of the week"]])

data['Time'] = pd.to_datetime(data['Time']).dt.hour * 60 + pd.to_datetime(data['Time']).dt.minute
data['Time'] = data['Time'] / 60.0

# Suppression des colonnes inutilisées
data.drop(["Day of the week", "Traffic Situation"], inplace=True, axis=1)

# Détection des outliers
def detect_outliers(df, feature):
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]

outliers = {}
for feature in ['CarCount', 'BikeCount', 'BusCount', 'TruckCount']:
    outliers[feature] = detect_outliers(data, feature)
    print(f'Outliers for {feature}: {len(outliers[feature])}')

# Suppression des outliers
for feature in outliers:
    data = data[~data.index.isin(outliers[feature].index)]

X, y = data.iloc[:, 1:7], data.iloc[:, -1]

# Normalisation des données
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=1)

# Définition du modèle LSTM de base
model_base = Sequential()
model_base.add(LSTM(50, input_shape=(1, X_reshaped.shape[2])))
model_base.add(Dense(1))

model_base.compile(optimizer='adam', loss='mean_squared_error')
model_base.summary()

# Entraînement du modèle
history_base = model_base.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Prédiction sur les données de test
predictions_base = model_base.predict(X_test)
rmse_base = sqrt(mean_squared_error(y_test, predictions_base))

# Sauvegarde du RMSE et des prédictions
joblib.dump(rmse_base, 'rmse_base.pkl')
np.save('predictions_base.npy', predictions_base)

print(f'RMSE of the Basic Model: {rmse_base}')

# Visualisation avec Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='Real values'))
fig.add_trace(go.Scatter(x=list(range(len(predictions_base))), y=predictions_base.flatten(), mode='lines', name='Predicted Values - Initial basic LSTM'))
fig.update_layout(title='Comparison of Real and Predicted Values (Basic Model)', xaxis_title='Samples', yaxis_title=' Traffic Situation')
fig.write_html("base_comparison_real_pred.html")
fig.show()

# Comparaison des valeurs prédites et des valeurs réelles
comparison_df = pd.DataFrame({'real values': y_test.values.flatten(), 'Predicted Values': predictions_base.flatten()})
comparison_sample = comparison_df.sample(10)  # Prendre un échantillon de 10 valeurs pour la comparaison
print(comparison_sample)

# Calcul des métriques d'évaluation
mae = mean_absolute_error(y_test, predictions_base)
mse = mean_squared_error(y_test, predictions_base)
rmse = sqrt(mse)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

# Visualisation des métriques d'évaluation
fig_metrics = px.bar(x=['MAE', 'MSE', 'RMSE'], y=[mae, mse, rmse], title='Evaluation Metrics', labels={'x': 'Metric', 'y': 'Value'})
fig_metrics.write_html("base_metrics_evaluation.html")
fig_metrics.show()

# Tracer les valeurs réelles et prédites sur une échelle de temps
fig_time = go.Figure()
fig_time.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='original data'))
fig_time.add_trace(go.Scatter(x=y_test.index, y=predictions_base.flatten(), mode='lines', name='predict data', line=dict(dash='dash')))
fig_time.update_layout(title='Comparaison original data and predict data', xaxis_title='time', yaxis_title='Trafic situation')
fig_time.write_html("base_comparison_time.html")
fig_time.show()

# Sauvegarde des valeurs réelles
np.save('y_test.npy', y_test)