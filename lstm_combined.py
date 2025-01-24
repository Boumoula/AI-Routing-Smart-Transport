import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import joblib
import xgboost as xgb  # Importing XGBoost


# Load the data
data = pd.read_csv("Traffic.csv")

# Encode categorical variables with a specified order
order_traffic = ['low', 'normal', 'high', 'heavy']
order_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

encoder_traffic = OrdinalEncoder(categories=[order_traffic])
encoder_days = OrdinalEncoder(categories=[order_days])

data["Traffic_Situation"] = encoder_traffic.fit_transform(data[["Traffic Situation"]])
data["Day of the week"] = encoder_days.fit_transform(data[["Day of the week"]])

data['Time'] = pd.to_datetime(data['Time']).dt.hour * 60 + pd.to_datetime(data['Time']).dt.minute
data['Time'] = data['Time'] / 60.0

data['Date'] = pd.to_datetime(data['Date']).map(pd.Timestamp.timestamp)  # Convert the date to timestamp

# Outlier detection
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

# Remove outliers
for feature in outliers:
    data = data[~data.index.isin(outliers[feature].index)]

# Use all features for training, but only Time and Date for prediction
X = data[['Time', 'Date', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount']]
y = data['Traffic_Situation']

# Data normalization
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=1)

# Define the improved LSTM model
model_improved = Sequential()
model_improved.add(LSTM(100, return_sequences=True, input_shape=(1, X_reshaped.shape[2])))
model_improved.add(Dropout(0.3))
model_improved.add(LSTM(100, return_sequences=True))
model_improved.add(Dropout(0.3))
model_improved.add(LSTM(50))
model_improved.add(Dropout(0.3))
model_improved.add(Dense(1))

model_improved.compile(optimizer='adam', loss='mean_squared_error')
model_improved.summary()

# Train the LSTM model
history_improved = model_improved.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), verbose=2)

# Prediction on test data with the LSTM model
predictions_improved = model_improved.predict(X_test)
rmse_improved = sqrt(mean_squared_error(y_test, predictions_improved))

# Save the LSTM model and scalers
model_improved.save('lstm_improved_model.h5')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(encoder_traffic, 'encoder_traffic.pkl')

print(f'Combined model RMSE : {rmse_improved}')

# Train the XGBoost model on LSTM predictions
xgb_model = xgb.XGBRegressor()  # Initialize the XGBoost model
xgb_model.fit(X_train.squeeze(), y_train)  # Train the XGBoost model on the actual labels

# Prediction on test data with the XGBoost model
predictions_xgb = xgb_model.predict(X_test.squeeze())

# Combine the LSTM and XGBoost predictions
predictions_combined = (predictions_improved.flatten() + predictions_xgb) / 2  # Average the predictions

# Calculate evaluation metrics for the combined predictions
mae_combined = mean_absolute_error(y_test, predictions_combined)
mse_combined = mean_squared_error(y_test, predictions_combined)
rmse_combined = sqrt(mse_combined)

print(f'MAE (combined): {mae_combined}')
print(f'MSE (combined): {mse_combined}')
print(f'RMSE (combined): {rmse_combined}')

fig_combined = go.Figure()
fig_combined.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='lines', name='Real value'))
fig_combined.add_trace(go.Scatter(x=list(range(len(predictions_combined))), y=predictions_combined, mode='lines', name='Predicted value - Combined model'))
fig_combined.update_layout(title='Comparison of Real and Predicted Values (Combined Model)', xaxis_title='Sample', yaxis_title='Traffic Situation')
fig_combined.write_html("combined_comparison_real_pred.html")
fig_combined.show()

# Visualize evaluation metrics for the combined model
fig_metrics_combined = px.bar(x=['MAE (combined)', 'MSE (combined)', 'RMSE (combined)'], y=[mae_combined, mse_combined, rmse_combined], title='Evaluation Metrics', labels={'x': 'Metric', 'y': 'Value'})
fig_metrics_combined.write_html("combined_metrics_evaluation.html")
fig_metrics_combined.show()

# Plot real and predicted values over time for the combined model
fig_time_combined = go.Figure()
fig_time_combined.add_trace(go.Scatter(x=y_test.index, y=y_test, mode='lines', name='Original data'))
fig_time_combined.add_trace(go.Scatter(x=y_test.index, y=predictions_combined, mode='lines', name='Predicted data (combined)', line=dict(dash='dash')))
fig_time_combined.update_layout(title='Comparison of Original and Predicted Data (Combined)', xaxis_title='Time', yaxis_title='Traffic Situation')
fig_time_combined.write_html("combined_comparison_time.html")
fig_time_combined.show()

# Comparison of predicted and real values
comparison_df = pd.DataFrame({'Real Values': y_test.values.flatten(), 'Predicted Values': predictions_combined.flatten()})
comparison_sample = comparison_df.sample(10)  # Take a sample of 10 values for comparison
print(comparison_sample)

# Save the real values for the combined model
np.save('y_test_combined.npy', y_test)
np.save('predictions_combined.npy', predictions_combined)