import numpy as np
import plotly.graph_objects as go

# Charger les valeurs réelles pour tous les modèles
y_test_base = np.load('y_test.npy')
y_test_improved = np.load('y_test.npy')
y_test_combined = np.load('y_test_combined.npy')

# Charger les prédictions pour tous les modèles
predictions_base = np.load('predictions_base.npy')
predictions_improved = np.load('predictions_improved.npy')
predictions_combined = np.load('predictions_combined.npy')

# Définir le nombre d'échantillons à utiliser pour la visualisation
sample_size = 200  # Choisissez le nombre d'échantillons que vous souhaitez afficher

# Création du graphe pour la comparaison des valeurs réelles et prédictions
fig_comparison = go.Figure()

# Ajouter les valeurs réelles
fig_comparison.add_trace(go.Scatter(x=list(range(sample_size)), y=y_test_improved[:sample_size], mode='lines', name='Real data', line=dict(color='red')))

# Ajouter les prédictions de chaque modèle
fig_comparison.add_trace(go.Scatter(x=list(range(sample_size)), y=predictions_base[:sample_size].flatten(), mode='lines', name='Basic LSTM prediction', line=dict(color='blue', dash='dash')))
fig_comparison.add_trace(go.Scatter(x=list(range(sample_size)), y=predictions_improved[:sample_size].flatten(), mode='lines', name='Improved LSTM prediction', line=dict(color='green', dash='dash')))
fig_comparison.add_trace(go.Scatter(x=list(range(sample_size)), y=predictions_combined[:sample_size].flatten(), mode='lines', name='Combined LSTM prediction', line=dict(color='orange', dash='dash')))

# Mise à jour du layout du graphe
fig_comparison.update_layout(title='Comparison of Actual Values and Predictions for All Models (First 200 Samples)', xaxis_title='Sample', yaxis_title='Traffic Situation')

# Enregistrer le graphe au format HTML
fig_comparison.write_html("comparison_real_pred.html")

# Afficher le graphe dans le notebook (optionnel)
fig_comparison.show()
