import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Dataset de ejemplo
data = {
    'start_station': ['A', 'A', 'B', 'C', 'E'],
    'end_station': ['B', 'C', 'D', 'F', 'F'],
    'cost': [2.0, 1.5, 3.0, 5.0, 2.0],
    'travel_time': [15, 10, 20, 25, 30],
    'demand': [120, 100, 200, 150, 50]
}

# Convertir a DataFrame
df = pd.DataFrame(data)

# Seleccionar las características para clustering
X = df[['cost', 'travel_time', 'demand']]

# Aplicar K-Means
kmeans = KMeans(n_clusters=2)  # Especifica el número de clusters
df['cluster'] = kmeans.fit_predict(X)

# Visualización de clusters
plt.scatter(df['travel_time'], df['cost'], c=df['cluster'], cmap='viridis')
plt.xlabel('Tiempo de Viaje (min)')
plt.ylabel('Costo')
plt.title('Clustering de Rutas de Transporte')
plt.show()
