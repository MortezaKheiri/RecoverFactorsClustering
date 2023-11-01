import numpy as np
from sklearn.cluster import KMeans

def cluster(list_of_data, n_clusters):
  """Clusters a list of data into n clusters using the K-Means algorithm.
  """

  # Convert the list of data points to a NumPy array.
  data_array = np.array(list_of_data).reshape(-1, 1)

  # Create a KMeans object with n_clusters clusters.
  kmeans = KMeans(n_clusters=n_clusters)

  # Fit the KMeans object to the data.
  kmeans.fit(data_array)

  # Predict the cluster labels for each data point.
  cluster_labels = kmeans.predict(data_array)

  return cluster_labels

def elbow_method(list_of_data, max_clusters=10):
  """Detects the best n_clusters value for the K-Means algorithm using the elbow method.
  """
  # Convert the list of data points to a NumPy array.
  data_array = np.array(list_of_data).reshape(-1, 1)

  # Compute the within-cluster sum of squares (WCSS) for each number of clusters.
  wcss = []
  for n_clusters in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data_array)
    wcss.append(kmeans.inertia_)

  # Compute the difference in WCSS between consecutive numbers of clusters.
  delta_wcss = np.diff(wcss)

  # Find the index of the largest delta_wcss. This is the elbow point.
  elbow_index = np.argmax(delta_wcss)

  # Return the best n_clusters value, which is the number of clusters at the elbow point.
  best_n_clusters = elbow_index + 1

  return best_n_clusters

# Sorted Recovery Factors as list_of_data
# -------------[RR2, RR3, RR4, RR1, RR10, RR6, RR8, RR5]
list_of_data = [0.39, 0.44, 0.53, 0.63, 0.67, 0.78, 0.92, 1.09]
n = elbow_method(list_of_data, max_clusters=8)

clusters = cluster(list_of_data, n)

print(clusters)
