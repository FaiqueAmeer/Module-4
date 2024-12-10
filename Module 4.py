import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

# To Load the dataset
data = pd.read_csv('U:\Inst 414\police_data.csv')

# For Standardize column names
data.columns = data.columns.str.strip().str.lower().str.replace(' ', '_')

# To Handle missing values
required_columns = [
    'driver_gender', 'driver_age', 'driver_race', 'violation', 
    'search_conducted', 'stop_outcome', 'is_arrested', 'drugs_related_stop'
]
data.dropna(subset=required_columns, inplace=True)

# TO Encode categorical columns
label_encoders = {}
for col in ['driver_gender', 'driver_race', 'violation', 'stop_outcome']:
    le = LabelEncoder()
    data[col + '_encoded'] = le.fit_transform(data[col])
    label_encoders[col] = le

# Selected features for clustering
features = data[[
    'driver_age', 'driver_gender_encoded', 'driver_race_encoded', 
    'violation_encoded', 'search_conducted', 'is_arrested', 'drugs_related_stop'
]]

# To Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Determine the optimal number of clusters using the elbow method
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

# Plots for the elbow curve elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.title('Elbow Method to Determine Optimal k')
plt.show()

# Perform clustering with the optimal k assume k=3 based on elbow plot
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(scaled_features)

# Analyze and interpret clusters
for cluster_id in range(3):
    cluster_data = data[data['cluster'] == cluster_id]
    print(f"Cluster {cluster_id}:")
    print(cluster_data[[
        'driver_gender', 'driver_age', 'driver_race', 
        'violation', 'search_conducted', 'stop_outcome', 'is_arrested'
    ]].head(5))

# Visualize the clusters (age vs. is_arrested)
plt.figure(figsize=(10, 6))
for cluster_id in range(3):
    cluster_points = data[data['cluster'] == cluster_id]
    plt.scatter(
        cluster_points['driver_age'], cluster_points['is_arrested'], 
        label=f'Cluster {cluster_id}'
    )

plt.xlabel('Driver Age')
plt.ylabel('Is Arrested')
plt.title('Clusters Based on Age and Arrest Status')
plt.legend()
plt.show()


data.to_csv('clustered_police_data.csv', index=False)
