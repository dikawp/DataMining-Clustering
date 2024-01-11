from flask import Flask, render_template
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from io import BytesIO
import base64
import warnings

app = Flask(__name__)
warnings.filterwarnings("ignore")


# --- AMBIL DATASET
url = 'https://drive.google.com/file/d/1uNeh9zMmJtBRfp9-z6Ch_tIOh0aqkfCe/view?usp=sharing'  # file awal
path = 'https://drive.google.com/uc?export=download&id=' + url.split('/')[-2]

df = pd.read_csv(path)

# --- PLOTTING SSE
sse_plot_filename = 'static/sse_plot.png'
sse = []
for i in range(1, 10):
    km = KMeans(n_clusters=i, init='k-means++', random_state=42)  # Inisialisasi pusat klaster
    km.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    sse.append(km.inertia_)

plt.xlabel('Cluster')
plt.ylabel('Sum of squared error')
plt.plot(range(1, 10), sse)

sse_img_bytesio = BytesIO()
plt.savefig(sse_img_bytesio, format='png')
sse_img_bytesio.seek(0)
sse_img_data = base64.b64encode(sse_img_bytesio.read()).decode('utf-8')

# --- PLOTTING CLUSTERING
clustering_plot_filename = 'static/clustering_plot.png'
km = KMeans(n_clusters=5, init='k-means++', random_state=42)  # Inisialisasi pusat klaster
y_predicted = km.fit_predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])
df["Clusters"] = y_predicted + 1  # Menambah 1 ke setiap nilai klaster

centroid = km.cluster_centers_
# print(centroid)

plt.figure()

plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Clusters'], cmap='cividis')
plt.scatter(centroid[:, 1], centroid[:, 0], c='red', marker='X', s=100)
plt.title('Customers clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')

clustering_img_bytesio = BytesIO()
plt.savefig(clustering_img_bytesio, format='png')
clustering_img_bytesio.seek(0)
clustering_img_data = base64.b64encode(clustering_img_bytesio.read()).decode('utf-8')

# --- PLOTTING CLUSTER DISTRIBUTION
cluster_distribution_plot_filename = 'static/cluster_distribution_plot.png'
cluster_counts = df['Clusters'].value_counts().sort_index()

# Get unique colors for each cluster
colors = plt.cm.Spectral(range(len(cluster_counts)))

plt.figure()
plt.bar(cluster_counts.index, cluster_counts.values, color=colors)
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Number of Data')

# Display data points within each cluster
for i, count in enumerate(cluster_counts.values):
    plt.text(cluster_counts.index[i], count + 0.1, str(count), ha='center', va='bottom', fontsize=8)

cluster_distribution_img_bytesio = BytesIO()
plt.savefig(cluster_distribution_img_bytesio, format='png')
cluster_distribution_img_bytesio.seek(0)
cluster_distribution_img_data = base64.b64encode(cluster_distribution_img_bytesio.read()).decode('utf-8')


# --- PLOTTING GENDER
gender_distribution_plot_filename = 'static/gender_distribution_plot.png'
gender_distribution = df.groupby(['Clusters', 'Gender']).size().unstack().fillna(0)

plt.figure()

gender_distribution.plot(kind='bar', stacked=True, color=['lightblue', 'lightcoral'])
plt.title('Gender Distribution in Clusters')
plt.xlabel('Cluster')
plt.ylabel('Number of Data')

gender_distribution_img_bytesio = BytesIO()
plt.savefig(gender_distribution_img_bytesio, format='png')
gender_distribution_img_bytesio.seek(0)
gender_distribution_img_data = base64.b64encode(gender_distribution_img_bytesio.read()).decode('utf-8')


# --- PLOTTING AGE 
age_distribution_plot_filename = 'static/age_distribution_plot.png'

plt.figure()

for cluster in range(5):
    plt.hist(df[df['Clusters'] == cluster]['Age'], bins=20, alpha=0.5, label=f'Cluster {cluster + 1}')

plt.title('Age Distribution in Clusters')
plt.xlabel('Age')
plt.ylabel('Number of Data')
plt.legend()

age_distribution_img_bytesio = BytesIO()
plt.savefig(age_distribution_img_bytesio, format='png')
age_distribution_img_bytesio.seek(0)
age_distribution_img_data = base64.b64encode(age_distribution_img_bytesio.read()).decode('utf-8')


# --- ROUTING 
@app.route('/')
def index():
    sorted_data = df.sort_values(by='Clusters')
    
    sorted_data_records = sorted_data.to_dict('records')
    return render_template('index.html', data=sorted_data_records)

@app.route('/clustered')
def clustered():
    sorted_data = df.sort_values(by='Clusters')
    
    sorted_data_records = sorted_data.to_dict('records')
    return render_template('clustered.html', data=sorted_data_records)

@app.route('/sse_page')
def sse_page():
    return render_template('sse_page.html', sse_img_data=sse_img_data)


@app.route('/clustering_page')
def clustering_page():
    return render_template('clustering_page.html', clustering_img_data=clustering_img_data,
                           cluster_distribution_img_data=cluster_distribution_img_data,
                           gender_distribution_img_data=gender_distribution_img_data,
                           age_distribution_img_data=age_distribution_img_data)


if __name__ == '__main__':
    app.run(debug=True)
