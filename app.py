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
    km = KMeans(n_clusters=i)
    km.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
    sse.append(km.inertia_)

plt.xlabel('I')
plt.ylabel('Sum of squared error')
plt.plot(range(1, 10), sse)

# Save SSE plot dijadiin IMAGE biar bisa tampil
sse_img_bytesio = BytesIO()
plt.savefig(sse_img_bytesio, format='png')
sse_img_bytesio.seek(0)
sse_img_data = base64.b64encode(sse_img_bytesio.read()).decode('utf-8')

# --- PLOTTING CLUSTERING
clustering_plot_filename = 'static/clustering_plot.png'
km = KMeans(n_clusters=5)
y_predicted = km.fit_predict(df[['Annual Income (k$)', 'Spending Score (1-100)']])
df["Clusters"] = y_predicted

plt.figure()

plt.scatter(df['Annual Income (k$)'], df['Spending Score (1-100)'], c=df['Clusters'], cmap='rainbow')
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

plt.figure()
plt.bar(cluster_counts.index, cluster_counts.values, color='skyblue')
plt.title('Cluster Distribution')
plt.xlabel('Cluster')
plt.ylabel('Number of Data')

cluster_distribution_img_bytesio = BytesIO()
plt.savefig(cluster_distribution_img_bytesio, format='png')
cluster_distribution_img_bytesio.seek(0)
cluster_distribution_img_data = base64.b64encode(cluster_distribution_img_bytesio.read()).decode('utf-8')

# --- PLOTTING GENDER DISTRIBUTION IN CLUSTERS
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

# --- PLOTTING AGE DISTRIBUTION IN CLUSTERS
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

@app.route('/')
def index():
    data = df.to_dict('records')
    return render_template('index.html', data=data)


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
