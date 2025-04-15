from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def apply_pca(df_scaled):
    pca = PCA(n_components=2) 
    pca_result = pca.fit_transform(df_scaled)
    explained_variance = pca.explained_variance_ratio_

    # plt.figure(figsize=(8,5))
    # plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o', linestyle='--')
    # plt.xlabel('Number of Principal Components')
    # plt.ylabel('Explained Variance')
    # plt.title('PCA Explained Variance')
    # plt.savefig("static/graphs/pca_explained_variance.png")
    # plt.close()

    return pca_result

def find_best_k(pca_result):
    inertia = []
    sil_scores = []
    k_range = range(2, 20)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(pca_result)
        inertia.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(pca_result, kmeans.labels_))

    inertia_diff = np.diff(inertia)
    optimal_k_elbow = k_range[np.argmin(inertia_diff) + 1]

    optimal_k_silhouette = k_range[np.argmax(sil_scores)]

    # print(f"Optimal k by Elbow Method: {optimal_k_elbow}")
    # print(f"Optimal k by Silhouette Score: {optimal_k_silhouette}")

    # plt.figure(figsize=(8,5))
    # plt.plot(k_range, inertia, marker='o')
    # plt.axvline(optimal_k_elbow, color='red', linestyle='--', label=f'Elbow k={optimal_k_elbow}')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Inertia')
    # plt.title('Elbow Method for Optimal k')
    # plt.legend()
    # plt.savefig("static/graphs/elbow_method.png")
    # plt.close()

    # plt.figure(figsize=(8,5))
    # plt.plot(k_range, sil_scores, marker='o', color='green')
    # plt.axvline(optimal_k_silhouette, color='red', linestyle='--', label=f'Best k={optimal_k_silhouette}')
    # plt.xlabel('Number of Clusters (k)')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Scores for Different k')
    # plt.legend()
    # plt.savefig("static/graphs/silhouette_scores.png")
    # plt.close()

    return optimal_k_elbow, optimal_k_silhouette

def visualize_clusters(pca_result, labels, centroids, best_k):
    plt.figure(figsize=(8, 6))

    for i in range(best_k):
        plt.scatter(pca_result[labels == i, 0], pca_result[labels == i, 1], label=f'Cluster {i}')

    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='black', label='Centroids')

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'K-Means Clustering Visualization (k={best_k})')
    plt.legend()
    plt.savefig("static/graphs/kmeans_clusters.png")
    plt.close()