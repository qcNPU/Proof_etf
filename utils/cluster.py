import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def soft_kmeans(X, K, max_iters=100, sigma=1.0, tol=1e-4,visualize=False):
    # X: 输入数据集，shape 为 (N, D)，N为样本数，D为特征数
    # K: 簇的数量
    # max_iters: 最大迭代次数
    # sigma: 高斯核的宽度
    # tol: 收敛阈值

    # 随机初始化簇中心
    N, D = X.shape
    centroids = X[np.random.choice(N, K, replace=False)]  # 随机选取 K 个点作为初始质心

    # 计算隶属度矩阵
    U = np.zeros((N, K))

    for iteration in range(max_iters):
        # Step 1: 计算隶属度矩阵 U
        distances = cdist(X, centroids, 'euclidean')  # 计算每个点到所有簇中心的距离
        for i in range(N):
            # 计算每个点对所有簇的隶属度
            U[i] = np.exp(-distances[i]**2 / (2 * sigma**2))
            U[i] /= np.sum(U[i])  # 归一化隶属度矩阵

        # Step 2: 更新簇中心
        new_centroids = np.zeros_like(centroids)
        for j in range(K):
            # 计算簇中心的加权平均
            weighted_sum = np.sum((U[:, j] ** 2).reshape(-1, 1) * X, axis=0)
            new_centroids[j] = weighted_sum / np.sum(U[:, j] ** 2)

        # 检查是否收敛
        centroid_shift = np.linalg.norm(new_centroids - centroids)
        if centroid_shift < tol:
            print(f"Converged at iteration {iteration}")
            break

        centroids = new_centroids


    if visualize:
        plt.figure(figsize=(8, 6))
        # 使用 t-SNE 降维到 2D（为了可视化）
        tsne = TSNE(n_components=2, random_state=42, init='pca')
        X_tsne = tsne.fit_transform(X)

        # 可视化聚类
        plt.cla()  # 清空当前图像
        for j in range(K):
            plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=U[:, j], cmap='viridis', alpha=0.5, label=f"Cluster {j + 1}")
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')

        # plt.title(f"Iteration {iteration + 1}")
        plt.xlabel("t-SNE component 1")
        plt.ylabel("t-SNE component 2")
        plt.legend()
        plt.colorbar(label="Membership degree")
        plt.pause(0.1)  # 暂停一小段时间，更新图像
        plt.show()

    return centroids, U

def KmeansPlus(X,K,visualize=False):
    # 使用 Scikit-learn 的 KMeans 类进行聚类，K-means++ 默认被启用
    kmeans = KMeans(n_clusters=K, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X)

    # 获取簇中心和标签
    centroids = kmeans.cluster_centers_  # 簇中心
    labels = kmeans.labels_  # 每个点所属簇的标签

    if visualize:
        # 使用 t-SNE 将数据降到 2D
        tsne = TSNE(n_components=2, random_state=42, init='pca')
        X_tsne = tsne.fit_transform(X)
        # 可视化 t-SNE 结果
        plt.figure(figsize=(8, 6))

        # 使用不同的颜色绘制每个簇的点
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)

        # 添加颜色条
        plt.colorbar(scatter)

        # 设置标题
        plt.title("K-means Clustering with t-SNE Visualization")
        plt.xlabel("t-SNE component 1")
        plt.ylabel("t-SNE component 2")

        # 显示图像
        plt.show()

    return centroids,labels



def KmeansPlus_returnfeature(X, K):#这个是将属于各个簇的特征分别放到各个list中并返回
    clt = KMeans(n_clusters=K)
    clt.fit(X)
    labelIDs = np.unique(clt.labels_)

    feature_clusters=[]
    for labelID in labelIDs:
        idxs = np.where(clt.labels_ == labelID)[0]
        idxs = np.random.choice(idxs, size=min(30, len(idxs)),
            replace=False)
        cluster = []
        for i in idxs:
            cluster.append(((X[i]).squeeze(0)))

        cluster1 = torch.stack(cluster,dim=0)
        feature_clusters.append(cluster1)

    return torch.stack(feature_clusters,0)

