import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import SpectralClustering, KMeans
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def soft_kmeans(X, K, max_iters=100, sigma=1.0, tol=1e-4,visualize=False,tempra=1.0):
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
        # method1：
        # distances = cdist(X, centroids, 'cosine')  # 计算每个点到所有簇中心的距离
        # for i in range(N):
        #     计算每个点对所有簇的隶属度
            # U[i] = np.exp(-distances[i]**2 / (2 * sigma**2))
            # U[i] /= np.sum(U[i])  # 归一化隶属度矩阵
        # method2：两种方法没区别，结果一致
        cosine_sim = np.dot(X, centroids.T) / (np.linalg.norm(X, axis=1, keepdims=True) * np.linalg.norm(centroids, axis=1))  # 计算余弦相似度
        # 计算每个点对每个簇的隶属度
        for i in range(N):
            numerator = np.exp(cosine_sim[i] / tempra)  # 对每个簇的余弦相似度应用指数函数
            U[i] = numerator / np.sum(numerator)  # 归一化隶属度矩阵

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


def gen_mc_proto(input_data,proto_num,gen_proto_mode,seed):
    
    cur_proto, cur_proto_var, cur_proto_sim, cur_features = [], [], [], []
    if gen_proto_mode == 'spectral':
        affinity_matrix = cosine_similarity(input_data, input_data)
        clustering = SpectralClustering(n_clusters=proto_num, assign_labels='discretize', affinity='precomputed', n_init=10, random_state=seed)
        affinity_matrix = affinity_matrix.cpu().numpy()
        clustering.fit_predict(affinity_matrix)
    elif gen_proto_mode == 'kmeans':
        clustering = KMeans(n_clusters=proto_num, random_state=seed)
        clustering.fit(input_data.cpu().numpy())
    elif gen_proto_mode == 'kmeans++':
        centroids, U = KmeansPlus(input_data.cpu().numpy(), proto_num)
        # clustering = KMeans(n_clusters=proto_num, init='k-means++', max_iter=300, n_init=10, random_state=seed)
        # clustering.fit(input_data.cpu().numpy())
        return torch.tensor(centroids)
    elif gen_proto_mode == 'soft_kmeans':
        centroids, U = soft_kmeans(input_data.cpu().numpy(), proto_num, max_iters=100, sigma=1.0, tol=1e-4)
        return torch.tensor(centroids)
    else:
        raise NotImplementedError

    for label in range(proto_num):
        feature = input_data[clustering.labels_ == label, :]
        if not torch.is_tensor(feature):
            feature = torch.tensor(feature).cuda()
        var, mean = torch.var_mean(feature, dim=0)
        cur_proto.append(mean)
    cur_proto = torch.stack(cur_proto, dim=0)
    return cur_proto


def cosine_similarity(input, target):
    """
    input: (dim_input, embed_dim)
    target: (dim_ouput, embed_dim)
    similarity: (dim_input, dim_ouput)
    """
    input_norm = torch.nn.functional.normalize(input, dim=1, p=2)
    target_norm = torch.nn.functional.normalize(target, dim=1, p=2)
    similarity = _safe_matmul(input_norm, target_norm)
    similarity  = torch.nan_to_num(similarity)
    eps = 1e-8
    similarity[similarity<=eps] = eps
    # similarity[torch.isnan(similarity)] = eps
    return similarity


def _safe_matmul(x, y):
    """Safe calculation of matrix multiplication.
    If input is float16, will cast to float32 for computation and back again.
    """
    if x.dtype == torch.float16 or y.dtype == torch.float16:
        return torch.matmul(x.float(), y.float().t()).half()
    return torch.matmul(x, y.t())
