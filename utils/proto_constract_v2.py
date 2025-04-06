import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ===================== 配置参数 =====================
class VisualConfig:
    color_maps = ['red', 'red', 'black']  # 颜色顺序：Proof, Ours, ETF
    markers = ['o', 'o', '*']  # 标记类型：圆形，方形，星形
    sizes = [80, 80, 80]  # 尺寸大小
    alphas = [1.0, 1.0, 1.0]  # 透明度
    labels = ['Proof', 'Ours', 'ETF']  # 图例标签

    # t-SNE参数
    tsne_params = {
        'n_components': 2,
        'perplexity': 30,
        'n_iter': 2000,
        'learning_rate': 300,
        'metric': 'cosine',
        'random_state': 42
    }


# ===================== 数据加载与处理 =====================
def load_and_preprocess():
    """加载并合并三组Prototype数据"""
    # 加载数据（示例路径，请根据实际修改）
    proof = torch.load("proof.pth").view(100, -1).cpu().numpy()
    ours = torch.load("proofncscmp.pth").view(100, -1, 512)[:, 0, :].cpu().numpy()
    etf = torch.load("proofncscmp_etf.pth").view(100, -1).cpu().numpy()

    # 合并数据并创建标签
    combined = np.concatenate([proof, ours, etf])
    labels = np.array([0] * 100 + [1] * 100 + [2] * 100)

    # PCA降维
    from sklearn.decomposition import PCA
    pca = PCA(n_components=120)
    combined = pca.fit_transform(combined)
    return combined, labels


# ===================== 可视化引擎 =====================
def visualize_prototypes(embeddings, labels, titles):
    """极简专业风格可视化（合并每组为一个整体）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 隐藏坐标轴元素
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_color('black')
        ax.spines[:].set_linewidth(2)

    # ---- 左子图：Proof组 ----
    proof_indices = np.where(labels == 0)[0]
    ax1.scatter(
        embeddings[proof_indices, 0], embeddings[proof_indices, 1],
        c=VisualConfig.color_maps[0],
        marker=VisualConfig.markers[0],
        s=VisualConfig.sizes[0],
        alpha=VisualConfig.alphas[0],
        edgecolors='w',
        linewidths=0.8,
        label=VisualConfig.labels[0]
    )
    ax1.text(0.5, -0.08, f'({titles[0][0]}) {titles[0][1]}',
             transform=ax1.transAxes, ha='center', va='top', fontsize=18)

    # ---- 右子图：Ours + ETF ----
    # 绘制Ours组
    ours_indices = np.where(labels == 1)[0]
    ax2.scatter(
        embeddings[ours_indices, 0], embeddings[ours_indices, 1],
        c=VisualConfig.color_maps[1],
        marker=VisualConfig.markers[1],
        s=VisualConfig.sizes[1],
        alpha=VisualConfig.alphas[1],
        linewidths=0.8,
        label=VisualConfig.labels[1]
    )

    # 绘制ETF组
    etf_indices = np.where(labels == 2)[0]
    ax2.scatter(
        embeddings[etf_indices, 0], embeddings[etf_indices, 1],
        c=VisualConfig.color_maps[2],
        marker=VisualConfig.markers[2],
        s=VisualConfig.sizes[2],
        alpha=VisualConfig.alphas[2],
        linewidths=0.8,
        label=VisualConfig.labels[2]
    )
    ax2.text(0.5, -0.08, f'({titles[1][0]}) {titles[1][1]}',
             transform=ax2.transAxes, ha='center', va='top', fontsize=18)

    plt.subplots_adjust(bottom=0.10)  # 减小底部边距
    plt.tight_layout()

    plt.savefig('prototype_tsne.png', dpi=300)
    plt.savefig("prototype_tsne.pdf", format="pdf",  dpi=300)
    plt.close()


# ===================== 主流程 =====================
def main():
    combined_data, labels = load_and_preprocess()

    # 数据标准化
    from sklearn.preprocessing import StandardScaler
    combined_data = StandardScaler().fit_transform(combined_data)

    # 执行t-SNE
    print("Running t-SNE...")
    tsne = TSNE(**VisualConfig.tsne_params)
    embeddings = tsne.fit_transform(combined_data)

    # 可视化
    visualize_prototypes(embeddings, labels, titles=[('a', 'Proof'), ('b', 'Our Method')])


if __name__ == "__main__":
    main()