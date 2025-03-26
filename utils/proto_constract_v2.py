import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ===================== 配置参数 =====================
class VisualConfig:
    # 颜色配置（每个组生成10种渐变色）
    color_maps = ['#FF0000', '#0000FF', '#000000'] # 红，蓝，绿
    markers = ['o', 'o', '*']  # 圆形，圆形，星形
    sizes = [100, 100, 120]  # 标记尺寸
    alphas = [1, 1, 1]  # 完全不透明
    labels = ['Proof', 'Ours', 'ETF']

    # t-SNE参数
    tsne_params = {
        'n_components': 2,
        'perplexity': 29,
        'n_iter': 2000,
        'learning_rate': 300,
        'metric': 'cosine',
        'random_state': 42
    }


# ===================== 数据加载与处理 =====================
def load_and_preprocess():
    """加载并合并三组Prototype数据"""
    # 加载原始数据（示例路径，请根据实际修改）
    proof = torch.load("proof.pth").view(100, -1).cpu().numpy()
    ours = torch.load("proofncscmp.pth").view(100, -1, 512)[:, 0, :].cpu().numpy()
    etf = torch.load("proofncscmp_etf.pth").view(100, -1).cpu().numpy()
    # ours = torch.load("proofnc.pth").view(100, -1).cpu().numpy()
    # etf = torch.load("proofnc_etf.pth").view(100, -1).cpu().numpy()

    # 合并数据并创建标签
    combined = np.concatenate([proof, ours, etf])
    labels = np.array([0] * 100 + [1] * 100 + [2] * 100)

    # 先进行PCA降维到50维
    from sklearn.decomposition import PCA
    pca = PCA(n_components=91)  #数字越小，靠的越近
    combined = pca.fit_transform(combined)
    return combined, labels


# ===================== 可视化引擎 =====================
def visualize_prototypes(embeddings, labels, titles):
    """极简专业风格可视化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), facecolor='white')

    # 隐藏坐标轴元素
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_color('black')  # 保留框线
        ax.spines[:].set_linewidth(0.8)

    # 绘制左子图（Proof组）
    proof_indices = np.where(labels == 0)[0]
    for i in range(10):
        segment = proof_indices[i * 10: (i + 1) * 10]
        ax1.scatter(
            embeddings[segment, 0], embeddings[segment, 1],
            c=[VisualConfig.color_maps[0]],
            marker=VisualConfig.markers[0],
            s=VisualConfig.sizes[0],
            alpha=1,
            edgecolors='w',
            linewidths=0.8,
            label=VisualConfig.labels[0] if i == 0 else ""
        )
    # --- 子图标题配置 ---
    ax1.text(0.5, -0.12,  # 调整y坐标位置
             f'({titles[0][0]}) {titles[0][1]}',
             transform=ax1.transAxes,  # 使用当前轴的坐标系
             ha='center', va='top', fontsize=18)

    # 绘制右子图（Ours + ETF）
    for group, offset in [(1, 0), (2, 10)]:
        group_indices = np.where(labels == group)[0]
        for i in range(10):
            segment = group_indices[i * 10: (i + 1) * 10]
            ax2.scatter(
                embeddings[segment, 0], embeddings[segment, 1],
                c=[VisualConfig.color_maps[group - 1]],
                marker=VisualConfig.markers[group],
                s=VisualConfig.sizes[group],
                alpha=1,
                edgecolors='w',
                linewidths=0.8,
                label=VisualConfig.labels[group] if i == 0 else ""
            )
    # --- 子图标题配置 ---
    ax2.text(0.5, -0.12,  # 保持相同偏移量
             f'({titles[1][0]}) {titles[1][1]}',
             transform=ax2.transAxes,  # 使用当前轴的坐标系
             ha='center', va='top', fontsize=18)

    # 调整布局边距
    plt.subplots_adjust(bottom=0.15)  # 增加底部边距
    plt.tight_layout()

    plt.savefig('prototype_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig("prototype_comparison.pdf", format="pdf", bbox_inches='tight', dpi=300)
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
    visualize_prototypes(embeddings, labels,titles=[('a', 'Proof'), ('b', 'Our Method')])


if __name__ == "__main__":
    main()