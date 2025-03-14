import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from toolkit import *


def visualize_dual_pdf(embeddings_list, labels_list, titles, output_name="dual_methods.pdf"):
    """双方法PDF并排可视化"""
    with PdfPages(output_name) as pdf:
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))

        for ax, embeddings, labels, title in zip(axes, embeddings_list, labels_list, titles):
            # --- 可视化逻辑 ---
            num_classes = 10

            # 绘制Image Features
            for cls in range(num_classes):
                mask = (labels == cls) & (labels < 10)
                ax.scatter(
                    embeddings[mask, 0], embeddings[mask, 1],
                    c=VisualConfig.colors[cls],
                    marker=VisualConfig.feature_marker,
                    s=50, alpha=1,
                    edgecolors='w', linewidths=0.5
                )

            # 绘制Prototypes
            for cls in range(num_classes):
                mask = (labels == (cls + 10)) & (labels >= 10)
                ax.scatter(
                    embeddings[mask, 0], embeddings[mask, 1],
                    c=VisualConfig.colors[cls],
                    marker=VisualConfig.proto_marker,
                    s=200, edgecolors='black',
                    linewidths=1
                )

            # --- 子图样式配置 ---
            ax.text(0.5, -0.08, f'({title[0]}) {title[1]}',
                    transform=ax.transAxes,
                    ha='center', va='top', fontsize=18)

            # 移除坐标轴标签
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(0.8)

            # --- 图例配置（左上角）---
            marker_legend = [
                plt.Line2D([0], [0], marker='o', color='w',
                           label='Image Features',
                           markersize=12,
                           markerfacecolor='gray'),
                plt.Line2D([0], [0], marker='*', color='w',
                           label='Prototype',
                           markersize=18,
                           markerfacecolor='gray',
                           markeredgecolor='black')
            ]

            ax.legend(
                handles=marker_legend,
                loc='upper left',
                bbox_to_anchor=(0.02, 0.98),  # 左上角微调
                fontsize=12,
                frameon=True,
                framealpha=0.9,
                edgecolor='black'
            )

        plt.tight_layout()
        pdf.savefig(bbox_inches='tight')
        plt.close()

class VisualConfig:
    # 颜色配置（10个类别）
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    colors = [
        '#FF0000', '#0000FF', '#000000',  # 红、蓝、黑
        '#00FF00', '#FF00FF', '#A020F0',  # 绿、粉、亮紫（替换黄色）
        '#00FFFF', '#800080', '#FFA500',  # 青、紫、橙
        '#008000'  # 深绿
    ]

    # 标记配置
    feature_marker = 'o'  # 圆圈表示image feature
    proto_marker = '*'  # 星号表示Prototype

    # 尺寸配置
    feature_size = 50  # image feature点大小
    proto_size = 200  # Prototype点大小

    # t-SNE参数
    tsne_params = {
        'n_components': 2,
        'perplexity': 50,        # 增大perplexity以分散全局结构
        'n_iter': 5000,          # 增加迭代次数确保收敛
        'learning_rate': 500,    # 增大学习率避免局部最优
        'metric': 'cosine',
        'random_state': 42,
        'early_exaggeration': 24 # 增强前期簇分离
    }


# ===================== 数据加载与处理 =====================
def load_and_preprocess(setting):
    """加载并合并image feature和Prototype数据"""
    # 假设每个类别的image feature和Prototype已加载
    # setting = "proof"
    # setting = "proofncscmp"
    if "mp" in setting:
        image_features = torch.load("proofncscmp_class_image.pth").cpu().numpy()
        prototypes = torch.load("proofncscmp.pth").view(100,-1,512).cpu().numpy()
        num_prototypes = prototypes.shape[1]
    else:
        image_features = torch.load("proof_class_image.pth").cpu().numpy()
        prototypes = torch.load("proof.pth").cpu().numpy()
        num_prototypes = 1
    etf = torch.load("proofncscmp_etf.pth").view(100, -1).cpu().numpy()
    selected_indices = select_vectors(etf)
    image_features = image_features[selected_indices,:,:]
    prototypes = prototypes[selected_indices,:]

    # 从每个类别的64个样本中选择20个
    num_classes = 10
    samples_per_class = 64
    selected_features = []

    for cls in range(num_classes):
        # 随机选择20个样本（可替换为您的select_vectors逻辑）
        np.random.seed(42)  # 确保可重复性
        # idx = np.random.choice(64, samples_per_class, replace=False)
        idx = list(range(samples_per_class))
        selected_features.append(image_features[cls, idx, :])
    selected_features = np.vstack(selected_features)  # (10 * 20,512) = (200,512)
    # selected_features = image_features.reshape(-1,512)

    # 展开Prototype数据
    prototypes = prototypes.reshape(-1, 512)  # (10*num_prototypes, 512)

    # 合并数据
    combined_data = np.vstack([selected_features, prototypes])  # (200 + 10*N,512)

    # 创建标签
    feature_labels = np.repeat(np.arange(num_classes), samples_per_class)  # (200,)
    proto_labels = np.repeat(np.arange(num_classes), num_prototypes) + 10  # (10*N,)
    combined_labels = np.concatenate([feature_labels, proto_labels])  # (200+10N,)
    # 先进行PCA降维到50维
    from sklearn.decomposition import PCA
    pca = PCA(n_components=25)
    combined_data_pca = pca.fit_transform(combined_data)

    return StandardScaler().fit_transform(combined_data_pca), combined_labels, num_prototypes, setting
    # return StandardScaler().fit_transform(combined_data), combined_labels,num_prototypes,setting


def main():
    # 数据准备
    sets = ["proof","proofncscmp"]
    embs = []
    labs=[]
    for setting in sets:
        combined_data, labels, num_prototypes,setting = load_and_preprocess(setting)

        # 执行t-SNE
        print("Running t-SNE...")
        tsne = TSNE(**VisualConfig.tsne_params)
        embeddings = tsne.fit_transform(combined_data)

        embs.append(embeddings)
        labs.append(labels)

    # 调用可视化函数
    visualize_dual_pdf(
        embeddings_list=embs,
        labels_list=labs,
        titles=[('a', 'Proof'), ('b', 'Our Method')],
        output_name="feat-proto_comparison.pdf"
    )


if __name__ == "__main__":
    main()