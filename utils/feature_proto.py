import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from toolkit import *


# ===================== 配置参数 =====================
class VisualConfig:
    # 颜色配置（10个类别）
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # 标记配置
    feature_marker = 'o'  # 圆圈表示image feature
    proto_marker = '*'  # 星号表示Prototype

    # 尺寸配置
    feature_size = 50  # image feature点大小
    proto_size = 200  # Prototype点大小

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
    """加载并合并image feature和Prototype数据"""
    # 假设每个类别的image feature和Prototype已加载
    # setting = "proof"
    setting = "proofncscmp"
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

    # 展开Prototype数据
    prototypes = prototypes.reshape(-1, 512)  # (10*num_prototypes, 512)

    # 合并数据
    selected_features = np.vstack(selected_features)  # (10 * 20,512) = (200,512)
    combined_data = np.vstack([selected_features, prototypes])  # (200 + 10*N,512)

    # 创建标签
    feature_labels = np.repeat(np.arange(num_classes), samples_per_class)  # (200,)
    proto_labels = np.repeat(np.arange(num_classes), num_prototypes) + 10  # (10*N,)
    combined_labels = np.concatenate([feature_labels, proto_labels])  # (200+10N,)

    return StandardScaler().fit_transform(combined_data), combined_labels,num_prototypes,setting


# ===================== 可视化引擎 =====================
# ===================== 可视化引擎 =====================
def visualize_features_and_prototypes(embeddings, labels, num_prototypes=5,setting=None):
    """支持多Prototype的层级可视化"""
    num_classes = 10
    plt.figure(figsize=(15, 10))

    # 定义Prototype的不同标记（例如每个类别3个不同标记）
    proto_markers = ['*', '^', 's']  # 星号、三角形、正方形

    # 绘制image features（前200个点）
    for cls in range(num_classes):
        mask = (labels == cls) & (labels < 10)
        print(f"Class {cls} 实际样本数:", np.sum(mask))
        plt.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=VisualConfig.colors[cls],
            marker=VisualConfig.feature_marker,
            s=50, alpha=0.6,
            edgecolors='w', linewidths=0.5,
            label=f'Class {cls} Features' if cls == 0 else None
        )

    # 绘制Prototypes（后10*N个点）
    for cls in range(num_classes):
        for proto_id in range(num_prototypes):
            mask = (labels == (cls + 10)) & (labels >= 10)
            # 选择当前Prototype实例的索引
            proto_mask = np.where(labels == (cls + 10))[0][proto_id::num_prototypes]

            plt.scatter(
                embeddings[proto_mask, 0], embeddings[proto_mask, 1],
                c=VisualConfig.colors[cls],
                marker=proto_markers[0],
                s=200, edgecolors='black',
                linewidths=1,
                label=f'Prototype' if cls == 0 else None  # 只显示第一个类别的图例
            )

    # 图例优化
    handles, labels_legend = plt.gca().get_legend_handles_labels()
    plt.legend(handles, ['Image Features', 'Prototype'],
               loc='upper right', framealpha=0.9)

    # 坐标轴美化
    plt.xlabel("t-SNE 1", fontsize=12)
    plt.ylabel("t-SNE 2", fontsize=12)
    # plt.title(f"{num_prototypes} Prototypes per Class", fontsize=14)
    plt.grid(alpha=0.2)
    plt.savefig(f'proto_cover_feat_{setting}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f"proto_cover_feat_{setting}.pdf", format="pdf", bbox_inches='tight', dpi=300)
    plt.show()

# ===================== 主流程 =====================
def main():
    # 数据准备
    combined_data, labels, num_prototypes,setting = load_and_preprocess()

    # 执行t-SNE
    print("Running t-SNE...")
    tsne = TSNE(**VisualConfig.tsne_params)
    embeddings = tsne.fit_transform(combined_data)

    # 可视化
    visualize_features_and_prototypes(embeddings, labels,num_prototypes,setting)


if __name__ == "__main__":
    main()