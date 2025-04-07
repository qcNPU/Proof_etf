import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ===================== 配置参数 =====================
class VisualConfig:
    color_maps = ['red', 'red', 'black']  # Proof, Ours, ETF
    markers = ['o', 'o', '*']  # 标记类型
    sizes = [100, 100, 100]  # 尺寸
    alphas = [1.0, 1.0, 1.0]  # 透明度
    labels = ['Proof', 'Ours', 'ETF']  # 图例

    # 聚焦框参数 (x_min, y_min, x_max, y_max)
    zoom_boxes = {
        # Proof组（左子图）聚焦框
        'left': (
            -3.0,  # x_min = -10.89 + (20.09 - (-10.89))/3
            -18.0,  # y_min = -28.38 + (-4.90 - (-28.38))/3
            7.0,  # x_max = -10.89 + 2*(20.09 - (-10.89))/3
            -8.0  # y_max = -28.38 + 2*(-4.90 - (-28.38))/3
        ),
        # Ours+ETF组（右子图）聚焦框
        'right': (
            -7.0,  # x_min = -22.50 + (23.67 - (-22.50))/3
            9.0,  # y_min = -8.22 + (30.36 - (-8.22))/3
            8.0,  # x_max = -22.50 + 2*(23.67 - (-22.50))/3
            24.0  # y_max = -8.22 + 2*(30.36 - (-8.22))/3
        )
    }
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
    pca = PCA(n_components=140, random_state=42)
    combined = pca.fit_transform(combined)
    return combined, labels


# ===================== 可视化引擎 =====================
def draw_focus_boxes(ax, box_type, color='blue', linestyle='--'):
    """在指定坐标轴上绘制单个聚焦框"""
    x_min, y_min, x_max, y_max = VisualConfig.zoom_boxes[box_type]
    ax.add_patch(plt.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=4,
        edgecolor=color,
        facecolor='none',
        linestyle=linestyle
    ))


def visualize_prototypes(embeddings, labels, titles):
    """极简专业风格可视化（修复框线错位问题）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # ---- 左子图：Proof组 ----
    proof_idx = np.where(labels == 0)[0]
    ax1.scatter(
        embeddings[proof_idx, 0], embeddings[proof_idx, 1],
        c=VisualConfig.color_maps[0],
        marker=VisualConfig.markers[0],
        s=VisualConfig.sizes[0]*1.1,
        alpha=VisualConfig.alphas[0],
        edgecolors='w',
        linewidths=0.8
    )

    # 设置左图坐标范围
    x_min, x_max = embeddings[proof_idx, 0].min(), embeddings[proof_idx, 0].max()
    y_min, y_max = embeddings[proof_idx, 1].min(), embeddings[proof_idx, 1].max()
    ax1.set_xlim(x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min))
    ax1.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))

    # 添加左图聚焦框（关键修改点）
    draw_focus_boxes(ax1, box_type='left')  # 明确指定框类型

    # ---- 右子图：Ours + ETF ----
    ours_idx = np.where(labels == 1)[0]
    etf_idx = np.where(labels == 2)[0]
    combined_idx = np.concatenate([ours_idx, etf_idx])
    print("Proof组坐标范围:", embeddings[proof_idx].min(axis=0), embeddings[proof_idx].max(axis=0))
    print("Ours+ETF坐标范围:", embeddings[combined_idx].min(axis=0), embeddings[combined_idx].max(axis=0))

    for i in [1, 2]:  # 分别绘制Ours和ETF
        idx = np.where(labels == i)[0]
        ax2.scatter(
            embeddings[idx, 0], embeddings[idx, 1],
            c=VisualConfig.color_maps[i],
            marker=VisualConfig.markers[i],
            s=VisualConfig.sizes[i]*1.1,
            alpha=VisualConfig.alphas[i],
            linewidths=0.8
        )

    # 设置右图坐标范围
    x_min, x_max = embeddings[combined_idx, 0].min(), embeddings[combined_idx, 0].max()
    y_min, y_max = embeddings[combined_idx, 1].min(), embeddings[combined_idx, 1].max()
    ax2.set_xlim(x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min))
    ax2.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))

    # 添加右图聚焦框（关键修改点）
    draw_focus_boxes(ax2, box_type='right')  # 明确指定框类型

    # ---- 样式统一 ----
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_color('black')
        ax.spines[:].set_linewidth(2)
        ax.text(0.5, -0.08,
                f'({titles[0][0] if ax == ax1 else titles[1][0]}) {titles[0][1] if ax == ax1 else titles[1][1]}',
                transform=ax.transAxes, ha='center', va='top', fontsize=30,fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/qc/python_tool/prototype_tsne.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/qc/python_tool/result/prototype_tsne.pdf', format="pdf", dpi=300, bbox_inches='tight')
    plt.close()


def visualize_zoomed(embeddings, labels, titles):
    """放大视图可视化"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 样式设置
    for ax in [ax1, ax2]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines[:].set_color('black')
        ax.spines[:].set_linewidth(2)

    # 左子图：Proof组放大
    left_box = VisualConfig.zoom_boxes['left']
    proof_idx = np.where((labels == 0) &
                         (embeddings[:, 0] >= left_box[0]) &
                         (embeddings[:, 0] <= left_box[2]) &
                         (embeddings[:, 1] >= left_box[1]) &
                         (embeddings[:, 1] <= left_box[3]))[0]

    ax1.scatter(
        embeddings[proof_idx, 0], embeddings[proof_idx, 1],
        c=VisualConfig.color_maps[0],
        marker=VisualConfig.markers[0],
        s=VisualConfig.sizes[0] * 3.5,  # 放大后适当增大标记
        alpha=VisualConfig.alphas[0],
        edgecolors='w',
        linewidths=0.8
    )
    ax1.set_xlim(left_box[0], left_box[2])
    ax1.set_ylim(left_box[1], left_box[3])

    # 右子图：Ours + ETF放大
    right_box = VisualConfig.zoom_boxes['right']
    for i, color in enumerate(VisualConfig.color_maps[1:], 1):
        idx = np.where((labels == i) &
                       (embeddings[:, 0] >= right_box[0]) &
                       (embeddings[:, 0] <= right_box[2]) &
                       (embeddings[:, 1] >= right_box[1]) &
                       (embeddings[:, 1] <= right_box[3]))[0]

        ax2.scatter(
            embeddings[idx, 0], embeddings[idx, 1],
            c=color,
            marker=VisualConfig.markers[i],
            s=VisualConfig.sizes[i] * 3.5,  # 放大后适当增大标记
            alpha=VisualConfig.alphas[i],
            linewidths=0.8
        )
    ax2.set_xlim(right_box[0], right_box[2])
    ax2.set_ylim(right_box[1], right_box[3])

    # 添加标题
    ax1.text(0.5, -0.08, f'({titles[0][0]}) {titles[0][1]} (Zoomed)',
             transform=ax1.transAxes, ha='center', va='top', fontsize=30,fontweight='bold')
    ax2.text(0.5, -0.08, f'({titles[1][0]}) {titles[1][1]} (Zoomed)',
             transform=ax2.transAxes, ha='center', va='top', fontsize=30,fontweight='bold')

    plt.tight_layout()
    plt.savefig('/home/qc/python_tool/prototype_tsne_zoomed.png', dpi=300, bbox_inches='tight')
    plt.savefig("/home/qc/python_tool/result/prototype_tsne_zoomed.pdf", format="pdf", dpi=300)
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
    visualize_zoomed(embeddings, labels, titles=[('c', 'Proof'), ('d', 'Our Method')])


if __name__ == "__main__":
    main()