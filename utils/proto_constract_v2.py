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


def visualize_combined(embeddings, labels, titles):
    """合并原图和放大图为两行布局"""
    fig, axs = plt.subplots(2, 2, figsize=(22, 16))  # 2行2列布局
    plt.subplots_adjust(wspace=0.15, hspace=0.25)  # 调整子图间距

    # ===================== 第一行：原图 =====================
    # ---- 左子图：Proof组 ----
    proof_idx = np.where(labels == 0)[0]
    axs[0, 0].scatter(
        embeddings[proof_idx, 0], embeddings[proof_idx, 1],
        c=VisualConfig.color_maps[0],
        marker=VisualConfig.markers[0],
        s=VisualConfig.sizes[0] * 1.1,
        alpha=VisualConfig.alphas[0],
        edgecolors='w',
        linewidths=0.8
    )
    x_min, x_max = embeddings[proof_idx, 0].min(), embeddings[proof_idx, 0].max()
    y_min, y_max = embeddings[proof_idx, 1].min(), embeddings[proof_idx, 1].max()
    axs[0, 0].set_xlim(x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min))
    axs[0, 0].set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    draw_focus_boxes(axs[0, 0], box_type='left')

    # ---- 右子图：Ours + ETF ----
    for i in [1, 2]:
        idx = np.where(labels == i)[0]
        axs[0, 1].scatter(
            embeddings[idx, 0], embeddings[idx, 1],
            c=VisualConfig.color_maps[i],
            marker=VisualConfig.markers[i],
            s=VisualConfig.sizes[i] * 1.1,
            alpha=VisualConfig.alphas[i],
            linewidths=0.8
        )
    combined_idx = np.concatenate([np.where(labels == 1)[0], np.where(labels == 2)[0]])
    x_min, x_max = embeddings[combined_idx, 0].min(), embeddings[combined_idx, 0].max()
    y_min, y_max = embeddings[combined_idx, 1].min(), embeddings[combined_idx, 1].max()
    axs[0, 1].set_xlim(x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min))
    axs[0, 1].set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
    draw_focus_boxes(axs[0, 1], box_type='right')

    # ===================== 第二行：放大图 =====================
    # ---- 左子图：Proof组放大 ----
    left_box = VisualConfig.zoom_boxes['left']
    proof_zoom_idx = np.where((labels == 0) &
                              (embeddings[:, 0] >= left_box[0]) &
                              (embeddings[:, 0] <= left_box[2]) &
                              (embeddings[:, 1] >= left_box[1]) &
                              (embeddings[:, 1] <= left_box[3]))[0]
    axs[1, 0].scatter(
        embeddings[proof_zoom_idx, 0], embeddings[proof_zoom_idx, 1],
        c=VisualConfig.color_maps[0],
        marker=VisualConfig.markers[0],
        s=VisualConfig.sizes[0] * 3.5,
        alpha=VisualConfig.alphas[0],
        edgecolors='w',
        linewidths=0.8
    )
    axs[1, 0].set_xlim(left_box[0], left_box[2])
    axs[1, 0].set_ylim(left_box[1], left_box[3])

    # ---- 右子图：Ours+ETF放大 ----
    right_box = VisualConfig.zoom_boxes['right']
    for i in [1, 2]:
        idx = np.where((labels == i) &
                       (embeddings[:, 0] >= right_box[0]) &
                       (embeddings[:, 0] <= right_box[2]) &
                       (embeddings[:, 1] >= right_box[1]) &
                       (embeddings[:, 1] <= right_box[3]))[0]
        axs[1, 1].scatter(
            embeddings[idx, 0], embeddings[idx, 1],
            c=VisualConfig.color_maps[i],
            marker=VisualConfig.markers[i],
            s=VisualConfig.sizes[i] * 3.5,
            alpha=VisualConfig.alphas[i],
            linewidths=0.8
        )
    axs[1, 1].set_xlim(right_box[0], right_box[2])
    axs[1, 1].set_ylim(right_box[1], right_box[3])

    # ===================== 统一样式与标题 =====================
    # 定义标题标签顺序
    position_labels = [
        [('a', 'Proof'), ('b', 'Our Method')],
        [('c', 'Proof (Zoomed)'), ('d', 'Our Method (Zoomed)')]
    ]

    for i in range(2):
        for j in range(2):
            ax = axs[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines[:].set_color('black')
            ax.spines[:].set_linewidth(2)

            # 根据行列索引获取标签
            label, text = position_labels[i][j]
            ax.text(0.5, -0.04,
                    f'({label}) {text}',
                    transform=ax.transAxes,
                    ha='center',
                    va='top',
                    fontsize=28,
                    fontweight='bold')
    # ===================== 添加连接线 =====================
    from matplotlib.patches import ConnectionPatch

    # 左图连接参数
    left_box = VisualConfig.zoom_boxes['left']
    conn_params_left = {
        "linestyle": ":",  # 虚线样式
        "linewidth": 3.5,  # 线宽
        "color": "dimgray",  # 高级灰
        "arrowstyle": "->,head_width=0.4,head_length=0.8",  # 箭头样式
        "mutation_scale": 20  # 箭头大小
    }

    # 绘制左图两条连接线
    for x_pos in [left_box[0], left_box[2]]:  # 左框的左右x坐标
        connection = ConnectionPatch(
            xyA=(x_pos, left_box[1]),  # 原图聚焦框下边缘
            xyB=(x_pos, left_box[3]),  # 放大图上边缘
            coordsA=axs[0, 0].transData,  # 原图坐标系
            coordsB=axs[1, 0].transData,  # 放大图坐标系
         ** conn_params_left
        )
        fig.add_artist(connection)

    # 右图连接参数
    right_box = VisualConfig.zoom_boxes['right']
    conn_params_right = conn_params_left.copy()
    conn_params_right.update({"color": "dimgray"})  # 右图使用不同颜色

    # 绘制右图两条连接线
    for x_pos in [right_box[0], right_box[2]]:  # 右框的左右x坐标
        connection = ConnectionPatch(
            xyA=(x_pos, right_box[1]),  # 原图聚焦框下边缘
            xyB=(x_pos, right_box[3]),  # 放大图上边缘
            coordsA=axs[0, 1].transData,
            coordsB=axs[1, 1].transData,
         ** conn_params_right
        )
        fig.add_artist(connection)

    # ===================== 图例配置 =====================
    legend_elements = [
        plt.Line2D([0], [0],
                   marker='o',
                   color='w',
                   label='Prototype',
                   markerfacecolor='red',
                   markersize=18,
                   markeredgecolor='red',
                   markeredgewidth=0),
        plt.Line2D([0], [0],
                   marker='*',
                   color='w',
                   label='ETF Target',
                   markerfacecolor='black',
                   markersize=22,
                   markeredgewidth=0)
    ]

    # 通用图例参数
    legend_style = {
        "loc": 'upper left',
        "bbox_to_anchor": (0.010, 0.990),  # 紧贴左上角 (0.5%边距)
        "frameon": True,
        "fancybox": False,  # 直角边框
        "edgecolor": 'black',  # 边框颜色
        "facecolor": 'white',  # 背景色
        "framealpha": 1.0,  # 不透明
        "borderpad": 0.3,  # 边框内边距
        "borderaxespad": 0.2,  # 边框与坐标轴间距
        "handletextpad": 0.5,  # 图标文字间距
        "fontsize": 20,
        "labelspacing": 0.4  # 标签间距
    }

    # 左列子图配置（仅Prototype）
    for row in [0, 1]:
        axs[row, 0].legend(
            handles=[legend_elements[0]],
         ** legend_style
        )

    # 右列子图配置（Prototype + ETF）
    for row in [0, 1]:
        axs[row, 1].legend(
            handles=legend_elements,
         ** legend_style
        )

    plt.tight_layout()
    plt.savefig('/home/qc/python_tool/result/prototype_tsne.png', dpi=300, bbox_inches='tight')
    plt.savefig('/home/qc/python_tool/result/prototype_tsne.pdf', format='pdf', dpi=300)
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

    # 使用合并后的可视化函数
    visualize_combined(embeddings, labels, titles=[('a', 'Proof'), ('b', 'Our Method')])


if __name__ == "__main__":
    main()