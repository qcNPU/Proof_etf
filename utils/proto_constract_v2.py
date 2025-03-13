import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ===================== 配置参数 =====================
class VisualConfig:
    # 颜色和形状配置
    colors = ['#FF1F5B', '#009ADE', '#00CD6C']  # 红，蓝，绿
    markers = ['o', 's', '*']  # 圆形，方形，三角形
    sizes = [100, 100, 120]  # 标记尺寸
    alphas = [1, 1, 1]  # 透明度
    labels = ['Proof', 'Ours', 'ETF']  # 图例标签

    # t-SNE参数
    tsne_params = {
        'n_components': 2,
        'perplexity': 30,
        'n_iter': 2000,
        'learning_rate': 300,
        'metric': 'cosine',
        # 'metric' :"euclidean",
        'random_state': 42
    }


# ===================== 数据加载与处理 =====================
def load_and_preprocess():
    """加载并合并三组Prototype数据"""
    # 加载原始数据
    proof = torch.load("proof.pth").view(100, -1).cpu().numpy()
    # ours = torch.load("proofnc.pth").view(100, -1).cpu().numpy()
    # etf = torch.load("proofnc_etf.pth").view(100, -1).cpu().numpy()
    ours = torch.load("proofncscmp.pth").view(100, -1,512)[:,0,:].cpu().numpy()
    etf = torch.load("proofncscmp_etf.pth").view(100, -1).cpu().numpy()

    # 合并数据并创建标签
    combined = np.concatenate([proof, ours, etf])
    labels = np.array([0] * 100 + [1] * 100 + [2] * 100)  # 0:proof, 1:ours, 2:etf

    # 标准化处理
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(combined), labels


# ===================== 可视化引擎 =====================
def visualize_prototypes(embeddings, labels):
    """专业级可视化布局"""
    plt.figure(figsize=(15, 10))

    # 绘制三组数据
    for i in range(3):
        mask = (labels == i)
        plt.scatter(
            embeddings[mask, 0],
            embeddings[mask, 1],
            c=VisualConfig.colors[i],
            marker=VisualConfig.markers[i],
            s=VisualConfig.sizes[i],
            alpha=VisualConfig.alphas[i],
            edgecolors='w',
            linewidths=0.8,
            label=VisualConfig.labels[i]
        )

    # 高级图例配置
    legend = plt.legend(
        title='Prototype Groups',
        title_fontsize=12,
        fontsize=10,
        loc='upper right',  # 图例位置
        bbox_to_anchor=(0.98, 0.98),  # 微调锚点
        frameon=True,
        framealpha=0.9,  # 背景透明度
    )
    legend.get_frame().set_edgecolor('black')  # 边框颜色
    legend.get_frame().set_linewidth(0.5)  # 边框宽度

    # 坐标轴美化
    # plt.xlabel("t-SNE 1", fontsize=12)
    # plt.ylabel("t-SNE 2", fontsize=12)
    # plt.title("Prototype Distribution Comparison", fontsize=14, pad=20)
    plt.grid(alpha=0.2)

    # 输出设置
    plt.tight_layout()
    plt.savefig('prototype_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig("prototype_comparison.pdf", format="pdf", bbox_inches='tight', dpi=300)
    plt.show()


# ===================== 主流程 =====================
def main():
    # 数据准备
    combined_data, labels = load_and_preprocess()

    # 执行t-SNE
    print("Running t-SNE...")
    tsne = TSNE(**VisualConfig.tsne_params)
    embeddings = tsne.fit_transform(combined_data)

    # 可视化
    visualize_prototypes(embeddings, labels)


if __name__ == "__main__":
    main()