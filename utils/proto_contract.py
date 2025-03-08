import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

# ===================== 数据生成 =====================
# 假设两种方法生成的高维prototype数据


def visualize_proto_contract(method1_prototypes,method2_prototypes,prototype_dim = 512):
    # 降维
    tsne = TSNE(n_components=2, perplexity=5)
    projected_method1 = tsne.fit_transform(method1_prototypes)
    projected_method2 = tsne.fit_transform(method2_prototypes)

    # 构建DataFrame
    df = pd.DataFrame({
        "x": np.concatenate([projected_method1[:,0], projected_method2[:,0]]),
        "y": np.concatenate([projected_method1[:,1], projected_method2[:,1]]),
        "Method": ["Prior Method"]*20 + ["Our Method"]*20,
        "Class": [f"Class_{i}" for i in range(20)] * 2
    })

    # ===================== 可视化 =====================
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x="x", y="y", hue="Method", style="Method",
                    palette={"Prior Method": "red", "Our Method": "blue"},
                    markers={"Prior Method": "o", "Our Method": "s"},
                    s=80, edgecolor='k')

    # 标注统计信息
    prior_intra_dist = np.mean([np.linalg.norm(method1_prototypes[i]-method1_prototypes[j])
                               for i in range(20) for j in range(i+1, 20)])
    our_inter_dist = np.mean([np.linalg.norm(method2_prototypes[i]-method2_prototypes[j])
                             for i in range(20) for j in range(20) if i != j])

    plt.text(0.05, 0.95,
             f"Prior Method Intra-Dist: {prior_intra_dist:.2f}\nOur Method Inter-Dist: {our_inter_dist:.2f}",
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.8))

    plt.title("Prototype Distribution Comparison (t-SNE Projection)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("proto_contract_tsne.png", dpi=300)
    plt.close()

if __name__ == '__main__':
    prototype_dim = 512  # 假设原型维度为512
    method1_prototypes = np.random.randn(20, prototype_dim)  # 20个类
    method2_prototypes = np.random.randn(20, prototype_dim)
    visualize_proto_contract(method1_prototypes,method2_prototypes)