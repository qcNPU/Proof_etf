import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ===================== 数据准备 =====================
theta = np.linspace(0, 2 * np.pi, 10, endpoint=False)
r_old = np.array([1.5, 3, 2, 1.0, 4, 1, 0.5, 0.0, 0.5, 1])  # 旧方法数据
r_new = np.full(theta.shape, 4)  # 新方法全部延伸到圆周
colors = plt.cm.tab10(np.arange(10))  # 使用tab10颜色映射

# ===================== 可视化设置 =====================
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# 绘制带箭头的径向线（旧方法虚线，新方法实线）
for i in range(len(theta)):
    # 旧方法（虚线）
    arrow = FancyArrowPatch(
        (0, 0),  # 起点（圆心）
        (theta[i], r_old[i]),  # 终点（极坐标）
        arrowstyle='->',
        linestyle='--',
        color=colors[i],
        linewidth=1,
        mutation_scale=15,  # 调整箭头大小
        transform=ax.transData,  # 使用极坐标变换
    )
    ax.add_patch(arrow)

    # 新方法（实线）
    arrow = FancyArrowPatch(
        (0, 0),  # 起点（圆心）
        (theta[i], r_new[i]),  # 终点（极坐标）
        arrowstyle='->',
        linestyle='-',
        color=colors[i],
        linewidth=1,
        mutation_scale=15,
        transform=ax.transData,
    )
    ax.add_patch(arrow)

# 添加圆周上的星星标记
ax.scatter(theta, r_new,
           marker='*',
           s=80,  # 星星大小
           color='gold',
           edgecolor='black',
           zorder=4,
           linewidths=1)

# ===================== 样式优化 =====================
ax.set_xticklabels([])  # 移除角度刻度标签
ax.set_theta_offset(np.pi / 2)  # 0度起始位置在顶部
ax.set_theta_direction(-1)  # 顺时针方向
ax.set_rgrids([0, 1, 2, 3, 4], angle=0)  # 设置径向刻度
ax.grid(alpha=0.3)  # 网格透明度
ax.set_facecolor('white')  # 背景颜色

plt.savefig('enhanced_radial_plot.png', dpi=300, bbox_inches='tight')
plt.show()