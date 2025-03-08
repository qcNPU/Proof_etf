import numpy as np
import matplotlib.pyplot as plt

# ===================== 配置参数 =====================
num_class = 10  # 与图中10个数字标签对应
prior_length = 0.7  # 短箭头长度（对应红色）
our_length = 0.9  # 长箭头长度（对应蓝色）
circle_radius = 1.0  # 圆周半径
text_offset = 0.1  # 类别标签偏移量
arrow_colors = {'Prior': '#D62728', 'Our': '#1F77B4'}  # 红蓝配色

# ===================== 数据生成 =====================
theta = np.linspace(0, 2 * np.pi, num_class, endpoint=False)
etf_points = np.array([np.cos(theta), np.sin(theta)]).T * circle_radius

# ===================== 可视化 =====================
fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')  # 强制白底

# ---- 绘制圆形基底 ----
ax.add_artist(plt.Circle((0, 0), circle_radius, fill=False, color='gray', linewidth=1))

# ---- 绘制双组箭头 ----
for i in range(num_class):
    # 红色短箭头（实线+虚线延伸）
    ax.plot([0, prior_length * etf_points[i, 0]],
            [0, prior_length * etf_points[i, 1]],
            color=arrow_colors['Prior'], linewidth=2)
    ax.plot([prior_length * etf_points[i, 0], etf_points[i, 0]],
            [prior_length * etf_points[i, 1], etf_points[i, 1]],
            '--', color=arrow_colors['Prior'], linewidth=1)

    # 蓝色长箭头（实线+虚线延伸）
    ax.plot([0, our_length * etf_points[i, 0]],
            [0, our_length * etf_points[i, 1]],
            color=arrow_colors['Our'], linewidth=2)
    ax.plot([our_length * etf_points[i, 0], etf_points[i, 0]],
            [our_length * etf_points[i, 1], etf_points[i, 1]],
            '--', color=arrow_colors['Our'], linewidth=1)

# ---- 添加类别标签 ----
for i, (x, y) in enumerate(etf_points):
    angle = np.degrees(theta[i])
    # 智能调整标签位置防止重叠
    ha = 'left' if (angle > 150 or angle < 30) else 'right' if angle > 210 else 'center'
    va = 'bottom' if 10 < angle < 170 else 'top' if 190 < angle < 350 else 'center'
    ax.text(x * (1 + text_offset), y * (1 + text_offset), str(i),
            ha=ha, va=va, fontsize=12, color='black')

# ---- 隐藏所有坐标元素 ----
ax.set_xticks([])
ax.set_yticks([])
ax.spines[:].set_visible(False)
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.set_aspect('equal')

plt.savefig("white_background_prototype.png", dpi=300, bbox_inches='tight')
plt.close()