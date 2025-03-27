import matplotlib.pyplot as plt

# 读取数据并提取平均值
sizes = []
naive_avg = []
optimized_avg = []

with open('1.1数据.txt', 'r', encoding='utf-8') as f:
    content = f.read().strip()
    blocks = content.split('\n\n')  # 按空行分割不同Size的数据块

for block in blocks:
    lines = block.split('\n')
    header = lines[0]  # 标题行（忽略）
    avg_line = lines[-1]  # 最后一行是平均值数据
    parts = avg_line.split()
    sizes.append(int(parts[0]))
    naive_avg.append(float(parts[2]))
    optimized_avg.append(float(parts[3]))

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(sizes, naive_avg, marker='o', linestyle='-', linewidth=2, label='Naive Matrix (μs)')
plt.plot(sizes, optimized_avg, marker='s', linestyle='-', linewidth=2, label='Optimized Matrix (μs)')

# 设置坐标轴和标题
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Matrix Size (N x N)', fontsize=12)
plt.ylabel('Execution Time (μs)', fontsize=12)
plt.title('Performance Comparison: Naive vs Optimized Matrix Multiplication', fontsize=14)
plt.xticks(sizes, labels=[str(size) for size in sizes], fontsize=10)
plt.yticks(fontsize=10)

# 添加网格和图例
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# 显示数据标签
for x, y1, y2 in zip(sizes, naive_avg, optimized_avg):
    plt.text(x, y1, f'{y1:.1f}', ha='center', va='bottom', fontsize=8)
    plt.text(x, y2, f'{y2:.1f}', ha='center', va='top', fontsize=8)

plt.tight_layout()
plt.show()