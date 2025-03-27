import re
import matplotlib.pyplot as plt

# 初始化数据存储
data = {}

# 读取并解析数据
with open('1.2数据.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip() and not line.startswith(('ize', '-', 'Global Stats'))]

for line in lines:
    # 使用正则表达式匹配字段（支持带连字符的算法名称）
    match = re.match(r'^(\d+)\s+\d+\s+([\w-]+)\s+\d+\.?\d*\s+\d+\.?\d*\s+(\d+\.?\d*)\s+\?', line)
    if match:
        size = int(match.group(1))
        algorithm = match.group(2)
        avg = float(match.group(3))
        if size not in data:
            data[size] = {'Naive': [], '2-Way': [], '4-Way': []}
        data[size][algorithm].append(avg)
    else:
        print(f"无法解析的行: {line}")

# 调试输出（检查数据是否正确解析）
print("解析到的数据:")
for size, alg_data in data.items():
    print(f"Size: {size}")
    for alg, times in alg_data.items():
        print(f"  {alg}: {len(times)} samples")
    print()

# 处理数据
sizes = sorted(data.keys())
algorithms = ['Naive', '2-Way', '4-Way']
markers = ['o', 's', '^']
colors = ['#1f77b4', '#2ca02c', '#d62728']

# 计算子图布局
num_sizes = len(sizes)
nrows = 2
ncols = (num_sizes + 1) // 2

# 创建子图
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 8))

# 处理单图情况
if num_sizes == 1:
    axes = [[axes]]

for i, size in enumerate(sizes):
    row = i // ncols
    col = i % ncols
    ax = axes[row][col]

    for j, alg in enumerate(algorithms):
        times = data[size][alg]
        if not times:
            print(f"警告：Size {size} 的 {alg} 算法没有数据")
            continue
        ax.plot(range(1, len(times) + 1), times, marker=markers[j], linestyle='-',
                linewidth=2, color=colors[j], label=f'{alg}')

        # 添加数据标签
        for k, y in enumerate(times, start=1):
            ax.text(k, y, f'{y:.1f}', ha='center', va='bottom' if j == 0 else 'top',
                    fontsize=6, color=colors[j])

    ax.set_title(f'Size: {size}')
    ax.set_xlabel('Run Number')
    ax.set_ylabel('Execution Time (μs)')
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    if i == 0:
        ax.legend(fontsize=8)

plt.tight_layout()
plt.show()