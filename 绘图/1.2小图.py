import re
import matplotlib.pyplot as plt

# 初始化数据存储
data = {}

# 读取并解析数据
with open('1.2数据.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

# 找到标题行位置
header_index = lines.index('ize        Group     Algorithm      Min(μs)       Max(μs)       Avg(μs)       Valid')

# 分割数据块（跳过分隔线和Global Stats）
blocks = []
current_block = []
for line in lines[header_index + 1:]:
    if line.startswith('Global Stats'):
        if current_block:
            blocks.append(current_block)
        current_block = []
    elif not line.startswith('-'):  # 跳过分隔线
        current_block.append(line)
if current_block:
    blocks.append(current_block)

for block in blocks:
    if not block:
        continue

    # 提取Size（第一行的第一个数字）
    try:
        size = int(block[0].split()[0])
    except (IndexError, ValueError) as e:
        print(f"解析Size时出错，块内容: {block[:3]}, 错误信息: {e}")
        continue

    # 初始化存储
    data[size] = {'Naive': [], '2-Way': [], '4-Way': []}

    for line in block:
        # 调试输出：打印原始行内容
        print(f"解析行: {line}")

        # 处理可能的多空格分隔
        parts = re.split(r'\s{2,}', line)
        print(f"分割后字段: {parts}")

        if len(parts) >= 7:
            try:
                algorithm = parts[2]
                avg = float(parts[5])
                data[size][algorithm].append(avg)
                print(f"成功解析: {algorithm} {avg}")
            except (IndexError, ValueError) as e:
                print(f"解析错误: {line}, 错误信息: {e}")
        else:
            print(f"跳过无效行: {line}")

# 调试输出（检查数据是否正确解析）
print("\n最终解析到的数据:")
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