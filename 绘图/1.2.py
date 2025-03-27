import re
import matplotlib.pyplot as plt
from collections import defaultdict


def parse_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    data = defaultdict(lambda: defaultdict(list))

    # 修正后的正则表达式（允许连字符）
    pattern = re.compile(
        r'^(\d+)\s+(\d+)\s+([\w-]+)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+\?'
    )

    for line in lines:
        if line.startswith(('Global Stats', 'ize', '-------')):
            continue
        match = pattern.match(line)
        if match:
            try:
                size = int(match.group(1))
                algorithm = match.group(3)
                avg = float(match.group(6))
                data[size][algorithm].append(avg)
            except (ValueError, IndexError) as e:
                print(f"解析错误行：{line}，错误信息：{str(e)}")

    return data


# 后续函数与之前相同...


# 处理数据并生成统计信息
def process_data(data):
    processed = {}
    for size in sorted(data.keys()):
        processed[size] = {}
        for alg in data[size]:
            processed[size][alg] = {
                'avg': sum(data[size][alg]) / len(data[size][alg]),
                'min': min(data[size][alg]),
                'max': max(data[size][alg])
            }
    return processed


# 绘制对比图表
def plot_comparison(data):
    sizes = sorted(data.keys())
    algorithms = ['Naive', '2-Way', '4-Way']
    colors = ['#1f77b4', '#2ca02c', '#d62728']

    plt.figure(figsize=(12, 8))

    # 绘制平均线
    for i, alg in enumerate(algorithms):
        avg_times = [data[size][alg]['avg'] for size in sizes]
        plt.plot(sizes, avg_times, marker='o', color=colors[i],
                 label=f'{alg} (Avg)', linewidth=2)

        # 绘制误差范围
        min_times = [data[size][alg]['min'] for size in sizes]
        max_times = [data[size][alg]['max'] for size in sizes]
        plt.fill_between(sizes, min_times, max_times,
                         color=colors[i], alpha=0.2)

    # 美化图表
    plt.title('Algorithm Performance Comparison', fontsize=14)
    plt.xlabel('Data Size', fontsize=12)
    plt.ylabel('Time (μs)', fontsize=12)
    plt.xticks(sizes, ['1e6', '5e6', '1e7'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 主函数
if __name__ == '__main__':
    file_path = '1.2数据.txt'  # 请确保文件路径正确
    raw_data = parse_data(file_path)
    processed_data = process_data(raw_data)
    plot_comparison(processed_data)