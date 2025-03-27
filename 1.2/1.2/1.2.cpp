#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <numeric>

// ================= 求和算法实现 =================
double naive_sum(const double* a, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += a[i];
    return sum;
}

double optimized_sum(const double* a, int n) {
    double sum0 = 0.0, sum1 = 0.0;
    int i = 0;
    for (; i < n - 1; i += 2) {
        sum0 += a[i];
        sum1 += a[i + 1];
    }
    for (; i < n; ++i) sum0 += a[i];
    return sum0 + sum1;
}

double unrolled_sum(const double* a, int n) {
    double sum[4] = { 0 };
    int i = 0;
    for (; i <= n - 4; i += 4) {
        sum[0] += a[i];
        sum[1] += a[i + 1];
        sum[2] += a[i + 2];
        sum[3] += a[i + 3];
    }
    for (; i < n; ++i) sum[0] += a[i];
    return sum[0] + sum[1] + sum[2] + sum[3];
}

// ================= 性能测试框架 =================
struct TestGroupResult {
    std::vector<double> group_times; // 每组测试的多次运行时间（微秒）
};

template<typename Func, typename... Args>
TestGroupResult run_test_group(Func func, int warmup, int runs, Args... args) {
    TestGroupResult result;

    // 预热阶段
    for (int i = 0; i < warmup; ++i) func(args...);

    // 正式测试并记录每次运行时间
    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func(args...);
        auto end = std::chrono::high_resolution_clock::now();
        double time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        result.group_times.push_back(time_us);
    }

    return result;
}

// ================= 主测试逻辑 =================
void run_enhanced_sum_benchmark() {
    // 测试参数配置
    const std::vector<int> test_sizes = { 1000000, 5000000, 10000000 };
    const int warmup = 5;
    const int runs_per_group = 50;    // 每组运行次数
    const int groups_per_size = 50;    // 每个规模测试组数

    // 输出表头
    std::cout << std::left
        << std::setw(12) << "Size"
        << std::setw(10) << "Run"
        << std::setw(20) << "Naive Sum (μs)"
        << std::setw(20) << "Optimized Sum (μs)"
        << std::setw(20) << "Unrolled Sum (μs)"
        << "\n" << std::string(84, '-') << "\n";

    for (int n : test_sizes) {
        double* data = new double[n];
        std::fill_n(data, n, 1.0);
        const double expected = n * 1.0;

        std::vector<double> naive_avg(groups_per_size);
        std::vector<double> optimized_avg(groups_per_size);
        std::vector<double> unrolled_avg(groups_per_size);

        for (int group = 0; group < groups_per_size; ++group) {
            // 运行Naive算法测试
            auto naive_result = run_test_group(naive_sum, warmup, runs_per_group, data, n);
            naive_avg[group] = std::accumulate(naive_result.group_times.begin(), naive_result.group_times.end(), 0.0)
                / runs_per_group;

            // 运行Optimized算法测试
            auto optimized_result = run_test_group(optimized_sum, warmup, runs_per_group, data, n);
            optimized_avg[group] = std::accumulate(optimized_result.group_times.begin(), optimized_result.group_times.end(), 0.0)
                / runs_per_group;

            // 运行Unrolled算法测试
            auto unrolled_result = run_test_group(unrolled_sum, warmup, runs_per_group, data, n);
            unrolled_avg[group] = std::accumulate(unrolled_result.group_times.begin(), unrolled_result.group_times.end(), 0.0)
                / runs_per_group;
        }

        // 输出每组结果
        for (int group = 0; group < groups_per_size; ++group) {
            std::cout << std::setw(12) << n
                << std::setw(10) << group + 1
                << std::fixed << std::setprecision(2)
                << std::setw(20) << naive_avg[group]
                << std::setw(20) << optimized_avg[group]
                    << std::setw(20) << unrolled_avg[group]
                        << "\n";
        }

        // 计算平均值
        double naive_total = std::accumulate(naive_avg.begin(), naive_avg.end(), 0.0) / groups_per_size;
        double optimized_total = std::accumulate(optimized_avg.begin(), optimized_avg.end(), 0.0) / groups_per_size;
        double unrolled_total = std::accumulate(unrolled_avg.begin(), unrolled_avg.end(), 0.0) / groups_per_size;

        // 输出平均值行
        std::cout << std::setw(12) << n
            << std::setw(10) << "Avg"
            << std::fixed << std::setprecision(5)
            << std::setw(20) << naive_total
            << std::setw(20) << optimized_total
            << std::setw(20) << unrolled_total
            << "\n\n";

        delete[] data;
    }
}

int main() {
    run_enhanced_sum_benchmark();
    return 0;
}