#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <cmath>
#include <numeric> // 添加numeric头文件用于计算平均值
// 矩阵和向量初始化（全1填充）
void init_data(double* matrix, double* vector, int n) {
    for (int i = 0; i < n * n; ++i) matrix[i] = 1.0;
    for (int i = 0; i < n; ++i) vector[i] = 1.0;
}
// 矩阵列向量内积 - 平凡算法
void naive_matrix_vector_product(const double* matrix, const double* vector, double* result, int n) {
    for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) {
            sum += matrix[i * n + j] * vector[i]; // 行优先存储
        }
        result[j] = sum;
    }
}

// 矩阵列向量内积 - Cache优化算法
void optimized_matrix_vector_product(const double* matrix, const double* vector, double* result, int n) {
    memset(result, 0, n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        double v = vector[i];
        for (int j = 0; j < n; ++j) {
            result[j] += matrix[i * n + j] * v; // 按行遍历
        }
    }
}
// 求和 - 平凡算法
double naive_sum(const double* a, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += a[i];
    }
    return sum;
}

// 求和 - 超标量优化（两路展开）
double optimized_sum(const double* a, int n) {
    double sum0 = 0.0, sum1 = 0.0;
    int i;
    for (i = 0; i < n - 1; i += 2) {
        sum0 += a[i];
        sum1 += a[i + 1];
    }
    for (; i < n; ++i) sum0 += a[i];
    return sum0 + sum1;
}

// 循环展开（四路展开）
double unrolled_sum(const double* a, int n) {
    double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
    int i = 0;
    for (; i <= n - 4; i += 4) {
        sum0 += a[i];
        sum1 += a[i + 1];
        sum2 += a[i + 2];
        sum3 += a[i + 3];
    }
    for (; i < n; ++i) sum0 += a[i];
    return sum0 + sum1 + sum2 + sum3;
}
// 性能测试模板函数
template<typename Func, typename... Args>
double benchmark(Func func, int warmup, int runs, Args... args) {
    // 预热阶段
    for (int i = 0; i < warmup; ++i) {
        func(args...);
    }

    // 正式计时
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < runs; ++i) {
        func(args...);
    }
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / static_cast<double>(runs);
}

// 正确性验证（允许浮点误差）
bool verify_result(const double* result, int n, double expected) {
    const double eps = 1e-6;
    for (int i = 0; i < n; ++i) {
        if (std::abs(result[i] - expected) > eps) {
            std::cerr << "Validation failed! Expected: " << expected
                << ", Got: " << result[i] << " at index " << i << "\n";
            return false;
        }
    }
    return true;
}


int main() {
    const std::vector<int> test_sizes = { 256, 512, 1024, 2048 };
    const int warmup = 5;
    const int runs = 100;
    const int group_runs = 50; 

    std::cout << std::left << std::setw(10) << "Size"
        << std::setw(8) << "Run"
        << std::setw(25) << "Naive Matrix (μs)"
        << std::setw(25) << "Optimized Matrix (μs)"
        << "\n";

    for (int n : test_sizes) {
        double* matrix = new double[n * n];
        double* vector = new double[n];
        double* result_mat = new double[n];

        // 存储每组运行结果
        std::vector<double> naive_results;
        std::vector<double> optimized_results;
        for (int group = 0; group < group_runs; ++group) {
            // 每次组运行前重新初始化数据
            init_data(matrix, vector, n);

            // 测试平凡算法
            double t_naive = benchmark(naive_matrix_vector_product, warmup, runs,
                matrix, vector, result_mat, n);
            verify_result(result_mat, n, n);
            naive_results.push_back(t_naive);

            // 测试优化算法
            init_data(matrix, vector, n); // 重置数据
            double t_optimized = benchmark(optimized_matrix_vector_product, warmup, runs,
                matrix, vector, result_mat, n);
            verify_result(result_mat, n, n);
            optimized_results.push_back(t_optimized);



            // 输出单次运行结果
            std::cout << std::setw(10) << n
                << std::setw(8) << group + 1
                << std::setw(25) << t_naive
                << std::setw(25) << t_optimized
                << "\n";
        }

        // 计算并输出平均值
        double naive_avg = std::accumulate(naive_results.begin(), naive_results.end(), 0.0) / group_runs;
        double optimized_avg = std::accumulate(optimized_results.begin(), optimized_results.end(), 0.0) / group_runs;

        std::cout << std::setw(10) << n
            << std::setw(8) << "Avg"
            << std::setw(25) << naive_avg
            << std::setw(25) << optimized_avg
            << "\n\n";

        delete[] matrix;
        delete[] vector;
        delete[] result_mat;
    }
    return 0;
}