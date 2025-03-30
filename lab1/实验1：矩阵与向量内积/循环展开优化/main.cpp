#include<iostream>
#include<Windows.h>
#include<stdlib.h>
using namespace std;
const int N = 10001;
double b[N][N], sum[N];
double a[N];

void init(int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            b[i][j] = i + j;
    for (int i = 0; i < n; i++)
        a[i] = i + 1;
}

int main() {
    long long head, tail, freq;
    init(N);
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    int step = 10;
    for (int n = 20; n < N; n += step) {
        int counter = 0;
        double total_time = 0.0;
        QueryPerformanceCounter((LARGE_INTEGER*)&head);
        while (total_time < 1.0) { // 计时确保运行1秒左右
            counter++;
            for (int i = 0; i < n; i++)
                sum[i] = 0.0;

            // 优化后的双重循环（循环展开）
            for (int j = 0; j < n; j++) {
                double aj = a[j]; // 提前提取a[j]，减少重复访问
                // 展开因子4，处理批量元素
                int k;
                for (k = 0; k + 3 < n; k += 4) {
                    sum[k] += b[j][k] * aj;
                    sum[k + 1] += b[j][k + 1] * aj;
                    sum[k + 2] += b[j][k + 2] * aj;
                    sum[k + 3] += b[j][k + 3] * aj;
                }
                // 处理剩余不足4个的元素
                for (; k < n; k++) {
                    sum[k] += b[j][k] * aj;
                }
            }

            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            total_time = (tail - head) * 1.0 / freq;
        }
        cout << n << ' ' << counter << ' ' << total_time << ' ' << total_time / counter * 1000 << endl;
        if (n == 100) step = 100;
        if (n == 1000) step = 1000;
    }
    return 0;
}
