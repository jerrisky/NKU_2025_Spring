#include<iostream>
#include<Windows.h>
#include<stdlib.h>
using namespace std;
const int N = 65536;
double a[N];

void init(int n) {
    for (int i = 0; i < n; i++)
        a[i] = i + 1;
}

int main() {
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    for (int n = 1; n <= N; n *= 2) {
        int counter = 0;
        double total_time = 0.0;
        while (total_time < 0.1) { // 累计时间超过0.1秒即可观察输出
            counter++;
            init(n);
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            for (int m = n; m > 1; m /= 2) { // log(n)个步骤
                int half_m = m / 2;
                // 展开因子设为4
                int i;
                for ( i = 0; i <= half_m - 4; i += 4) {
                    a[i] = a[2*i] + a[2*i + 1];
                    a[i+1] = a[2*(i+1)] + a[2*(i+1)+1];
                    a[i+2] = a[2*(i+2)] + a[2*(i+2)+1];
                    a[i+3] = a[2*(i+3)] + a[2*(i+3)+1];
                }
                // 处理剩余元素
                for (; i < half_m; i++) {
                    a[i] = a[2*i] + a[2*i + 1];
                }
            }
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            total_time += (tail - head) * 1.0 / freq;
        }
        cout << n << ' ' << counter << ' ' << total_time << ' '
             << total_time / counter * 1000000000 <<' '<<a[0]<<endl;
    }
    return 0;
}
