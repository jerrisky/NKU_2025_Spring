#include<iostream>
#include<Windows.h>
#include<stdlib.h>
using namespace std;
const int N =65536;

double a[N];
double sum;
void init(int n) {
    for (int i = 0; i < n; i++)
        a[i] = i + 1;
}
int main() {
    long long head, tail, freq;
    init(N);
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    for (int n = 1; n <= N; n*=2) {
        int counter = 0;
        double total_time = 0.0;
        while (total_time < 0.1) { // 累计时间超过0.1秒即可观察输出
            counter++;
            sum=0;
            QueryPerformanceCounter((LARGE_INTEGER*)&head);
            for(int i=0;i<n;i++)
                sum+=a[i];
            QueryPerformanceCounter((LARGE_INTEGER*)&tail);
            // 计算当前执行时间并累计
            total_time+=(tail - head) * 1.0 / freq;
        }
        cout << n << ' ' << counter << ' ' << total_time << ' ' << total_time / counter * 1000000000 << endl;
    }
    return 0;
}
