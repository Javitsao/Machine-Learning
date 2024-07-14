#include<bits/stdc++.h>
using namespace std;
#define N 256
#define n 10
int find_mistake(int i, double w[], double x[][n + 1], int y[]) {
    double sum = 0;
    for (int j = 0; j <= n; j++) {
        sum += w[j] * x[i][j];
    }
    if(sum == 0){
        if(y[i] >= 0) return 0;
        else return 1;
    }
    else if (sum * y[i] < 0) {
        return 1;
    } else {
        return 0;
    }
}

int main() {
    int M = N * 4;
    int y[N];
    double x[N][n+1], w[n+1];
    for (int i = 0; i < N; i++) {
        x[i][0] = 1;
        for (int j = 1; j <= n; j++) {
            scanf("%lf", &x[i][j]);
        }
        scanf("%d", &y[i]);
    }
    for(int i = 0; i < N; i++){
        for(int j = 0; j <= n; j++){
            x[i][j] /= 2;
        }
    }
    int update[1000] = {};
    int error_sum = 0;
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j <= n; j++) {
            w[j] = 0;
        }
        srand(i);
        int correct = 0;
        while(correct < M){
            int num = rand() % N;
            if (find_mistake(num, w, x, y)) {
                update[i]++;
                for (int j = 0; j <= n; j++) {
                    w[j] += y[num] * x[num][j];
                }
                correct = 0;
            }
            else correct++;
        }
        int error = 0;
        for (int j = 0; j < N; j++) {
            if (find_mistake(j, w, x, y)) {
                error++;
            }
        }
        error_sum += error;
        //w_0[i] = w[0];
    }
    sort(update, update + 1000);
    printf("median = %d\n", (update[499] + update[500]) / 2);
    return 0;
}
