#include<bits/stdc++.h>
#include<ctime>
#include<random>
using namespace std;
#define size 2
#define tao 0.2
#define N 10000
double E_in = 10, theta, E_out;
int s;
void cal_in(double x[], double y[], int cur_s, double cur_theta){
    int error = 0;
    for(int i = 0; i < size; i++){
        double h = (x[i] - cur_theta) * cur_s;
        if(!(h == 0 && y[i] == -1) && h * y[i] <= 0){
            error++;
        }
    }
    if((double)error / (double)size < E_in){
        E_in = (double)error / (double)size;
        theta = cur_theta;
        s = cur_s;
    }
    else if((double)error / (double)size == E_in){
        if((double)cur_s * cur_theta < (double)s * theta){
            theta = cur_theta;
            s = cur_s;
        }
    }
}
int main()
{
    double E_sum = 0;
    for(int i = 0; i < N; i++){
        default_random_engine e(i);
        uniform_real_distribution<double> u(-0.5, 0.5);
        uniform_real_distribution<double> t(0, 1);
        double x[size] = {}, y[size] = {};
        E_in = 10;
        for(int j = 0; j < size; j++){
            x[j] = u(e);
        }
        sort(x, x + size);
        for(int j = 0; j < size; j++){
            y[j] = (x[j] > 0)? 1: -1;
        }
        for(int j = 0; j < size; j++){
            if(t(e) < tao) y[j] = (x[j] <= 0)? 1: -1;
            else y[j] = (x[j] <= 0)? -1: 1;
        }
        for(int j = 0; j < size - 1; j++){
            cal_in(x, y, -1, (x[j] + x[j + 1]) / 2);
            cal_in(x, y, 1, (x[j] + x[j + 1]) / 2);
        }
        cal_in(x, y, -1, -10);
        cal_in(x, y, 1, -10);
        E_out = min(abs(theta), 0.5) * (1 - 2 * tao) + tao;
        //printf("s: %d  theta: %lf  E_out: %lf  E_in: %lf  x[1]: %lf\n", s, theta, E_out, E_in, x[1]);
        E_sum += E_out - E_in;
    }
    double E_mean = E_sum / (double)N;
    printf("%lf\n", E_mean);
}