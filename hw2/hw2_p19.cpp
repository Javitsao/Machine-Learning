#include<bits/stdc++.h>
#include<ctime>
#include<random>
using namespace std;
#define size 192
#define tao 0
#define N 1
double E_in = 10, theta, E_out, max_E_in = -10, min_E_in = 10;
int s;
struct new_x{
    double x, y;
};
double all_x[10][size] = {}, y[size] = {};
new_x x[size];
void cal_in(int cur_s, double cur_theta){
    int error = 0;
    for(int i = 0; i < size; i++){
        double h = (x[i].x - cur_theta) * cur_s;
        if(!(h == 0 && x[i].y == -1) && h * x[i].y <= 0){
            error++;
        }
    }
    //printf("error = %d\n", error);
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
bool cmp(new_x a,new_x b){
    return a.x < b.x;
}
int main()
{
    for(int j = 0; j < size; j++){
        for(int k = 0; k < 10; k++) {
            scanf("%lf", &all_x[k][j]);
        }
        scanf("%lf", &y[j]);
    }
    //printf("x[0][0] = %lf\n", x[0][0]);
    for(int w = 0; w < 10; w++){
        for(int i = 0; i < size; i++){
            x[i].x = all_x[w][i];
            x[i].y = y[i];
        }
        double E_sum = 0;
        for(int i = 0; i < N; i++){
            E_in = 10;
            sort(x, x + size, cmp);
            printf("x = %lf\n", x[0].x);
            for(int j = 0; j < size - 1; j++){
                cal_in(-1, (x[j].x + x[j + 1].x) / 2);
                cal_in(1, (x[j].x + x[j + 1].x) / 2);
            }
            cal_in(-1, -1000000);
            cal_in(1, -1000000);
            E_out = min(abs(theta), 0.5) * (1 - 2 * tao) + tao;
            if(E_in > max_E_in) max_E_in = E_in;
            if(E_in < min_E_in) min_E_in = E_in;
        }
        //printf("%lf\n", E_in);
    }
    printf("\n%lf\n", max_E_in - min_E_in);
    
}