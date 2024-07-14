#include<bits/stdc++.h>
#include<ctime>
#include<random>
using namespace std;
#define size 64
#define tao 0
#define N 1
double E_out = 10, theta, rec_E_out = 10;
int s;
struct new_x{
    double x, y;
};
double all_x[10][size] = {}, y[size] = {};
new_x x[size];

int main()
{
    for(int j = 0; j < size; j++){
        for(int k = 0; k < 10; k++){
            scanf("%lf", &all_x[k][j]);
        }
        scanf("%lf", &y[j]);
    }
    int w = 1;
    for(int i = 0; i < size; i++){
        x[i].x = all_x[w][i];
        x[i].y = y[i];
    }
    double E_sum = 0;
    s = 2;
    int error = 0;
    for(int i = 0; i < size; i++){
        double h = (x[i].x - 112.8) * -1;
        if(!(h == 0 && x[i].y == -1) && h * x[i].y <= 0){
            error++;
        }
    }
    E_out = (double)error / 64;    
    printf("%lf\n", E_out);   
}