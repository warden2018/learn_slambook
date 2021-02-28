#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;

/*
高斯-牛顿法求解的对象是f(x),而不是目标函数F(x)。所以，在求解f(x)的雅克比时候，千万不能搞错。
*/
int main(int argc,char* argv[]){
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    int N = 100;

    double w_sigma = 1.0;
    double inv_w_sigma = 1.0 / w_sigma;
    cv::RNG rng;

    vector<double> x_data, y_data;
    for(size_t i = 0; i < N; i++){
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    //start iterating
    int iterations = 100;
    double cost = 0, lastCost = 0;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for(size_t iter = 0; iter < iterations; iter++){
        Matrix3d H = Matrix3d::Zero();
        Vector3d b = Vector3d::Zero();
        cost = 0;

        for(size_t i = 0; i < N; i++){
            double xi = x_data[i],yi = y_data[i];
            double error = yi - exp(ae * xi * xi + be * xi + ce); // This is the f(x)
            Vector3d J = Vector3d::Zero();

            J[0] = - xi * xi * exp(ae * xi * xi + be * xi + ce); //d(error) / da
            J[1] = - xi * exp(ae * xi * xi + be * xi + ce);//d(error) / db
            J[2] = - exp(ae * xi * xi + be * xi + ce);//d(error) / dc

            H += J * inv_w_sigma * inv_w_sigma * J.transpose();
            b += - J * inv_w_sigma * inv_w_sigma * error;
            cost += error * error;
        }

        //solve Hx=b
        Vector3d dx = H.ldlt().solve(b);
        if(isnan(dx[0])){
            cout << "Result is nan!" << endl;
            break;
        }

        if(iter > 0 && cost >=lastCost) {
            cout << "cost: [" << cost << "] >= last cost: [" << lastCost <<  "]" << " break." << endl;
            break;
        }

        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        lastCost = cost;
        cout << "total cost: " << cost << "Update: " << dx.transpose() << " Estimated params: " << ae << "," << be << "," << ce << endl;
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Solve time cost: " << time_used.count() << " seconds." << endl;
    return 0;
}