#include <opencv2/opencv.hpp>
#include <string>

using namespace std;

string image_file = "../distorted.png";


int main(int argc,char* argv[]){

    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;
    cv::Mat image = cv::imread(image_file,0); //image type is: CV_8UC1
    int rows = image.rows, cols = image.cols;

    cv::Mat undistorted_image = cv::Mat(rows,cols,CV_8UC1);//construct undistored image, same size of the original one

    //calculate by code. (u,v)is in pixel frame. First step should be pixel frame to image frame
    for(size_t v = 0;v < rows; v++){
        for(size_t u = 0; u < cols; u++){
            double x = (u - cx) / fx; //归一化平面上的坐标（理想状态下）
            double y = (v - cy) / fy; 
            double r = sqrt(x * x + y * y);
            double x_dis = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
            double y_dis = y * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p2 * x * y + p1 * (r * r + 2 * y * y);
            double u_distorted = fx * x_dis + cx;
            double v_distorted = fy * y_dis + cy;

            //set value
            if(u_distorted >= 0 && v_distorted >= 0 && u_distorted < image.cols && v_distorted < image.rows){
                undistorted_image.at<uchar>(v, u) = image.at<uchar>((int) v_distorted, (int) u_distorted);
            }else{
                undistorted_image.at<uchar>(v, u) = 0;
            }
        }
    }

    cv::namedWindow("Original image", cv::WINDOW_NORMAL);
    cv::namedWindow("Undistorted image", cv::WINDOW_NORMAL);
    cv::moveWindow("Original image",100,50);
    cv::imshow("Original image",image);
    cv::moveWindow("Undistorted image",500,500);
    cv::imshow("Undistorted image",undistorted_image);
    cv::waitKey(0);

    return 0;
}
