#include <iostream>
#include <chrono>

using namespace std;


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc,char* argv[]){
    cv::Mat image;
    image = cv::imread(argv[1]);

    if(image.data==nullptr){
        cerr << "Image file" << argv[1] << "does not exit." << endl;
    }

    //file read and show some basic informations
    cout << "Image width: " << image.cols << ",height: " << image.rows <<
         ",channel: " << image.channels() << endl;

    cv::imshow("image",image);
    cv::waitKey(0);

    //distinguish the type of image
    if(image.type()!=CV_8UC1 && image.type()!=CV_8UC3){
        cout << "Image type is not satisfied.Please input another gray or color image." << endl;
        return 0;
    }

    //iterate image
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for(size_t y = 0;y < image.rows; y++) { // for each row
        //row ponter: cv::Mat::ptr
        unsigned char *row_ptr = image.ptr<unsigned char>(y); //y_th row pointer is row_ptr
        for(size_t x = 0;x < image.cols; x++){ //for each col. in the row
            unsigned char *data_ptr = &row_ptr[x * image.channels()];
            for(size_t c = 0; c < image.channels(); c++){ //for each channel in P(x,y)
                unsigned char data = data_ptr[c];
            }
        }
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast <chrono::duration<double>> (t2 - t1);
    cout << "Duration of iterating the whole image is: " << time_used.count() << " S." << endl;
    //Copy of cv::Mat
    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0,0,100,100)).setTo(255);
    cv::imshow("image_clone",image_clone);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;

}

