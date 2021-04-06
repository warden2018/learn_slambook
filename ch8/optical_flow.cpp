#include <chrono>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;



const string file_1 = "../LK1.png"; //First image
const string file_2 = "../LK2.png"; //second image


// typedef Eigen::Matrix<double, 4, 4> Matrix4d;


/**
 * Get the gray scale value from the image (bi-linear interpolated) 
 * 双线性内插法
 * @param [in] img input image reference 
 * @param [in] x x coordination
 * @param [in] y y coordination
 * @return the gray scale value I(x,y)
 */
inline float GetPixelValue(const cv::Mat& img, float x, float y) {
    //确定x和y的有效性，避免内存访问出现越界
    if(x < 0) x = 0;
    if(y < 0) y = 0;
    if(x >= img.cols) x = img.cols - 1;
    if(y >= img.rows) y = img.rows - 1;
    //在一个单位边长为1的正方形当中，已知四个顶点的灰度值，已知P(x,y)在这个正方形中
    //的相对位置xx,yy,求灰度的线性插值
    float xx = x - floor(x);
    float yy = y - floor(y);
    uchar* data = &img.data[int(y) * img.step + int(x)]; //把二维的CV_8UC1看做一个一维数组，寻找这个正方形左上顶点的位置
    //插值结果：按照P(x,y)到各个点的距离来计算，离得近就权重大，离得远就权重小
    float grey_interpolated = data[0] * (1 - xx) * (1 - yy) //正方形左上点
                            + data[1] * xx * (1 - yy) // 正方形右上点
                            + data[img.step] * yy * (1 - xx) //正方形左下点
                            + data[img.step + 1] * xx * yy; //正方形右下点
    return grey_interpolated;
}


//Optical flow tracker interface used by single level and multilevel
/*
* 注意下面的类成员是引用，也就是不会在类对象开辟自己的空间维护特征点和成功标志位
* 所有实例化的对象：例如 vector<KeyPoint>, Mat 都是在main()函数里面完成的。
*/
class OpticalFlowTracker {
public:
    OpticalFlowTracker(
        const Mat &img1,
        const Mat &img2,
        const vector<KeyPoint> &keypoints_input,
        vector<KeyPoint> &keypoints_output,
        vector<bool> &success,
        bool inverse = true,
        bool has_initial = false
    ) : img1_(img1), img2_(img2), keypoints_1_(keypoints_input),keypoints_2_(keypoints_output),
        success_(success), inverse_(inverse), has_initial_(has_initial) {
        }

    void calculateOpticalFlow_GN(const Range &range);
    void calculateOpticalFlow_LSQ(const Range &range); //最小二乘解
    //vector<KeyPoint> getKeyPoints();
private:
    const Mat& img1_;
    const Mat& img2_;
    const vector<KeyPoint>& keypoints_1_;
    vector<KeyPoint>& keypoints_2_;
    vector<bool>& success_;
    bool inverse_;
    bool has_initial_;
};

void OpticalFlowTracker::calculateOpticalFlow_LSQ(const Range &range) {
    int half_patch_size = 4;
    const double time_interval = 0.03;
    for(size_t i = range.start; i < range.end; i++) {
        auto kp = keypoints_1_[i];
        double dx = 0, dy = 0;
        Eigen::MatrixXd A(4*half_patch_size*half_patch_size,2);
        Eigen::VectorXd b(4*half_patch_size*half_patch_size);
        
        for(int x = -half_patch_size; x < half_patch_size; x++) {
            for(int y = -half_patch_size; y < half_patch_size; y++) {
                int row_id = (x +  half_patch_size) * 2 * half_patch_size + y + half_patch_size;
                //cout << "row_id: " << row_id << endl;
                A(row_id,0) = 0.5 * (GetPixelValue(img2_,kp.pt.x + x + 1, kp.pt.y + y)
                                - GetPixelValue(img2_,kp.pt.x + x - 1, kp.pt.y + y));
                A(row_id,1) = 0.5 * (GetPixelValue(img2_,kp.pt.x + x, kp.pt.y + y + 1)
                                - GetPixelValue(img2_,kp.pt.x + x, kp.pt.y + y - 1));
                b(row_id) = (GetPixelValue(img2_,kp.pt.x + x, kp.pt.y + y) - 
                            GetPixelValue(img1_,kp.pt.x + x, kp.pt.y + y)) / time_interval;
            }        
        }

        Eigen::Vector2d vel = -(A.transpose() * A).inverse() * A.transpose() * b; //光流速度
        Eigen::Vector2d pos_delta = vel * time_interval; 
        success_[i] = true;
        keypoints_2_[i].pt = kp.pt + Point2f(pos_delta(0),pos_delta(1));
    }
}


void OpticalFlowTracker::calculateOpticalFlow_GN(const Range &range) {
    int half_patch_size = 4;
    int iteration = 20;
    //遍历第一张图的所有特征点
    for(size_t i = range.start;i < range.end; ++i){
        auto kp = keypoints_1_[i];
        double dx = 0;
        double dy = 0;
        if(has_initial_) {//如果没有使能初始化，那么优化的初始值dx和dy分别都是0
            dx = keypoints_2_[i].pt.x - kp.pt.x;
            dy = keypoints_2_[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastcost = 0;
        bool success = true;

        //开始高斯牛顿法迭代求dx,dy.图像灰度值关于dx, dy的雅克比是一个2*1的列向量。雅克比矩阵每一个元素其实分别是
        //x和y方向的图像梯度，计算方法是通过中值差分，dx(i,j) = 0.5*(I(i+1,j) - I(i-1,j))
        //dy(i,j)=0.5*(I(i,j+1) - I(i,j-1))
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        Eigen::Vector2d J = Eigen::Vector2d::Zero();
        Eigen::Vector2d b = Eigen::Vector2d::Zero();
        //这里需要特别说明反向法光流的原理。
        for(int iter = 0; iter < iteration; iter++) {
            if(inverse_ == false) {//如果不是反向光流，每次迭代都需要重新计算H和b
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            }
            else {//如果是反向光流，每次迭代不需要重新计算H。在I1上面计算好，每次只更新b即可。
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;
            for(int x = -half_patch_size; x < half_patch_size; x++) {
                for(int y = -half_patch_size; y < half_patch_size; y++) {
                    //计算P(img.p.x + x,omg.p.y + y)灰度值的误差，这里是测量值 - 预测值
                    double error = GetPixelValue(img1_,kp.pt.x + x, kp.pt.y + y) - 
                                    GetPixelValue(img2_,kp.pt.x + x + dx,kp.pt.y + y + dy);

                    if(inverse_==false){
                        J = - 1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2_,kp.pt.x + dx + x + 1, kp.pt.y + dy + y)
                                - GetPixelValue(img2_,kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2_,kp.pt.x  + dx + x, kp.pt.y + dy + y + 1)
                                - GetPixelValue(img2_,kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );
                    }else if(iter==0){//反向光流计算，用I1上面的点计算梯度作为雅克比矩阵，一次计算，整个过程不变
                        J = - 1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1_,kp.pt.x + x + 1, kp.pt.y + y)
                                - GetPixelValue(img1_,kp.pt.x  + x - 1, kp.pt.y  + y)),
                            0.5 * (GetPixelValue(img1_,kp.pt.x  + x, kp.pt.y  + y + 1)
                                - GetPixelValue(img1_,kp.pt.x  + x, kp.pt.y  + y - 1))
                        );
                    }
                b += - error * J;
                if(inverse_ == false || iter == 0){//如果不是反向光流，每次都需要更新H；如果是反向光流而且第一次迭代
                //，也需要更新H，但是更新之后就不需要在后面的迭代中更新H了。
                    H += J * J.transpose();
                }
                cost += error * error;
                }
            }

            Eigen::Vector2d update = H.ldlt().solve(b);

            if(std::isnan(update[0])){
                //有时候如果图像块是纯色，比如纯黑色或者纯白色，计算出来的H是不可逆的，导致无解.本次迭代终止
                cout << "update is nan." << endl;
                success = false;
                break;
            }

            if(iter > 0 && cost > lastcost) { //如果本次迭代cost更大，说明结果不好，这次的更新不采纳，终止本次迭代
                break;
            }

            dx += update[0];
            dy += update[1];
            lastcost = cost;
            success = true;

            if(update.norm() < 1e-2) { //更新量较小，认为达到最优，终止迭代
                //cout << "Converge." << endl;
                break;
            }
        }
        success_[i] = success;
        //更新I2上面的特征点坐标
        keypoints_2_[i].pt = kp.pt + Point2f(dx,dy);
        //cout << "Keypoints2[" <<i << "] x: " << keypoints_2_[i].pt.x << " y: " << keypoints_2_[i].pt.y << endl;
    }
}
/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false,
    bool has_initial_guess = false
);

/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] kp1 keypoints in img1
 * @param [out] kp2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
    const Mat &img1,
    const Mat &img2,
    const vector<KeyPoint> &kp1,
    vector<KeyPoint> &kp2,
    vector<bool> &success,
    bool inverse = false
);



int main(int argc, char* argv[]) {
    Mat img_1 = imread(file_1,0);
    Mat img_2 = imread(file_2,0);
    vector<KeyPoint> kpts1;
    vector<KeyPoint> kpts2_single;
    vector<bool> success_single;
    vector<KeyPoint> kpts2_multiple;
    vector<bool> success_multiple;
    bool inverse = true;
    bool has_init = false;


    Ptr<GFTTDetector> detector = GFTTDetector::create(500,0.01,20); //TODO: 这几个参数代表什么意思？
    detector->detect(img_1,kpts1); //检测到第一张图的特征点，存储在ktps1里面
    
    //跟踪第二张图里面ktps1在第二张图上面哪里
    //线性最小二乘方法可以在OpticalFlowSingleLevel里面调整
    
    //第一种方法是单层光流single level optical flow
    cout << "Key Points detected in imge1 size is: " << kpts1.size() << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    OpticalFlowSingleLevel(img_1,img_2,kpts1,kpts2_single,success_single,inverse,has_init);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    auto time_used_single = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Time cost by single optical flow is: " << time_used_single.count() << endl;


    //第二种方法是多层金字塔光流 multiple optical flow
    t1 = chrono::steady_clock::now();
    OpticalFlowMultiLevel(img_1,img_2,kpts1,kpts2_multiple,success_multiple,inverse);
    //cout << "key points by single optical flow size is: " << kpts2_single.size();
    t2 = chrono::steady_clock::now();
    auto time_used_multiple = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Time cost by multiple optical flow is: " << time_used_multiple.count() << endl;

    //第三种方法Opencv
    vector<Point2f> pt1, pt2;
    for (auto &kp: kpts1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    t1 = chrono::steady_clock::now();
    cv::calcOpticalFlowPyrLK(img_1, img_2, pt1, pt2, status, error);
    t2 = chrono::steady_clock::now();
    auto time_used_opencv = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Time cost by opencv: " << time_used_opencv.count() << endl;

    /*画出来各个方法的效果
    * 起点是I1的特征点，终点是估计出来的对应特征点
    */
    Mat img2_single;
    cv::cvtColor(img_2, img2_single, CV_GRAY2BGR);
    for (int i = 0; i < kpts2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kpts2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kpts1[i].pt, kpts2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    //multiple 
    Mat img2_multiple;
    cv::cvtColor(img_2, img2_multiple, CV_GRAY2BGR);
    for (int i = 0; i < kpts2_multiple.size(); i++) {
        if (success_multiple[i]) {
            cv::circle(img2_multiple, kpts2_multiple[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multiple, kpts1[i].pt, kpts2_multiple[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    //使用Opencv的方法 
    Mat img2_opencv;
    cv::cvtColor(img_2, img2_opencv, CV_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_opencv, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_opencv, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }


    // cv::imshow("Orignal image 1", img_1);
    // cv::imshow("Orignal image 2", img_2);
    cv::imshow("tracked single level", img2_single);
    cv::imshow("tracked multiple level", img2_multiple);
    cv::imshow("track by opencv", img2_opencv);
    cv::waitKey(0);
    return 0;
}

//这个函数主要完成并行地对每一个特征点完成光流计算，得到该特征点在下一帧图像上的运动，结果放在kp2里面
void OpticalFlowSingleLevel(const Mat &img1,const Mat &img2,const vector<KeyPoint> &kp1,vector<KeyPoint> &kp2,
                        vector<bool> &success,bool inverse,bool has_initial_guess) {
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1,img2,kp1,kp2,success,inverse,has_initial_guess);
    //cout << "tracker established!" << endl << "kp2 size: " << kp2.size() << endl;
    parallel_for_(Range(0,kp1.size()),
                std::bind(&OpticalFlowTracker::calculateOpticalFlow_GN,&tracker,placeholders::_1));

    //cout << "Finished calculateOpticalFlow: kp2 first element: " << kp2[0].pt.x << endl;
}

void OpticalFlowMultiLevel(const Mat &img1,const Mat &img2,const vector<KeyPoint> &kp1,
                vector<KeyPoint> &kp2,vector<bool> &success,bool inverse) {
    int pyramids = 4;//四层金字塔
    double pyramid_scale = 0.5;
    double scales[] = {1.0,0.5,0.25,0.125};

    //创建两张图像的金字塔
    vector<Mat> pyramids_1,pyramids_2;
    for(int i = 0; i < pyramids; i++) {
        if(i==0){
            pyramids_1.push_back(img1);
            pyramids_2.push_back(img2);
        }
        else{
            Mat img1_pyr,img2_pyr;
            cv::resize(pyramids_1[i-1],img1_pyr,
                        cv::Size(pyramids_1[i-1].cols * pyramid_scale,pyramids_1[i-1].rows * pyramid_scale));
            cv::resize(pyramids_2[i-1],img2_pyr,
                        cv::Size(pyramids_2[i-1].cols * pyramid_scale,pyramids_2[i-1].rows * pyramid_scale));
            pyramids_1.push_back(img1_pyr);
            pyramids_2.push_back(img2_pyr);
        }
    }

    //从粗到精。pyramids_1和pyramids_2里面的图像随着索引号的增加，图像的缩小比例变大。第一张图是原始图像
    //第一步，需要把在原始图像当中计算得到的特征点坐标转换到金字塔各个层上面，对应起来
    vector<KeyPoint> kp1_pyr,kp2_pyr;//存储的是缩放最小图像上面特征点的坐标
    for(auto& kp : kp1) {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for(int level = pyramids - 1; level >= 0; level--) {
        success.clear();

        OpticalFlowSingleLevel(pyramids_1[level],pyramids_2[level],kp1_pyr,kp2_pyr,success,inverse,true);
        if(level > 0) {
            for(auto& kp : kp1_pyr) {
                kp.pt /= pyramid_scale;
            }
            for(auto& kp : kp2_pyr) {
                kp.pt /= pyramid_scale;
            }
        }
    }

    for(auto& kp : kp2_pyr) {
        kp2.push_back(kp);
    }
}