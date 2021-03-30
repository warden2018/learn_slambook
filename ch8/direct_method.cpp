#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "common/directMethod_flags.h"



/*
 * 直接法demo
 * 已知的是若干图像，图像上对应的像素点的深度，计算相机运动SE3
 * 为了获取深度，demo中通过disparity image计算像素的深度，这个深度坐标其实是拍摄第一张图时候相机坐标系下的坐标。
 * 和原版本不同的是，代码添加了对GFlag的支持，能够通过conf/direct_method.conf传递参数，避免为了修改参数重复编译代码
 * 如何运行这个demo：
 * cd build
 * ./direct_method -flagfile=../conf/direct_method.conf
 */
using namespace std;
using namespace cv;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef Eigen::Matrix<double,6,6> Matrix6d;
typedef Eigen::Matrix<double,2,6> Matrix26d;
typedef Eigen::Matrix<double,6,1> Vector6d;

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;

boost::format fmt_others("../%06d.png");


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

/**
 * Create pyramids images
 * @param [in] img input image reference
 * @param [in] pyramids: level number of the pyramid
 * @param [in] pyramidsScale: scale of neighbor levels
 * @param [out] pyramidImgs
 */
inline bool CreatePyramids(const Mat& img, const int pyramids, const double pyramidsScale,vector<Mat>& pyramidImgs) {
    for(int i = 0; i < pyramids; i++) {
        if(i == 0) {
            pyramidImgs[0] = img;
        } else {
            Mat img_pyr;
            cv::resize(pyramidImgs[i - 1], img_pyr,
                       cv::Size(pyramidImgs[i - 1].cols * pyramidsScale, pyramidImgs[i - 1].rows * pyramidsScale));
            pyramidImgs[i] = img_pyr;
        }
    }
}

//为了实现并行化计算雅克比矩阵写的类,该类每次计算使用的是一张图像里面随机选择出来的
//那些点周围的patch
class JacobianAccumulator {
public:
    JacobianAccumulator(
            const cv::Mat& img1,
            const cv::Mat& img2,
            const VecVector2d& pixels_points,
            const vector<double>& depths,
            Sophus::SE3d& T21) :
            img1_(img1),img2_(img2),pixels_points_(pixels_points),
            depths_(depths),T21_(T21) {
        projected_points_ = VecVector2d(pixels_points.size(),Eigen::Vector2d(0,0));
    }

    void Accumulate_jacobian(const cv::Range& range);

    Matrix6d Get_Hessian() const {return H_;}

    Vector6d Get_bias() const {return b_;}

    double Get_totalCost() const {return cost_;}

    VecVector2d Get_projectedPoints() const {return projected_points_;}

/**
 * Set Hessian cost and bias to zero.
 *
 */
    void Reset() {
        H_ = Matrix6d::Zero();
        b_ = Vector6d::Zero();
        cost_ = 0;
    }


private:
    const cv::Mat& img1_;
    const cv::Mat& img2_;
    const VecVector2d& pixels_points_; //相机1的pixel坐标
    const vector<double>& depths_;
    Sophus::SE3d &T21_;//相机1的坐标左乘该矩阵可以得到在相机2坐标系下的坐标（3D）
//    VecVector2d projected_points_; //相机1里面选出来的那些像素点对应的三维点投影到相机2像素上面的坐标
//                                    //用来评判相机1到相机2的转换矩阵的结果怎么样
//
//    std::mutex hessian_mutex_;
//    Matrix6d H_ = Matrix6d::Zero();
//    Vector6d b_ = Vector6d::Zero();
//    double cost_ = 0;

    mutable VecVector2d projected_points_; // projected points

    mutable std::mutex hessian_mutex_;
    mutable Matrix6d H_ = Matrix6d::Zero();
    mutable Vector6d b_ = Vector6d::Zero();
    mutable double cost_ = 0;
};

void JacobianAccumulator::Accumulate_jacobian(const cv::Range &range) {
    int good_points_inC2 = 0;
    //cout << "T21_: " << T21_.matrix() << endl;
    //cout << "pixels_points_ size: " << pixels_points_.size() << endl;
    //cout << "depths_ size: " << depths_.size() << endl;
    //cout << "img2_ size. cols: " << img2_.cols << ";rows: " << img2_.rows << endl;
    //遍历所有image1随机产生的像素点
    for(size_t i = range.start; i < range.end; i++) {
        Matrix6d hessian = Matrix6d::Zero(); //每一个patch对应一个hessain
        Vector6d bias = Vector6d::Zero();//每一个patch对应一个bias
        double cost_tmp = 0;//每一个patch对应一个cost_tmp

        //1.计算三维点投影到第二张图像上的像素坐标
        //计算相机1坐标系下的三维点坐标
        Eigen::Vector3d point_3d_inC1 =
                depths_[i] * Eigen::Vector3d((pixels_points_[i][0] - cx) / fx,
                                             (pixels_points_[i][1] - cy) / fy, 1.0);
        //计算相机2坐标系下的三维坐标
        Eigen::Vector3d point_3d_inC2 = T21_ * point_3d_inC1;
        //检查计算结果的Z是否为负数
        if(point_3d_inC2[2] <= 0) {
            //cout << "The " << i << "th 3D piont in camera 2 is negative. Abort it." << endl;
            continue;
        }

        //计算相机2下的像素坐标,之前一直认为像素坐标都是整数，其实在把相机坐标系投影到像素坐标系过程中，计算得到的是double类型，
        //但是没办法通过非整形的坐标找对应的灰度值，所以就需要做两个方向的线性插值（双线性内插值），就是下面的GetPixelValue函数。
        float u = fx * point_3d_inC2[0] / point_3d_inC2[2] + cx;
        float v = fy * point_3d_inC2[1] / point_3d_inC2[2] + cy;
        //检查计算得到的像素坐标是不是太靠图像的边缘
        if(u < FLAGS_halfPatchSize || u > (img2_.cols - FLAGS_halfPatchSize)
                || v < FLAGS_halfPatchSize || v > (img2_.rows - FLAGS_halfPatchSize)) {
            //cout << "Projected pixel in img2 is out of border. Should deprecate it." << endl;
            continue;
        }


        projected_points_[i] = Eigen::Vector2d(u,v); //给成员赋值.如果上面的两个检查没有通过，
        cout << "projected_points_" << i << "th element is: x: " << projected_points_[i].x() << "y: "
                << projected_points_[i].y() << endl;

        good_points_inC2++; //每在camera2上面找到有效的三维点，那么就把这个变量加1
        //准备一下求灰度关于李代数的雅克比计算需要的camera2下面的坐标，其实就是换一个名字，和公式推导保持一致
        double X = point_3d_inC2[0];
        double Y = point_3d_inC2[1];
        double Z = point_3d_inC2[2];
        double inverse_Z = 1 / Z;
        double inverse_Z2 = inverse_Z * inverse_Z;

        //2. 计算光度误差和雅克比矩阵
        for(int x = - FLAGS_halfPatchSize; x < FLAGS_halfPatchSize; x++) {
            for(int y = - FLAGS_halfPatchSize; y < FLAGS_halfPatchSize;y++) {
                //每一个点灰度值误差
                double error = GetPixelValue(img1_,pixels_points_[i][0] + x,
                                             pixels_points_[i][1] + y) -
                                             GetPixelValue(img2_,u + x, v + y);
                Matrix26d J_u_xi = Matrix26d::Zero();
                Eigen::Vector2d J_img_u = Eigen::Vector2d::Zero();

                J_u_xi(0,0) = fx * inverse_Z;
                J_u_xi(0,2) = - fx * X * inverse_Z2;
                J_u_xi(0,3) = - fx * X  * Y * inverse_Z2;
                J_u_xi(0,4) = fx + fx * X * X * inverse_Z2;
                J_u_xi(0,5) = - fx * Y * inverse_Z;

                J_u_xi(1,1) = fy * inverse_Z;
                J_u_xi(1,2) = - fy * Y * inverse_Z2;
                J_u_xi(1,3) = - fy - fy * Y * Y * inverse_Z2;
                J_u_xi(1,4) = fy * X * Y * inverse_Z2;
                J_u_xi(1,5) = fy * X * inverse_Z;

                J_img_u(0) = 0.5 * (GetPixelValue(img2_,u + x + 1,v + y)
                                        - GetPixelValue(img2_,u + x -1,v + y));
                J_img_u(1) = 0.5 * (GetPixelValue(img2_,u + x,v + y + 1)
                                    - GetPixelValue(img2_,u + x,v + y -1));

                //总的雅克比矩阵是-(J_img_uT*J_u_xi)T
                Vector6d J = -1.0 * (J_img_u.transpose() * J_u_xi).transpose(); // J是patch里面一个像素计算出来的
                //雅克比矩阵
                //把这一个小patch里面的雅克比相加成hessian和bias
                hessian += J * J.transpose(); //hessian是这个patch
                bias += - error * J; //当前patch的bias
                cost_tmp += error * error; // 当前patch的cost
            }
        }//patch的最外层循环
        if(good_points_inC2) {
            unique_lock<mutex> lck(hessian_mutex_);
            H_ += hessian;
            b_ += bias;
            cost_ += cost_tmp / good_points_inC2;
        }
    }
}

/**
 * pose estimation using direct method
 * @param img1 -- camera1 图像
 * @param img2 -- camera2 图像
 * @param pixel_points -- camera1对应的像素点
 * @param depths -- camera1 看到的深度
 * @param T21 -- camera1 坐标转换到camera2对应的李代数
 */
void DirectPoseEstimationSingleLayer(const cv::Mat img1,
                                     const cv::Mat img2,
                                     const VecVector2d& pixel_points,
                                     const vector<double>& depths,
                                     const int& image_index,
                                     Sophus::SE3d& T21) {
    double cost = 0, lastcost = 0;
    JacobianAccumulator jac_accu(img1,img2,pixel_points,depths,T21);

    //开始循环
    for(int iter = 0; iter < FLAGS_interations; iter++) {
        jac_accu.Reset(); //把H_,b_,cost_归零
        //cv::parallel_for_(cv::Range(0,pixel_points.size()),jac_accu);
        cv::parallel_for_(cv::Range(0,pixel_points.size()),
                          std::bind(&JacobianAccumulator::Accumulate_jacobian,&jac_accu,std::placeholders::_1));
//        jac_accu.Accumulate_jacobian(cv::Range(0,pixel_points.size()));
        Matrix6d H = jac_accu.Get_Hessian();
        Vector6d b = jac_accu.Get_bias();

        //解得到在当前T21下的一个小增量，是se(3)的，需要左乘当前的T21得到新的T21
        Vector6d update = H.ldlt().solve(b);
        T21 = Sophus::SE3d::exp(update) * T21;
        cost = jac_accu.Get_totalCost(); //目标函数的cost


        if(std::isnan(update[0])) {
            //出现这种可能的情况是因为我们选择的patch黑/白？？H是不可逆的。
            cout << "update is nan." << endl;
            break;
        }
        if(iter > 0 && cost > lastcost) {
            cout << "Cost increased: " << lastcost << " to " << cost << endl;
            break;
        }

        if(update[0] < 1e-3) {
            //收敛
            break;
        }

        lastcost = cost;
        cout << "iteration: " << iter << " cost: " << cost << endl;
    }
    cout << "T21: \n"
         << T21.matrix() << endl;

    //画出投影
    cv::Mat img2_show;
    cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    VecVector2d projection = jac_accu.Get_projectedPoints();
    cout << "projected points size: " << projection.size() << endl;
    cout << "pixel_points size: " << pixel_points.size() << endl;

    for (size_t i = 0; i < pixel_points.size(); ++i) {
        auto p_ref = pixel_points[i];
        auto p_cur = projection[i];
        cout << "p_cur.x: [" << p_cur.x() <<"] p_cur.y: [" << p_cur.y() << endl;
        cout << "p_ref.x: [" << p_ref.x() <<"] p_ref.y: [" << p_ref.y() << endl;

        if (p_cur[0] > 0 && p_cur[1] > 0) {
            cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
                     cv::Scalar(0, 250, 0));
        }
    }
    std::string title = "current " + std::to_string(image_index) + "th image";
    cv::imshow(title, img2_show);
    cv::waitKey();
}



/**
 * multiple pyramid pose estimation using direct method
 * @param img1 -- camera1 图像
 * @param img2 -- camera2 图像
 * @param pixel_points -- camera1对应的像素点
 * @param depths -- camera1 看到的深度
 * @param T21 -- camera1 坐标转换到camera2对应的李代数
 */
void DirectPoseEstimationMultiLayer(const cv::Mat img1,
                                     const cv::Mat img2,
                                     const VecVector2d& pixel_points,
                                     const vector<double>& depths,
                                     const int& image_index,
                                     Sophus::SE3d& T21) {
    double scales[] = {1.0,
                       1.0 * FLAGS_pyr_scale,
                       1.0 * FLAGS_pyr_scale * FLAGS_pyr_scale,
                       1.0 * FLAGS_pyr_scale * FLAGS_pyr_scale * FLAGS_pyr_scale
                       };
    //创建图像金字塔
    vector<cv::Mat> pyr1, pyr2;
    pyr1.resize(FLAGS_pyramids);
    pyr2.resize(FLAGS_pyramids);

    CreatePyramids(img1,FLAGS_pyramids,FLAGS_pyr_scale,pyr1);
    CreatePyramids(img2,FLAGS_pyramids,FLAGS_pyr_scale,pyr2);

    //对每一层图像进行单层直接法
    double fxG = fx, fyG = fy, cxG = cx, cyG = cy;  // backup the old values
    for(int level = FLAGS_pyramids - 1; level >= 0; level--) {
        VecVector2d pixel_ref_pyr;
        for(auto &px : pixel_points) {
            pixel_ref_pyr.push_back(scales[level] * px);
        }
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        DirectPoseEstimationSingleLayer(pyr1[level],pyr2[level],pixel_ref_pyr,depths,image_index,T21);
    }
}

int main(int argc,char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    Mat left_img = imread(FLAGS_left_img_file,0); //读入双目视觉的左图，并且是灰度图
    Mat disparity_img = imread(FLAGS_disparity_file,0);

    RNG rng; //随机数生成器
    VecVector2d pixels_points;
    vector<double> depths;
    //随机在left_img上面产生2000个像素坐标，按顺序存储，depths里面放的是对应的像素的深度信息
    cout << "FLAGS_nPoints is: " << FLAGS_nPoints << endl;
    for(int i = 0; i < FLAGS_nPoints; i++) {
        int x = rng.uniform(FLAGS_nBorder,left_img.cols - FLAGS_nBorder);
        int y = rng.uniform(FLAGS_nBorder,left_img.rows - FLAGS_nBorder);
        int disparity = disparity_img.at<uchar>(y,x);
        double depth = fx * baseline / disparity;
        depths.push_back(depth); //特征点的深度
        pixels_points.push_back(Eigen::Vector2d(x,y)); //特征点的像素坐标
    }

    //cout << "depths size: " << depths.size() << endl;

    Sophus::SE3d T_current;

    for(int i = 1; i <= FLAGS_img_number; i++) {
        cv::Mat img = cv::imread((fmt_others % i).str(),0);

        if(img.empty()) {
            cout << "Can't read img2 from current dir: " << (fmt_others % i).str() << endl;
            continue;
        }
        //DirectPoseEstimationSingleLayer(left_img,img,pixels_points,depths,i,T_current);
        DirectPoseEstimationMultiLayer(left_img,img,pixels_points,depths,i,T_current);
    }
    cv::waitKey(0);
    return 0;
}