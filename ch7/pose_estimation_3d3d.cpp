#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <chrono>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

using namespace std;
using namespace cv;

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

//这里定义了固定的图像数据路径，方便运行调试
const std::string img_file1 = "../1.png";
const std::string img_file2 = "../2.png";
const std::string img_file3 = "../1_depth.png";
const std::string img_file4 = "../2_depth.png";


void find_feature_matches(const Mat &img1,
                            const Mat &img2,
                            vector<KeyPoint> &keypoints_1,
                            vector<KeyPoint> &keypoints_2,
                            vector<DMatch> &matches);

Point2d pixel2cam(const Point2d &p, const Mat &K);


void pose_estimiation_3d3d(const vector<Point3f> &pts1,
                            const vector<Point3f> &pts2,
                            Mat &R,
                            Mat &t);

void bundleAdjustment(const vector<Point3f> &pts1,
                            const vector<Point3f> &pts2,
                            Mat &R,
                            Mat &t);

int main(int argc,char* argv[]) {
    // if(argc != 4) {
    //     cout << "Usage: pose_estimation_3d2d img1[First] img2[Second] img3[Depth]." << endl;
    //     return 1;
    // }

    //read image from disk
    Mat img_1 = imread(img_file1,cv::IMREAD_COLOR);
    Mat img_2 = imread(img_file2,cv::IMREAD_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);
    
    vector<DMatch> matches;
    vector<KeyPoint> keypoints_1, keypoints_2;
    //找到两张图像的特征匹配对
    find_feature_matches(img_1,img_2,keypoints_1,keypoints_2,matches);
    //读入深度图像
    Mat depth_image1 = imread(img_file3,IMREAD_UNCHANGED);
    Mat depth_image2 = imread(img_file4,IMREAD_UNCHANGED);

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts1_3d;
    vector<Point3f> pts2_3d;
    for(DMatch m : matches) {
        ushort d1 = depth_image1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)]; // 根据特征点的y x（行 列）坐标找到深度图当中的深度信息
        ushort d2 = depth_image2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)]; // 根据特征点的y x（行 列）坐标找到深度图当中的深度信息
        if(d1 == 0 || d2 == 0) {
            cout << "Depth should not be less than 0!" << endl;
            continue;
        }
        float dd1 = d1 / 5000.0;
        float dd2 = d2 / 5000.0;

        Point2d p_cam1 = pixel2cam(keypoints_1[m.queryIdx].pt,K); //这里需要先通过第一张图计算到世界坐标系的3D点坐标
        Point2d p_cam2 = pixel2cam(keypoints_2[m.trainIdx].pt,K); //计算第二张图像极坐标系下的x和y坐标
        Point3f p_3d1 = Point3f(p_cam1.x * dd1 , p_cam1.y * dd1, dd1);
        Point3f p_3d2 = Point3f(p_cam2.x * dd2 , p_cam2.y * dd2, dd2);
        Point2f p_cam_trans = Point2f(keypoints_2[m.trainIdx].pt);//这里是第二张图像像素坐标
        pts1_3d.push_back(p_3d1);
        pts2_3d.push_back(p_3d2);
    }

    cout << "3D-3D pairs: " << pts1_3d.size() << endl;

    //3D-3D ICP 配准
    Mat R_svd,t_svd;
    pose_estimiation_3d3d(pts1_3d,pts2_3d,R_svd,t_svd);
    cout << "3D-3D using SVD method, R: \n" << R_svd << endl;
    cout << "3D-3D using SVD method, t: \n" << t_svd << endl;
    cout << "R_inv = " << R_svd.t() << endl;
    cout << "t_inv = " << -R_svd.t() * t_svd << endl;
    Mat R_g2o,t_g2o;
    bundleAdjustment(pts1_3d,pts2_3d,R_g2o,t_g2o);
    cout << "3D-3D using g2o method, R: \n" << R_g2o << endl;
    cout << "3D-3D using g2o method, t: \n" << t_g2o << endl;
    cout << "R_inv = " << R_g2o.t() << endl;
    cout << "t_inv = " << -R_g2o.t() * t_g2o << endl;
    //-- 验证E=t^R*scale
    // Mat R, t;
    // Mat t_x =
    //     (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
    //     t.at<double>(2, 0), 0, -t.at<double>(0, 0),
    //     -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    // cout << "t^R=" << endl << t_x * R << endl;

    //  //-- 验证对极约束
    // for (DMatch m: matches) {
    //     Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    //     Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1); //把像素坐标转换成归一化相机坐标，代入y2t^Ry1=0,检查等式左侧结果与0的接近程度。如果很小，说明估计效果良好
    //     Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    //     Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
    //     Mat d = y2.t() * t_x * R * y1;
    //     cout << "epipolar constraint = " << d << endl;
    // }
    return 0;
}

void find_feature_matches(const Mat &img_1,
                            const Mat &img_2,
                            vector<KeyPoint> &keypoints_1,
                            vector<KeyPoint> &keypoints_2,
                            vector<DMatch> &matches) {
    Mat descriptors_1, descriptors_2;
    vector<DMatch> raw_matches;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    //检测Oriented FAST 角点位置
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    detector->detect(img_1,keypoints_1);
    detector->detect(img_2,keypoints_2);

    //根据角点位置计算BRIEF描述子
    descriptor->compute(img_1,keypoints_1,descriptors_1);
    descriptor->compute(img_2,keypoints_2,descriptors_2);

    //根据计算得到的描述子进行匹配，使用汉明距离

    matcher->match(descriptors_1,descriptors_2,raw_matches);

    //过滤raw_matches
    double dis_max = 0;
    double dis_min = 99999;

    //找到matches里面距离最大和最小，代表了特征点匹配程度，distance越大，匹配程度越低
    for(int i = 0; i < raw_matches.size();i++) {
        if(raw_matches[i].distance < dis_min) {
            dis_min = raw_matches[i].distance;
        }
        if(raw_matches[i].distance > dis_max) {
            dis_max = raw_matches[i].distance;
        }
    }

    printf("-- Max dist : %f \n",dis_max);
    printf("-- Min dist : %f \n",dis_min);

    for(int i = 0;i < raw_matches.size();i++) {
        if(raw_matches[i].distance < max(2* dis_min,30.0)) {
            matches.push_back(raw_matches[i]);
        }
    }
    cout <<"Final matches size is: " << matches.size() << endl;
}


Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}


void pose_estimiation_3d3d(const vector<Point3f> &pts1,
                            const vector<Point3f> &pts2,
                            Mat &R,
                            Mat &t) {
    //第一步：计算两组点的质心
    Point3f p1_c,p2_c;
    int N = pts1.size();
    if(N <= 0) {
        cout << "The number of given points is less than 0!;" << endl;
        return;
    }
    for(int i = 0; i < pts1.size(); i++) {
        p1_c += pts1[i];
        p2_c += pts2[i];
    }

    p1_c = p1_c / N;
    p2_c = p2_c / N;
    //第二步：计算去质心后的坐标
    vector<Point3f> qts1,qts2;
    for(int i = 0 ; i < N; i++) {
        qts1.push_back(pts1[i] - p1_c);
        qts2.push_back(pts2[i] - p2_c);
    }

    //第三步：计算q1*q2T
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for(int i = 0 ; i < N; i++) {
        W += Eigen::Vector3d(qts1[i].x,qts1[i].y,qts1[i].z) * Eigen::Vector3d(qts2[i].x,qts2[i].y,qts2[i].z).transpose();
    }

    cout << "W: \n" << W.matrix() << endl;

    //第四步：计算W的SVD分解
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W,Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();


    cout << "U = \n" << U.matrix() << endl;
    cout << "V = \n" << V.matrix() << endl;

    //第五步：计算旋转矩阵R
    Eigen::Matrix3d R_ = U * (V.transpose());
    if(R_.determinant() < 0) {
        R_ = - R_;
    }
    //第六步：计算平移向量
    Eigen::Vector3d t_ = Eigen::Vector3d(p1_c.x,p1_c.y,p1_c.z) - R_ * Eigen::Vector3d(p2_c.x,p2_c.y,p2_c.z);

    //第七步：Eigen转换成cv::Mat
    R = (Mat_<double>(3,3) << 
            R_(0,0),R_(0,1),R_(0,2),
            R_(1,0),R_(1,1),R_(1,2),
            R_(2,0),R_(2,1),R_(2,2));

    t = (Mat_<double>(3,1) <<
            t_(0,0),
            t_(1,0),
            t_(2,0));
}

//定义g2o下的节点和边，借助g2o来优化求相机的运动
//在这个模型中，只有一个节点就是相机的位姿变换，这个位姿变换是从cam2到cam1，所以观测是cam1,构造边时候传递进去的是cam2下面的坐标
//误差模型就是p_cam1 - T * p_cam2
class PoseVertex : public g2o::BaseVertex<6,Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update) override {
        Eigen::Matrix<double,6,1> update_eigen;
        update_eigen << update[0] , update[1] , update[2] , update[3] , update[4] , update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

class ProjectXYZRGBDOnlyEdge : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, PoseVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ProjectXYZRGBDOnlyEdge(const Eigen::Vector3d &point) : _point(point){}

    virtual void computeError() override {
        const PoseVertex* v = static_cast<const PoseVertex *> (_vertices[0]);
        const Sophus::SE3d T = v->estimate();
        _error = _measurement - T * _point;//误差项，是一个3*1的向量.要注意，误差项是谁减去谁（这里是观测值减去预测值p_cam1 - T * p_cam2），要和下面的雅克比矩阵对应上。
    }

    //Jacobian -- 这里需要准备的数据是计算雅克比矩阵必须的。在这个case里面，我们需要的是相机坐标系下面的X,Y,Z
    virtual void linearizeOplus() override {
        const PoseVertex* v = static_cast<const PoseVertex *> (_vertices[0]);
        const Sophus::SE3d T = v->estimate();
        Eigen::Vector3d xyz_trans = T * _point;
        //雅克比矩阵：3*6.误差是xyz三个方向，相机变换是SE3的，六个维度
        //下面的操作，是Eigen提供的矩阵块赋值，https://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
        //<3,3>代表了3*3的块，后面的括号是表示这个块左上角的元素在原矩阵当中的索引位置。具体到这里，就是前三列是负单位矩阵,
        //后三列是变换后的p'的反对称矩阵，其实是so(3)的。
        //写到这里不得不再次回到李群李代数那一章节，复习了一下指数映射和对数映射。SO(3)上才是旋转矩阵，so(3)上要么是三维的向量，要么是这个三维向量对应
        //的反对称矩阵。还记得罗德里格斯公式吗？罗德里格斯公式是建立so(3) -> SO(3)的快捷计算的，只需要首先把三维向量单位化并且求模，代入公式就可以求出
        //SO(3)下面的旋转矩阵
        _jacobianOplusXi.block<3,3>(0,0) = - Eigen::Matrix3d::Identity();
        _jacobianOplusXi.block<3,3>(0,3) = Sophus::SO3d::hat(xyz_trans);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

public:
Eigen::Vector3d _point;
};

void bundleAdjustment(const vector<Point3f> &pts1,const vector<Point3f> &pts2,Mat &R,Mat &t) {
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;  // pose is 6, landmark is 3
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;     // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    // vertex
    PoseVertex *vertex_pose = new PoseVertex(); //相机位姿节点
    vertex_pose->setId(0);
    vertex_pose->setEstimate(Sophus::SE3d());
    optimizer.addVertex(vertex_pose);

    Eigen::Matrix3d K_eigen;
    //edges
    int index = 1;
    for (size_t i = 0; i< pts1.size(); ++i) {
        auto p1 = pts1[i];
        auto p2 = pts2[i];
        Eigen::Vector3d eigen_point3d(p2.x,p2.y,p2.z);
        ProjectXYZRGBDOnlyEdge *edge = new ProjectXYZRGBDOnlyEdge(eigen_point3d);
        edge->setId(index);
        edge->setVertex(0,vertex_pose);
        edge->setMeasurement(Eigen::Vector3d(pts1[i].x,pts1[i].y,pts1[i].z));//p_cam1 - T * p_cam2
        edge->setInformation(Eigen::Matrix3d::Identity());
        optimizer.addEdge(edge);
        index++;
    }
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    cout << "pose estimation by g2o = \n" << vertex_pose->estimate().matrix() << endl;
    
    //把结果转换成cv::Mat
    Eigen::Matrix3d R_ = vertex_pose->estimate().rotationMatrix();
    Eigen::Vector3d t_ = vertex_pose->estimate().translation();

    R = (Mat_<double>(3,3) << R_(0,0), R_(0,1),R_(0,2),
                                R_(1,0), R_(1,1),R_(1,2),
                                R_(2,0), R_(2,1),R_(2,2));

    t = (Mat_<double>(3,1) << t_(0,0),t_(1,0),t_(2,0));
}

