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
#include "ceres/ceres.h"
#include "EigenTypes.hpp"

using namespace std;
using namespace cv;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

//这里定义了固定的图像数据路径，方便运行调试
const std::string img_file1 = "../1.png";
const std::string img_file2 = "../2.png";
const std::string img_file3 = "../1_depth.png";


void find_feature_matches(const Mat &img1,
                            const Mat &img2,
                            vector<KeyPoint> &keypoints_1,
                            vector<KeyPoint> &keypoints_2,
                            vector<DMatch> &matches);

void pose_estimation_2d2d(vector<KeyPoint> &keypoints_1,
                        vector<KeyPoint> &keypoints_2,
                        vector<DMatch> matches,
                        Mat &R,
                        Mat &t);

Point2d pixel2cam(const Point2d &p, const Mat &K);


//BA by GaussNewton
void bundleAdjustmentGaussNewton(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose);


void bundleAdjustmentG2O(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose);

void bundleAdjustmentCeres(
    const VecVector3d &points_3d,
    const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose
);

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


void pose_estimation_2d2d(vector<KeyPoint> &keypoints_1,
                        vector<KeyPoint> &keypoints_2,
                        vector<DMatch> matches,
                        Mat &R,
                        Mat &t) {
    // 相机内参,TUM Freiburg2
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    //把keyPoints转换成vector<Point2f>
    vector<cv::Point2f> points_1;
    vector<cv::Point2f> points_2;
    //按照mathces，找到匹配对
    for(int i = 0; i < matches.size(); i++) {
        points_1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points_2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //这里要理解基础矩阵，本质矩阵和单应矩阵的区别联系：https://www.zhihu.com/question/27581884
    //基础矩阵(FundamentalMatrix)是三维空间P点在不同像素坐标系下形成的约束
    //本质矩阵(Essential Matrix)是三维空间P点在不同的相机坐标系下形成的约束
    //这就是为什么求基础矩阵时候不需要传入相机内参数，但是计算本质矩阵时候需要传入相机的内参数，把像素坐标转换成相机坐标
    Mat fundamental_matrix; //基础矩阵：左右两侧的点坐标系是图像坐标系
    fundamental_matrix = findFundamentalMat(points_1,points_2,FM_8POINT); // 八点法求基础矩阵，还有其他解法
    cout << "Fundamental matrix is: " << endl << fundamental_matrix << endl;

    Mat essential_matrix;
    Point2d principal_point(325.1, 249.7);  //相机光心, TUM dataset标定值
    double focal_length = 521;      //相机焦距, TUM dataset标定值
    essential_matrix = findEssentialMat(points_1,points_2,focal_length,principal_point);
    cout << "Essential matrix is: " << endl << essential_matrix << endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
  // 此函数仅在Opencv3中提供
  /** @overload
    @param E The input essential matrix.
    @param points1 Array of N 2D points from the first image. The point coordinates should be
    floating-point (single or double precision).
    @param points2 Array of the second image points of the same size and format as points1 .
    @param R Output rotation matrix. Together with the translation vector, this matrix makes up a tuple
    that performs a change of basis from the first camera's coordinate system to the second camera's
    coordinate system. Note that, in general, t can not be used for this tuple, see the parameter
    description below.
    @param t Output translation vector. This vector is obtained by @ref decomposeEssentialMat and
    therefore is only known up to scale, i.e. t is the direction of the translation vector and has unit
    length.
    @param focal Focal length of the camera. Note that this function assumes that points1 and points2
    are feature points from cameras with same focal length and principal point.
    @param pp principal point of the camera.
    @param mask Input/output mask for inliers in points1 and points2. If it is not empty, then it marks
    inliers in points1 and points2 for then given essential matrix E. Only these inliers will be used to
    recover pose. In the output mask only inliers which pass the cheirality check.

    This function differs from the one above that it computes camera intrinsic matrix from focal length and
    principal point:

    \f[A =
    \begin{bmatrix}
    f & 0 & x_{pp}  \\
    0 & f & y_{pp}  \\
    0 & 0 & 1
    \end{bmatrix}\f]
 */

//从上面的注释看，R和t分别是从1转换到2的变换矩阵。在后面的三角测量中，可以直接用
  recoverPose(essential_matrix, points_1, points_2, R, t, focal_length, principal_point);
  cout << "R is " << endl << R << endl;
  cout << "t is " << endl << t << endl;
}


Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void bundleAdjustmentGaussNewton(
                const VecVector3d &points_3d,
                const VecVector2d &points_2d,
                const Mat &K,
                Sophus::SE3d &pose) {
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    const int iterations = 10;
    
    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);
    double cx = K.at<double>(0,2);
    double cy = K.at<double>(1,2);

    double cost = 0;
    double lastcost = 0;

    for(int iter = 0; iter < iterations; iter++) {
        Eigen::Matrix<double,6,6> H = Eigen::Matrix<double,6,6>::Zero(); //高斯-牛顿法中的J*JT
        Vector6d b = Vector6d::Zero();
        cost = 0;
        for(int i = 0; i < points_3d.size();i++) {
            Eigen::Vector3d pc_3d = pose * points_3d[i]; //世界坐标系转换到相机坐标系
            Eigen::Matrix<double,2,6> J; //像素坐标系对相机位姿的雅克比矩阵的转置，实际上雅克比矩是（6行2列）====e(x+\delta x)=e(x)+JT(x)\delta x
            Eigen::Vector2d project_point(fx * pc_3d[0] / pc_3d[2] + cx,
                                            fy * pc_3d[1] / pc_3d[2] + cy); //相机坐标系转换到像素坐标系
            //Eigen::Vector2d e = points_2d[i] - project_point;//测量值减去投影值
            Eigen::Vector2d e = project_point - points_2d[i]; // 投影值减去测量值
            //更新cost
            cost += e.squaredNorm();

            if(pc_3d[2] == 0){
                cout << "The depth of 3D point in camera shouldn't be 0! Skip this point." << endl;
                continue;
            }
            double inv_z = 1.0 / pc_3d[2];
            double inv_z2 = inv_z * inv_z;

            J << fx * inv_z 
                , 0 
                , - fx * pc_3d[0] *inv_z2 
                , - fx * pc_3d[0] * pc_3d[1] * inv_z2
                , fx + fx * pc_3d[0] * pc_3d[0] * inv_z2
                , - fx * pc_3d[1] * inv_z
                , 0
                , fy * inv_z
                , - fy * pc_3d[1] * inv_z2
                , -fy - fy * pc_3d[1] * pc_3d[1] * inv_z2
                , fy * pc_3d[0] * pc_3d[1] * inv_z2
                , fy * pc_3d[0] * inv_z;

            //这里雅克比矩阵是6行2列的，2列是因为求解像素误差是按照行和列两个维度计算的，所以前面有一步骤是把这个中间计算结果再一次求开平方
            H += J.transpose() * J;
            b += -J.transpose() * e; 
        }
        //平移在前，旋转在后
        Vector6d delta_x;
        delta_x = H.ldlt().solve(b);
        if(isnan(delta_x[0])) {
            cout << "The increment is zero! Not normal." << endl;
            break;
        }

        if(iter >0 && cost >= lastcost) {
            cout << "This iteration cost [" << cost << "] is larger than last [" << lastcost << "]." << endl;
            break;
        }

        //上面的判断都不满足，说明是一次正常的迭代过程，需要对pose进行更新，使用Sophus的SE(3)方法
        pose = Sophus::SE3d::exp(delta_x) * pose;
        lastcost = cost;
        if(delta_x.norm() < 1e-6) {
            cout << "converge!"<< endl;
            break;
        }
        cout << "Iteration [" << iter << "] cost = [" << cost << "]." << endl;
    }

    cout << "Estimated pose using gauss-newton: \n" << pose.matrix() << endl;
}

//定义g2o下的节点和边，借助g2o来优化求相机的运动
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

class ProjectionEdge : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, PoseVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ProjectionEdge(const Eigen::Vector3d &pos,const Eigen::Matrix3d &K) : _pos3d(pos),_K(K){}

    virtual void computeError() override {
        const PoseVertex* v = static_cast<const PoseVertex *> (_vertices[0]);
        const Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pose_pixel = _K * (T * _pos3d); //未归一化的像素坐标系
        pose_pixel /= pose_pixel[2]; // 归一化像素坐标
        _error = pose_pixel.head<2>() - _measurement;//误差项，是一个2*1的向量.要注意，误差项是谁减去谁（这里是预测值减去观测值），要和下面的雅克比矩阵对应上。
    }

    //Jacobian -- 这里需要准备的数据是计算雅克比矩阵必须的。在这个case里面，我们需要的是相机坐标系下面的X,Y,Z,相机内参数
    virtual void linearizeOplus() override {
        const PoseVertex* v = static_cast<const PoseVertex *> (_vertices[0]);
        const Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam = T * _pos3d; //X',Y',Z'
        double fx = _K(0,0);
        double fy = _K(1,1);
        //double cx = _K(0,2);
        //double cy = _K(1,2);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z_2 = Z * Z;
        _jacobianOplusXi <<
            fx / Z, 0, -fx * X / Z_2, -fx * X * Y / Z_2, fx + fx * X * X / Z_2, -fx * Y / Z,
            0, fy / Z, -fy * Y / Z_2, -fy - fy * Y * Y / Z_2, fy * X * Y / Z_2, fy * X / Z;
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

public:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};

void bundleAdjustmentG2O(const VecVector3d &points_3d,const VecVector2d &points_2d,const Mat &K,Sophus::SE3d &pose) {
    // 构建图优化，先设定g2o
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

    K_eigen << 
            K.at<double>(0,0), K.at<double>(0,1),K.at<double>(0,2),
            K.at<double>(1,0), K.at<double>(1,1),K.at<double>(1,2),
            K.at<double>(2,0), K.at<double>(2,1),K.at<double>(2,2);

    //edges
    int index = 1;
    for (size_t i = 0; i< points_2d.size(); ++i) {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];

        ProjectionEdge *edge = new ProjectionEdge(p3d,K_eigen);
        edge->setId(index);
        edge->setVertex(0,vertex_pose);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    cout << "pose estimation by g2o = \n" << vertex_pose->estimate().matrix() << endl;
    pose = vertex_pose->estimate();
}

/* Template class for BA
/* 测量值：在前一个相机坐标系下的特征点坐标，在下一帧图像上测量到的这些特征点的像素坐标。相机内参数K是固定的。
/* operator()中，待优化的参数包含了平移旋转，ceres_rot是旋转，形式是轴角，
/* ceres_trans是平移
*/
struct ReprojectionError {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ReprojectionError(Eigen::Vector3d point,Eigen::Vector2d pixel,Eigen::Matrix3d K)
      : point_(point), pixel_(pixel),K_(K) {}

  template <typename T>
  bool operator()(const T* const ceres_rot, const T* const ceres_trans, T* residuals) const {
    //step1: 读取平移旋转的数据，转换成为Eigen的格式
    T p1[3];
    T p2[3];
    p1[0] = T(point_(0));
    p1[1] = T(point_(1));
    p1[2] = T(point_(2));
    //step2: 将世界坐标系下的点坐标转换为相机坐标系的坐标
    //旋转
    ceres::AngleAxisRotatePoint(ceres_rot, p1, p2);
    //平移
    p2[0] += ceres_trans[0];
    p2[1] += ceres_trans[1];
    p2[2] += ceres_trans[2];
    // cout << "After rotation and translation point: (" << p2[0] << ", "
    //                                         << p2[1] << ", "
    //                                         << p2[2] << ") " << endl;
    // cout << "K_ is: " << endl << K_ << endl; 
    //Step3: 计算投影到图像空间下的坐标
    Eigen::Vector<T,3> pixel_point;
    pixel_point(0) = K_(0,0) * p2[0] + K_(0,2) * p2[2];
    pixel_point(1) = K_(1,1) * p2[1] + K_(1,2) * p2[2];
    pixel_point(2) = K_(2,2) * p2[2];
    if(pixel_point(2)==0.0) {
        cout << "pixel z is 0." << endl;
        return false;
    }

    pixel_point(0) /= pixel_point(2); //x 
    pixel_point(1) /= pixel_point(2); //y
    //cout << "predicted pixel: (" << pixel_point(0) << "," << pixel_point(1) << ")" << endl;
    // The error is the difference between the predicted and observed position.
    residuals[0] = pixel_point(0) - pixel_[0];
    residuals[1] = pixel_point(1) - pixel_[1];
    //cout << "pixel residual: (" << residuals[0] << ", " << residuals[1] << endl;
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const Eigen::Vector3d point,
                                        const Eigen::Vector2d pixel,
                                        const Eigen::Matrix3d K) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3>(
                new ReprojectionError(point,pixel,K)));
  }


  Eigen::Vector3d point_;
  Eigen::Vector2d pixel_;
  Eigen::Matrix3d K_;
};


/** 
    @param points_3d 世界坐标系坐标集合
    @param points_2d 像素坐标系坐标
    @param K 相机外参数
    @param pose 相机1坐标系转换到相机2坐标系的SE3
 */
void bundleAdjustmentCeres(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose) {
    //create problem
    ceres::Problem problem;
    //loss function for filter out outliers
    ceres::LossFunction* loss_function = NULL;

    //transform Sophus::SE3d to ceres data array
    Eigen::Matrix4d init_pose= pose.matrix();
    cout << "init_pose: " << endl << init_pose << endl;
    Eigen::Isometry3d init_iospose(init_pose);
    Eigen::Vector<double, 6> vec = isometry2Params(init_iospose);
    cout << "vec: " << endl << vec.transpose() << endl;
    double ceres_rot[3] = {vec[0],vec[1],vec[2]};
    double ceres_trans[3] = {vec[3],vec[4],vec[5]};

    Eigen::Quaterniond from_se3(pose.rotationMatrix());
    //convert K from cv::Mat
    Eigen::Matrix3d K_eigen = Eigen::Matrix3d::Zero();
    K_eigen(0,0) = K.at<double>(0,0);
    K_eigen(1,1) = K.at<double>(1,1);
    K_eigen(0,2) = K.at<double>(0,2);
    K_eigen(1,2) = K.at<double>(1,2);
    K_eigen(2,2) = 1;
    
    cout << "measurement size is: " << points_2d.size() << endl;
    for(size_t i = 0; i< points_2d.size(); ++i) {
        ceres::CostFunction* cost_function =
        ReprojectionError::Create(points_3d[i],points_2d[i],K_eigen);//传递测量值到costFunction
        // cout << "Adding meaurement to ceres: point_3d: (" << points_3d[i].transpose()
        //         << "). point_2d: (" << points_2d[i].transpose()
        //         << "). Camera intrinsic matrix: " <<  endl << K_eigen << "[" << i << "]" << endl;
        problem.AddResidualBlock(cost_function,loss_function,ceres_rot,ceres_trans); //传递外参数
    }

    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    

    //extract result from ceres solver
    double ceres_results[6];
    ceres_results[0] = ceres_rot[0];
    ceres_results[1] = ceres_rot[1];
    ceres_results[2] = ceres_rot[2];

    ceres_results[3] = ceres_trans[3];
    ceres_results[4] = ceres_trans[4];
    ceres_results[5] = ceres_trans[5];


    Eigen::Isometry3d result = Eigen::params2Isometry(ceres_results);
    Eigen::Matrix3d rot_matrix = result.matrix().block<3,3>(0,0);
    Eigen::Matrix<double,3,1> trans_vec;
    trans_vec(0,0) = ceres_trans[0];
    trans_vec(1,0) = ceres_trans[1];
    trans_vec(2,0) = ceres_trans[2];

    cout << "Result translation: " << endl << trans_vec << endl;

    Sophus::SE3d result_sophus(rot_matrix,trans_vec);
    pose = result_sophus;
    // Eigen::Matrix3d rotation_result;
    // Eigen::Matrix3d rotation_result;
    // ceres::AngleAxisToRotationMatrix(ceres_rot,&rotation_result);
    // Eigen::Vector3d translation_result;
    // translation_result(0) = ceres_trans[0];
    // translation_result(1) = ceres_trans[1];
    // translation_result(2) = ceres_trans[2];
    // Sophus::SE3d updated(rotation_result,translation_result);
    // pose = updated * pose;
}


int main(int argc,char* argv[]) {
    // if(argc != 4) {
    //     cout << "Usage: pose_estimation_3d2d img1[First] img2[Second] img3[Depth]." << endl;
    //     return 1;
    // }

    //step1: 从硬盘读取图像
    Mat img_1 = imread(img_file1,cv::IMREAD_COLOR);
    Mat img_2 = imread(img_file2,cv::IMREAD_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);
    
    vector<DMatch> matches;
    vector<KeyPoint> keypoints_1, keypoints_2;
    //step2: 找到两张图像的特征匹配对
    find_feature_matches(img_1,img_2,keypoints_1,keypoints_2,matches);
    //step3: 读入深度图像
    Mat depth_image = imread(img_file3,IMREAD_UNCHANGED);
    //step4: 根据第一帧图像上的特征点位置，在深度图上找到该点的深度距离，并且通过相机的内参数反推导出相机归一化平面的x,y,再结合
    //深度生成3D点集合。
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for(DMatch m : matches) {
        ushort d = depth_image.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)]; // 根据特征点的y x（行 列）坐标找到深度图当中的深度信息
        if(d <= 0) {
            cout << "Depth should not be less than 0!" << endl;
            continue;
        }
        float dd = d / 5000.0;
        Point2d p_cam = pixel2cam(keypoints_1[m.queryIdx].pt,K); //这里需要先通过第一张图计算到世界坐标系的3D点坐标
        Point3f p_3d = Point3f(p_cam.x * dd , p_cam.y * dd, dd);
        Point2f p_cam_trans = Point2f(keypoints_2[m.trainIdx].pt);//pixel in second image
        pts_3d.push_back(p_3d); //3D coordinate in first camera
        pts_2d.push_back(p_cam_trans);
    }

    cout << "3D-2D pairs: " << pts_2d.size() << endl;

    //step5: 到此，3D和2D数据准备完成，后面就是使用不同的方法或者是不同的开源库做BA
    //通过高斯-牛顿BA求解最优的相机运动
    Sophus::SE3d pose_gn; //相机运动初始化时候，这个数值是I.这个值其实就是第一张图像拍摄时候相机的运动（我们定义的世界坐标系起始就在这个坐标系下面）
    cout << "Before gaussBewton BA, the pose is: " << endl << pose_gn.matrix() << endl;
    VecVector3d pts_3d_eigen;
    VecVector2d pts_2d_eigen;

    for(int i = 0; i < pts_2d.size(); i++) {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x,pts_3d[i].y,pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x,pts_2d[i].y));
    }
    bundleAdjustmentGaussNewton(pts_3d_eigen,pts_2d_eigen,K,pose_gn);

    Sophus::SE3d pose_g2o;
    bundleAdjustmentG2O(pts_3d_eigen,pts_2d_eigen,K,pose_g2o);
    cout << "Using g2o the camera pose estimation is: " << endl
            << pose_g2o.matrix() << endl;

    //ceres optimization
    Sophus::SE3d pose_ceres;
    bundleAdjustmentCeres(pts_3d_eigen,pts_2d_eigen,K,pose_ceres);
    cout << "Using ceres the camera pose estimation is: " << endl << pose_ceres.matrix() << endl;
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

