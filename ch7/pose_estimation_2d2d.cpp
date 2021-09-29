#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;



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

int main(int argc,char* argv[]) {
    if(argc != 3) {
        cout << "Usage: feature_extraction img1 img2." << endl;
        return 1;
    }

    //read image from disk
    Mat img_1 = imread(argv[1],cv::IMREAD_COLOR);
    Mat img_2 = imread(argv[2],cv::IMREAD_COLOR);
    assert(img_1.data != nullptr && img_2.data != nullptr);
    
    vector<DMatch> matches;
    vector<KeyPoint> keypoints_1, keypoints_2;
    //找到两张图像的特征匹配对
    find_feature_matches(img_1,img_2,keypoints_1,keypoints_2,matches);
    //计算基础矩阵和本质矩阵
    Mat R, t;
    pose_estimation_2d2d(keypoints_1,keypoints_2,matches,R,t);

    //-- 验证E=t^R*scale
    Mat t_x =
        (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
        t.at<double>(2, 0), 0, -t.at<double>(0, 0),
        -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    cout << "t^R=" << endl << t_x * R << endl;

     //-- 验证对极约束
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (DMatch m: matches) {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1); //把像素坐标转换成归一化相机坐标，代入y2t^Ry1=0,检查等式左侧结果与0的接近程度。如果很小，说明估计效果良好
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
        cout << "epipolar constraint = " << d << endl;
    }
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