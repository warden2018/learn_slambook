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

// 传递引用作为函数的参数，免去拷贝构造新对象，这样可以避免由于构造对象太大所花费太多的时间
void triangulation(
                const vector<KeyPoint> &keypoint_1,
                const vector<KeyPoint> &keypoint_2,
                const vector<DMatch> &matches,
                const Mat &R, const Mat &t,
                vector<Point3d> &points
);

inline cv::Scalar get_color(float depth) {
  float up_th = 20.0, low_th = 6.0, th_range = up_th - low_th;
  if (depth > up_th) depth = up_th;
  if (depth < low_th) depth = low_th;
  return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

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

    //三角测量得到特征点的世界坐标（相机移动之前的坐标是世界坐标系的原点）
    vector<Point3d> points;
    triangulation(keypoints_1,keypoints_2,matches,R,t,points);

    //-- 验证E=t^R*scale
    // Mat t_x =
    //     (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
    //     t.at<double>(2, 0), 0, -t.at<double>(0, 0),
    //     -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    // cout << "t^R=" << endl << t_x * R << endl;

    //  //-- 验证对极约束
    // Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    // for (DMatch m: matches) {
    //     Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
    //     Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1); //把像素坐标转换成归一化相机坐标，代入y2t^Ry1=0,检查等式左侧结果与0的接近程度。如果很小，说明估计效果良好
    //     Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
    //     Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
    //     Mat d = y2.t() * t_x * R * y1;
    //     cout << "epipolar constraint = " << d << endl;
    // }

    //验证三角测量结果重投影到图像上
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    Mat img1_plot = img_1.clone();
    Mat img2_plot = img_2.clone();
    for (int i = 0; i < matches.size(); i++) {
        // 第一个图
        float depth1 = points[i].z;
        cout << "depth: " << depth1 << endl;
        Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

        // 第二个图
        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        float depth2 = pt2_trans.at<double>(2, 0);
        cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
    }
    cv::imshow("img 1", img1_plot);
    cv::imshow("img 2", img2_plot);
    cv::waitKey();
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
    for(int i = 0; i < descriptors_1.rows;i++) {
        if(raw_matches[i].distance < dis_min) {
            dis_min = raw_matches[i].distance;
        }
        if(raw_matches[i].distance > dis_max) {
            dis_max = raw_matches[i].distance;
        }
    }

    printf("-- Max dist : %f \n",dis_max);
    printf("-- Min dist : %f \n",dis_min);

    for(int i = 0;i < descriptors_1.rows;i++) {
        if(raw_matches[i].distance <= max(2* dis_min,30.0)) {
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
    for(int i = 0; i <(int) matches.size(); i++) {
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

void triangulation(
                const vector<KeyPoint> &keypoint_1,
                const vector<KeyPoint> &keypoint_2,
                const vector<DMatch> &matches,
                const Mat &R, const Mat &t,
                vector<Point3d> &points) {
  Mat T1 = (Mat_<float>(3,4) << 
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0); //这里的3*4 矩阵代表了三维空间内的一点其次坐标投影到相机坐标系下的坐标。相机坐标和世界坐标系重合
  Mat T2 = (Mat_<float>(3,4) << 
        R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),t.at<double>(0,0),
        R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),t.at<double>(1,0),
        R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2),t.at<double>(2,0)); // 传递过来的R和t相当于R21和t21，就是在2坐标系下的坐标点X2=[R21|t21]X1得到

  Mat K = (Mat_<double>(3,3) << 520.9, 0,325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point2f> pts_1,pts_2;

  for(DMatch m : matches) {
      pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt,K));
      pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt,K));
  }
  Mat pts_4d;
  /** @brief This function reconstructs 3-dimensional points (in homogeneous coordinates) by using
    their observations with a stereo camera.

    @param projMatr1 3x4 projection matrix of the first camera, i.e. this matrix projects 3D points
    given in the world's coordinate system into the first image.
    @param projMatr2 3x4 projection matrix of the second camera, i.e. this matrix projects 3D points
    given in the world's coordinate system into the second image.
    @param projPoints1 2xN array of feature points in the first image. In the case of the c++ version,
    it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
    @param projPoints2 2xN array of corresponding points in the second image. In the case of the c++
    version, it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
    @param points4D 4xN array of reconstructed points in homogeneous coordinates. These points are
    returned in the world's coordinate system.

    @note
    Keep in mind that all input data should be of float type in order for this function to work.

    @note
    If the projection matrices from @ref stereoRectify are used, then the returned points are
    represented in the first camera's rectified coordinate system.

    @sa
    reprojectImageTo3D
 */
  cv::triangulatePoints(T1,T2,pts_1,pts_2,pts_4d); // 计算结果存储形式是按列排列的齐次坐标

  for(int i = 0; i < pts_4d.cols; i++) {
      Mat x = pts_4d.col(i);
      x /= x.at<float>(3,0);

      Point3d p (
          x.at<float>(0,0),
          x.at<float>(1,0),
          x.at<float>(2,0)
      );

      points.push_back(p);
  }
}