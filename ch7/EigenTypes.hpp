
/* Yang added in 20210927
 *
 * This file is used by ceres solver. 
 */

#ifndef CH7_EIGENTYPES_HPP
#define CH7_EIGENTYPES_HPP

#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace Eigen
{
using MatrixXb   = Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXub  = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic>;
using MatrixXbr  = Eigen::Matrix<int8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXubr = Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXui  = Eigen::Matrix<uint32_t, Eigen::Dynamic, Eigen::Dynamic>;

template <class T, int32_t Rows>
using Vector = Eigen::Matrix<T, Rows, 1>;

template <class T>
using Isometry3 = Transform<T, 3, Isometry>;



////////////////////////////////////////////////////////////////////////////////////////////////

/// Produces a minimal representation of a 3d rigid pose that can be used
/// for non-linear optimziation
template <class T>
Vector<T, 6> isometry2Params(const Eigen::Isometry3<T>& pose)
{
    Vector<T, 6> params;
    Matrix<T, 3, 3> R = fromEigen(pose.linear().template cast<T>());
    ceres::RotationMatrixToAngleAxis(ceres::ColumnMajorAdapter3x3(const_cast<const T*>(R.data())), params.data());
    params[3] = pose.translation()[0];
    params[4] = pose.translation()[1];
    params[5] = pose.translation()[2];
    return params;
}

/// Expans the minimal representation of a 3d rigid pose so that it can be
/// used for transforming points.
/// params is a 6 element vector
template <class T>
Eigen::Isometry3<T> params2Isometry(T const* const params)
{
    Eigen::Isometry3<T> pose;

    Matrix<T, 3, 3> R;
    ceres::AngleAxisToRotationMatrix(params, ceres::ColumnMajorAdapter3x3(R.data()));
    pose.linear()      = toEigen(R);
    pose.translation() = Eigen::Vector<T, 3>(params[3], params[4], params[5]);
    pose.makeAffine();

    return pose;
}

Eigen::Isometry3d params2Isometry(double const* const params)
{
    Eigen::Isometry3d pose;

    Matrix<double, 3, 3> R;
    ceres::AngleAxisToRotationMatrix(params, ceres::ColumnMajorAdapter3x3(R.data()));
    pose.linear()      = R;
    pose.translation() = Eigen::Vector<double, 3>(params[3], params[4], params[5]);
    pose.makeAffine();

    return pose;
}



Vector<double, 6> isometry2Params(const Eigen::Isometry3<double>& pose)
{
    Vector<double, 6> params;
    Matrix<double, 3, 3> R = pose.linear().template cast<double>();
    ceres::RotationMatrixToAngleAxis(ceres::ColumnMajorAdapter3x3(const_cast<const double*>(R.data())), params.data());
    params[3] = pose.translation()[0];
    params[4] = pose.translation()[1];
    params[5] = pose.translation()[2];
    return params;
}

}

#endif
