#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <cmath>
#include <chrono>
#include <opencv2/opencv.hpp>


using namespace std;
/*
* Note: After compiling and installing source code of g2o, CMake can't find the G2O.
* We need copy-paste cmake_modules/FindG2O.cmake to /usr/share/cmake-X.X/Modules/ . So
* that CMake could find the G2O module by find_package(G2O REQUIRED)
*/
//Vertex model: 3 params: a, b, c 
class CurveFittingVertex : public g2o::BaseVertex<3,Eigen::Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override {
        _estimate << 0, 0, 0;
    }

    virtual void oplusImpl(const double *update) override {
        _estimate += Eigen::Vector3d(update);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

    virtual void computeError() override {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex *> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();

        _error(0,0) = _measurement - std::exp(abc(0,0) * _x * _x + abc(1,0) * _x + abc(2,0));
    }

    //Jacobian
    virtual void linearizeOplus() override {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex *> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);

        _jacobianOplusXi[0] = - _x * _x * y;
        _jacobianOplusXi[1] = - _x  * y;
        _jacobianOplusXi[2] = - y;

    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

public:
double _x;
};



int main(int argc,char* argv[]) {
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

    double abc[3] = {ae,be,ce};


    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> BlockSolverType; //for each error, we optimize 3 params and the error dimension is 1
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; //Linear solver type

    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
                        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));

    g2o::SparseOptimizer optimizer; 
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    CurveFittingVertex* v = new CurveFittingVertex(); // construct the CurveFittingVertex with 3 params and 1 error
    v->setEstimate(Eigen::Vector3d(ae,be,ce));
    v->setId(0);
    optimizer.addVertex(v);


    for(size_t i =0; i < N; i++) {
        CurveFittingEdge* edge = new CurveFittingEdge(x_data[i]);// construct with the x_data
        edge->setId(i);
        edge->setVertex(0,v);
        edge->setMeasurement(y_data[i]);
        edge->setInformation(Eigen::Matrix<double,1,1>::Identity() * inv_w_sigma * inv_w_sigma); //information matrix
        optimizer.addEdge(edge);
    }


    //do optimization
    cout << "Start optimization!";
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "Solve time cost: " << time_used.count() << " seconds." << endl;

    //output result
    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "Estimated model: " << abc_estimate.transpose() <<endl;

    return 0;
}