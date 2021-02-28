#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <ctime>

using namespace std;
using namespace Eigen;


#define MATRIXSIZE 50


int main(int argc,char* argv[]){

    Matrix<float,2,3> matrix_23;
    matrix_23 << 1,2,3,4,5,6;

    cout << "matrix_23 from 1 to 6 is: \n" << matrix_23 <<endl;

    Vector3d v_3d;
    v_3d << 11,12,13;
    cout << "v_3d from 1 to 3 is: \n" << v_3d << endl;

    //initalize with all zeros

    Matrix3d matrix_33_zeros = Matrix3d::Zero();
    cout << "matrix_33_zeros from 1 to 9 is: \n" << matrix_33_zeros << endl;


    //if not sure the size of matrix, dynamic matrix can be used
    Matrix<double,Dynamic,Dynamic> matrix_dynamic;


    //operations of Matrix

    //use () to access elements from one Matrix
    cout << "print matrix_23: " << endl;
    for(int i=0;i<2;i++){
        for(int j=0;j<3;j++){
            cout << matrix_23(i,j) << "\t";
        }
        cout << endl;
    }

    //multiplication of matrix and vector
    Matrix<double,2,1> result_21 = matrix_23.cast<double>() * v_3d;
    cout << "[1,2,3;4,5,6] * [11,12,13]= " << result_21 << endl;

    //make sure correct dimesion of a matrix is given 
    //Matrix<double,2,3> wrong_demesion = matrix_23.cast<double>() * v_3d;

    //random matrix
    Matrix3d matrix_33_random = Matrix3d::Random();
    cout << "random 3x3 matrix is: \n" << matrix_33_random << endl;

    cout << "Transpose of the matrix is: \n" << matrix_33_random.transpose() << endl;

    cout << "sum of the elements is: \n" << matrix_33_random.sum() << endl;

    cout << "Trace of the matrix is: \n" << matrix_33_random.trace() << endl;

    cout << "Times 10: \n" << 10 * matrix_33_random <<endl;

    cout << "Inverse: \n" << matrix_33_random.inverse() << endl;

    cout << "det: \n" << matrix_33_random.determinant() << endl;



    //eigenvalues
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33_random.transpose()*matrix_33_random);

    cout << "Eigen values: \n" << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vectors: \n" << eigen_solver.eigenvectors() << endl;

    //solve equations in matrix form 
    Matrix<double,MATRIXSIZE,MATRIXSIZE> matrix_NN = MatrixXd::Random(MATRIXSIZE,MATRIXSIZE);
    matrix_NN = matrix_NN * matrix_NN.transpose(); //to ensure the matrix to be  semi-positive definite
    Matrix<double,MATRIXSIZE,1> v_Nd = MatrixXd::Random(MATRIXSIZE,1); //build the vector

    clock_t time_stt = clock();

    //inverse directly
    Matrix<double,MATRIXSIZE,1> x= matrix_NN.inverse() * v_Nd;
    cout << "Time of normal inverse is: " << 1000*(clock()-time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x= " << x.transpose() << endl;

    //QR decomposition to speed up 
    time_stt = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    cout << "Time of QR decomposition is: " << 1000*(clock()-time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x= " << x.transpose() << endl;

    //if the matrix is semi-positive definite, cholesky method can be also used
    time_stt = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    cout << "Time of ldlt decomposition is: " << 1000*(clock()-time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
    cout << "x= " << x.transpose() << endl;

    
    return 0;
}

