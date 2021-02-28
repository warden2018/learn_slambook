#include <iostream>
#include <cmath>

using namespace std;
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;



//This program demostrates Eigen geometry transform 


int main(int argc,char* argv[]){

    //Eigen/Geometry module provides all kinds of different translation and rotation expressions 

    //use Matrix3d or Matrix3f to be used as the rotation matrix
    Matrix3d rotation_matrix = Matrix3d::Identity();
    //rotation vector. rotate PI/4 aroud z axis
    AngleAxisd rotation_vector(M_PI/4,(Vector3d(0,0,1)));
    cout.precision(3);
    //the rotation vector can be expressed as the rotation matrix by calling matrix() method
    cout << "Rotation matrix =\n" << rotation_vector.matrix() << endl;
    rotation_matrix = rotation_vector.toRotationMatrix();
    //use AngleAxis to perform transform
    Vector3d v(1,0,0); // unit vector same direction with the x axis
    //rotate the vector by using the rotation vector
    Vector3d v_rotated = rotation_vector * v;
    cout << "vector \n[" << v << "] \n after rotation(by angle axis) is: \n[" << v_rotated <<"]"<< endl;
    //rotate the vector by using the rotation matrix
    v_rotated = rotation_matrix * v;
    cout << "vector \n[" << v << "] \n after rotation(by matrix) is: \n[" << v_rotated << "]"<< endl;
    
    //euler angles. From rotation matrix to eular angles
    Vector3d eular_angles = rotation_matrix.eulerAngles(2,1,0); // ZYX order. yaw, pitch and roll
    cout << "yaw, pitch and roll = " << eular_angles.transpose() << endl;

    //Euclidean translform use Eigen::Isometry
    Isometry3d T = Isometry3d::Identity(); // 4x4 matrix
    T.rotate(rotation_vector); // rotate as the rotation_vector
    T.pretranslate(Vector3d(1,3,4)); // set translation vector as [1,3,4]
    cout << "Transform matrix = \n" << T.matrix() << endl;

    //use T to transform the vector v
    Vector3d transformed_v = T * v;
    cout << "v transformed = \n [" << transformed_v << "]" << endl;

    //for Affine transformation and projective transformation; use Affine3d and projective3d respectively.

    //quaternion
    //the rotation vector can be the param for initialize the quaternion object
    Quaterniond q = Quaterniond(rotation_vector);
    cout << "Quaternion(x,y,z,w) from rotation vector is: \n[" << q.coeffs() << "]" << endl;
    //Be careful about the order of coeffs: (x,y,z,w). the real part is w!
    
    //can be also initialize with rotation matrix
    q = Quaterniond(rotation_matrix);
    cout << "Quaternion(x,y,z,w) from rotation matrix is: \n[" << q.coeffs() << "]" << endl;
    //Be careful about the order of coeffs: (x,y,z,w). the real part is w!

    //use quaternion to rotate the vector v (overide the multiple function)
    v_rotated = q * v;
    cout << "(1,0,0) after rotation by quaternion = \n [" << v_rotated << "]" << endl;


    //if use the normal quaternion 
    /*Eigen::Quaternion< _Scalar, _Options >::Quaternion (	
        const Scalar & 	w,
        const Scalar & 	x,
        const Scalar & 	y,
        const Scalar & 	z )
    */	
    cout << "should be equal to [" << (q * Quaterniond(0,1,0,0) * q.inverse()).coeffs() << "]" << endl;
    //caution again for the order of coeffs: x,y,z,w !!!!!

    //The exercise after:
    //In world coordinate: frame 1 q1=[0.35,0.2,0.3,0.1] t1=[0.3,0.1,0.1]
    //frame 2 q2=[-0.5,0.4,-0.1,0.2], t2=[-0.1,0.5,0.3]
    //in frame 1, a point's coordinate is: [0.5,0,0.2]
    //question: what's the point coordinate in frame 2?

    Quaterniond q1(0.35,0.2,0.3,0.1),q2(-0.5,0.4,-0.1,0.2);
    q1.normalize();
    q2.normalize();

    Vector3d p1(0.5,0,0.2);
    Vector3d t1(0.3,0.1,0.1);
    Vector3d t2(-0.1,0.5,0.3);

    Isometry3d T1w(q1),T2w(q2); 
    T1w.pretranslate(t1);
    T2w.pretranslate(t2);

    // T1w * Pw = Pr1, T2w * Pw = Pr2
    // Pw = T1w(-1) * Pr1 -> T2w * T1w(-1) * Pr1= Pr2

    Vector3d p2 = T2w * T1w.inverse() * p1;
    cout << "point in frame 2 coordinate is: \n[" << p2 << "]" << endl;

    return 0;
}