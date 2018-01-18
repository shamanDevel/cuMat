#include <iostream>
#include <cuMat/Core>

using namespace cuMat;

int main()
{
    //create a 6x6 matrix with zeros
    MatrixXi m(6, 6);
    m.setZero();

    //set the top right and bottom left corners to identity matrices
    m.block<3, 3, 1>(3, 0, 0) = MatrixXi::Identity(3, 3, 1); //static version
    m.block(0, 3, 0, 3, 3, 1) = MatrixXi::Identity(3, 3, 1); //dynamic version

    //result:
    std::cout << m << std::endl;
}
