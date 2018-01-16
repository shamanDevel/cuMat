
#include <iostream>
#include <cuMat/Core>

using namespace cuMat;

int main()
{
    //create a 2x4 matrix
    int data1[1][2][4] {
        {
            {1, 2, 6, 9},
            {3, 1, 7, 2}
        }
    };
    MatrixXiR m = MatrixXiR::fromArray(data1);

    //create a 2-dim column vector
    int data2[1][2][1] {
        {
            {0},
            {1}
        }
    };
    VectorXiR v = VectorXiR::fromArray(data2);

    MatrixXiR result = m + v;
    std::cout << "Broadcasting result:" << std::endl;
    std::cout << result << std::endl;;
}
