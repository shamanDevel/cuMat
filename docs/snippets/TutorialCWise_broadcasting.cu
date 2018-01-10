#include <iostream>
#include <cuMat/Core>

using namespace cuMat;

int main()
{
    //create a 2x4 matrix
    MatrixXiR m = MatrixXi::fromArray({
        {
            {1, 2, 6, 9},
            {3, 1, 7, 2}
        }
    });

    //create a 2-dim column vector
    VectorXiR v = VectorXi::fromArray({
        {
            {0},
            {1}
        }
    });

    MatrixXiR result = m + v;
    std::cout << "Broadcasting result:" << std::endl;
    std::cout << result << std::endl;;
}