#include <iostream>
#include <cuMat/Core>

int main()
{
    cuMat::BMatrixXf mat(3, 3, 1);
    cuMat::SimpleRandom rand(1);
    rand.fillUniform(mat, 0.0f, 1.0f);
    cuMat::BMatrixXf result = mat * 2;
    std::cout << result << std::endl;
}