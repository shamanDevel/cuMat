#include <iostream>
#include <typeinfo>
#include <cuMat/Core>
#include <Eigen/Core>
#include <chrono>
#include <vector>

#include <iostream>
#include <cuMat/Core>

int main()
{
    cuMat::BMatrixXf mat(3, 3, 1);
    cuMat::SimpleRandom rand(1);
    rand.fillUniform(mat, 0.0f, 1.0f);
    cuMat::BMatrixXf result = mat * 2;
    std::cout << result << std::endl;

    /*
     * Example output:
Matrix: rows=3 (dynamic), cols=3 (dynamic), batches=1 (dynamic), storage=Column-Major
0.580378  1.35463  1.41331
 1.88441 0.766545  1.83773
0.696211 0.899099 0.773809
     */
}