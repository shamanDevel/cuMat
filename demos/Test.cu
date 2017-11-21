#include <iostream>
#include <typeinfo>
#include <cuMat/Core>
#include <Eigen/Core>
#include <chrono>

using namespace cuMat;
using namespace std;

int main(int argc, char* args[])
{
    int size = 100000;
    //cuMat
    VectorXf v(size);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start, cuMat::Context::current().stream());
    VectorXf r = v * 5;
    cudaEventRecord(stop, cuMat::Context::current().stream());
    cudaEventSynchronize(stop);
    float elapsedCuMat;
    cudaEventElapsedTime(&elapsedCuMat, start, stop);
    //Eigen
    Eigen::VectorXf ve(size);
    auto startE = std::chrono::steady_clock::now();
    Eigen::VectorXf re = ve * 5;
    auto stopE = std::chrono::steady_clock::now();
    double elapsedEigen = std::chrono::duration_cast<std::chrono::duration<double> >(stopE - startE).count() * 1000;

    cout << "Time cuMat: " << elapsedCuMat << " ms" << endl;
    cout << "Time Eigen: " << elapsedEigen << " ms" << endl;
}