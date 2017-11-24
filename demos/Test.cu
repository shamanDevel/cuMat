#include <iostream>
#include <typeinfo>
#include <cuMat/Core>
#include <Eigen/Core>
#include <chrono>
#include <vector>

using namespace cuMat;
using namespace std;

int main(int argc, char* args[])
{
    std::vector<int> sizes = { 100, 1000, 10000, 100000, 1000000, 10000000 };
    for (int size : sizes) {
        //cuMat
        VectorXf v(size);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaDeviceSynchronize();
        cudaEventRecord(start, cuMat::Context::current().stream());
        //VectorXf r = v * 5;
        (v * 5).eval();
        cudaEventRecord(stop, cuMat::Context::current().stream());
        cudaEventSynchronize(stop);
        float elapsedCuMat;
        cudaEventElapsedTime(&elapsedCuMat, start, stop);
        //Eigen
        Eigen::VectorXf ve(size);
        auto startE = std::chrono::steady_clock::now();
        //Eigen::VectorXf re = ve * 5;
        (ve * 5).eval();
        auto stopE = std::chrono::steady_clock::now();
        double elapsedEigen = std::chrono::duration_cast<std::chrono::duration<double>>(stopE - startE).count() * 1000;

        cout << "Size: " << size << endl;
        cout << "Time cuMat: " << elapsedCuMat << " ms" << endl;
        cout << "Time Eigen: " << elapsedEigen << " ms" << endl;
    }
}