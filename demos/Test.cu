#include <iostream>
#include <cuMat/Core>

using namespace cuMat;
using namespace std;

// warnings occur for BATCHES=8..45
// Origin is ReductionOps.h, ReductionEvaluator.eval(), ~line 203
#ifndef BATCHES
#define BATCHES 32
#endif

int main()
{
    std::cout << "Batches: " << BATCHES << "\n";
    using MyVec = cuMat::Matrix<float, cuMat::Dynamic, 1, BATCHES, cuMat::ColumnMajor>;
    MyVec vec = MyVec::Zero(100000, 1, BATCHES);
    auto test = vec.sum<cuMat::Axis::Row | cuMat::Axis::Column>().eval();
    return 0;
	
	/*
    double data[2][2][2] {
        {
            {1, 2},
            {3, 4}
        },
        {
            {5, 6},
            {7, 8}
        }
    };
    BMatrixXdR mat = BMatrixXdR::fromArray(data);
    cout << "Input matrix: " << mat << endl;
    cout << "Full reduction: " << mat.sum<Axis::Row | Axis::Column | Axis::Batch>().eval() << endl;
    cout << " dynamic version: " << mat.sum(Axis::Row | Axis::Column | Axis::Batch).eval() << endl;
    cout << "Along rows: " << mat.sum<Axis::Row>().eval() << endl;
    cout << "Along columns: " << mat.sum<Axis::Column>().eval() << endl;
    cout << "Along batches: " << mat.sum<Axis::Batch>().eval() << endl;
    cout << "Along rows and columns: " << mat.sum<Axis::Row | Axis::Column>().eval() << endl;
    */

    /*
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
    std::cout << result << std::endl;
    */
}
