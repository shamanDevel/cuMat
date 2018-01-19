#include <iostream>
#include <cuMat/Core>

using namespace cuMat;
using namespace std;

int main()
{
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
    cout << "Full reduction: " << mat.sum<ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch>().eval() << endl;
    cout << "dynamic version: " << mat.sum(ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch).eval() << endl;
    cout << "Along rows: " << mat.sum<ReductionAxis::Row>().eval() << endl;
    cout << "Along columns: " << mat.sum<ReductionAxis::Column>().eval() << endl;
    cout << "Along batches: " << mat.sum<ReductionAxis::Batch>().eval() << endl;
    cout << "Along rows and columns: " << mat.sum<ReductionAxis::Row | ReductionAxis::Column>().eval() << endl;
}
