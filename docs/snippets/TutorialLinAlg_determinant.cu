#include <iostream>
#include <cuMat/Dense>

using namespace cuMat;
using namespace std;

int main()
{
    float data[1][3][3]{{
        { 6, 4, 1 },
        { 4, 5, 1 },
        { 1, 1, 6 }
    }};
    MatrixXfR A = MatrixXfR::fromArray(data);
    cout << "Here is the matrix A:\n" << A << endl;
    cout << "The determinant of A is " << static_cast<float>(A.determinant().eval()) << endl;
    cout << "The log-determinant of A is " << static_cast<float>(A.logDeterminant().eval()) << endl;
}