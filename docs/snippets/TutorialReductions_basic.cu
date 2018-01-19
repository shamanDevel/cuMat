#include <iostream>
#include <cuMat/Core>

using namespace cuMat;
using namespace std;

int main()
{
    double data[1][2][2] {
        {
            {1, 2},
            {3, 4}
        }
    };
    MatrixXdR mat = MatrixXdR::fromArray(data);
    //The reductions return a Matrix of static size 1x1x1, but still on GPU memory.
    //For these matrices, there exists an explicit conversion operator to convert 
    //  them to the CPU scalars, so that they can be used in regular C++ code.
    cout << "Here is mat.sum():       " << static_cast<double>(mat.sum()) << endl;
    cout << "Here is mat.prod():      " << static_cast<double>(mat.prod()) << endl;
    cout << "Here is mat.minCoeff():  " << static_cast<double>(mat.minCoeff()) << endl;
    cout << "Here is mat.maxCoeff():  " << static_cast<double>(mat.maxCoeff()) << endl;
    cout << "Are all elements > 2?    " << static_cast<bool>((mat > 2).all()) << endl;
    cout << "Is any element > 2?      " << static_cast<bool>((mat > 2).any()) << endl;
}
