/*
 * First, simple demo
 */

#include <iostream>
#include <typeinfo>
#include <cuMat/Core>

using namespace cuMat;
using namespace std;

int main(int argc, char* args[])
{
    //Shows different transposition modes

    //Input matrices
    MatrixXiR m1(5, 6);
    SimpleRandom r;
    r.fillUniform(m1, 0, 20);

    cout << "Input matrix 1:" << endl << m1 << endl;

    //no-op transposition
    auto eval1 = m1.transpose().eval(); //force evaluation in the best possible way
    cout << "eval1: " << typeid(eval1).name() << endl;
    cout << eval1 << endl;

    //transposition using BLAS
    MatrixXiR eval2 = m1.transpose(); //force evaluation in specific shape (same storage order)
    cout << "eval2: " << typeid(eval2).name() << endl;
    cout << eval2 << endl;

    //component-wise evaluation
    auto op3 = (m1 * 2).transpose(); //as soon as one cwise-op is in the tree, cwise transposition is used
        //nothing is evaluated yet
    auto eval3 = op3.eval(); //force evaluation in the best possible way
    cout << "eval3: " << typeid(eval3).name() << endl;
    cout << eval3 << endl;

}