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
    MatrixXiR m2(5, 6);
    SimpleRandom r;
    r.fillUniform(m1, 0, 20);
    r.fillUniform(m2, 0, 20);

    cout << "Input matrix 1:" << endl << m1 << endl;

    //no-op transposition
    auto op1 = m1.transpose(); //up to now, nothing is done
    cout << "op1: " << typeid(op1).name() << endl;
    auto eval1 = op1.eval(); //force evaluation in the best possible way
    cout << "eval1: " << typeid(eval1).name() << endl;
    cout << "Transposition 1:" << endl << eval1 << endl;

    //transposition using BLAS
    MatrixXiR eval2 = m1.transpose(); //force evaluation in specific shape (same storage order)
    cout << "eval2: " << typeid(eval2).name() << endl;
    cout << "Transposition 2:" << endl << eval2 << endl;

    //component-wise evaluation
    auto op3 = (m1 + m2).transpose(); //as soon as one cwise-op is in the tree, cwise transposition is used
    cout << "op2: " << typeid(op3).name() << endl;
    auto eval3 = op3.eval(); //force evaluation in the best possible way
    cout << "eval3: " << typeid(eval3).name() << endl;
    cout << "Transposition 3:" << endl << eval3 << endl;

}