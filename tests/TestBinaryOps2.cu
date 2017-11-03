#include <catch/catch.hpp>
#include <vector>

#include <cuMat/Core>

#include "Utils.h"
#include "TestBinaryOps.cuh"

// Test for broadcasting
// TODO

using namespace cuMat;

template<int _Rows, int _Cols, int _Batches>
Matrix<int, (_Rows>1) ? Dynamic : _Rows, (_Cols>1) ? Dynamic : _Cols, (_Batches>1) ? Dynamic : _Batches, ColumnMajor>
fromArray(int a[_Batches][_Cols][_Rows])
{
    typedef Matrix<int, (_Rows > 1) ? Dynamic : _Rows, (_Cols > 1) ? Dynamic : _Cols, (_Batches > 1) ? Dynamic : _Batches, ColumnMajor> mt;
    mt m(_Rows, _Cols, _Batches);
    m.copyFromHost(a);
    return m;
}

TEST_CASE("fromArray", "[binary]")
{
    
}
