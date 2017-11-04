#include <catch/catch.hpp>
#include <set>

#include <cuMat/src/Iterator.h>
#include "cuMat/src/Matrix.h"

using namespace cuMat;

template <typename _Derived>
void TestIndexMath(const MatrixBase<_Derived>& mat, std::array<Index, 3> stride)
{
    typedef StridedMatrixIterator<_Derived> Iterator;
    std::array<Index, 3> dims { mat.rows(), mat.cols(), mat.batches() };
    INFO("Dims: " << dims[0]<<","<<dims[1]<<","<<dims[2] << "; Stride: " << stride[0]<<","<<stride[1]<<","<<stride[2]);
    Index size = dims[0] * dims[1] * dims[2];
    std::set<Index> indices;
    for (Index r=0; r<dims[0]; ++r)
    {
        for (Index c=0; c<dims[1]; ++c)
        {
            for (Index b=0; b<dims[2]; ++b)
            {
                Index i = Iterator::toLinear({ r, c, b }, stride);
                REQUIRE(i >= 0);
                REQUIRE(i < size);
                indices.insert(i);
                std::array<Index, 3> coords = Iterator::fromLinear(i, dims, stride);
                REQUIRE(r == coords[0]);
                REQUIRE(c == coords[1]);
                REQUIRE(b == coords[2]);
            }
        }
    }
    REQUIRE(indices.size() == size);
}
template <typename _Derived>
void TestIndexMath(const MatrixBase<_Derived>& mat)
{
    Index rows = mat.rows();
    Index cols = mat.cols();
    Index batches = mat.batches();
    TestIndexMath(mat, { 1, rows, rows*cols });
    TestIndexMath(mat, { 1, rows*batches, rows });
    TestIndexMath(mat, { cols, 1, rows*cols });
    TestIndexMath(mat, { cols*batches, 1, cols });
    TestIndexMath(mat, { batches, batches*rows, 1 });
    TestIndexMath(mat, { batches*cols, batches, 1 });
}

TEST_CASE("index-math", "[iterator]")
{
    TestIndexMath(BMatrixXfR(5, 6, 7));
    TestIndexMath(BMatrixXfC(5, 6, 7));
}

