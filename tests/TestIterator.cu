#include <catch/catch.hpp>
#include <set>
#include <vector>

#include <cuMat/Core>

using namespace cuMat;

template <typename _Derived>
void TestIndexMath(const MatrixBase<_Derived>& mat, thrust::tuple<Index, Index, Index> stride)
{
    typedef StridedMatrixInputIterator<_Derived> Iterator;
    typedef thrust::tuple<Index, Index, Index> Index3;
    Index3 dims { mat.rows(), mat.cols(), mat.batches() };
    INFO("Dims: " << dims.get<0>()<<","<<dims.get<1>()<<","<<dims.get<2>() << "; Stride: " << stride.get<0>()<<","<<stride.get<1>()<<","<<stride.get<2>());
    Index size = dims.get<0>() * dims.get<1>() * dims.get<2>();
    std::set<Index> indices;
    for (Index r=0; r<dims.get<0>(); ++r)
    {
        for (Index c=0; c<dims.get<1>(); ++c)
        {
            for (Index b=0; b<dims.get<2>(); ++b)
            {
                Index i = Iterator::toLinear({ r, c, b }, stride);
                REQUIRE(i >= 0);
                REQUIRE(i < size);
                indices.insert(i);
                Index3 coords = Iterator::fromLinear(i, dims, stride);
                REQUIRE(r == coords.get<0>());
                REQUIRE(c == coords.get<1>());
                REQUIRE(b == coords.get<2>());
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

//This also tests if the iterator is copyable
template <typename _Iter>
__global__ void FillWithIteratorKernel(_Iter iter, int* out)
{
    _Iter copy = iter;
    ++iter;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int val = copy[idx];
    out[idx] = val;
}
TEST_CASE("iterator-read", "[iterator]")
{
    int data[2][4][3] = {
        {
            { 1, 2, 3 },
            { 4, 5, 6 },
            { 7, 8, 9 },
            { 10,11,12 }
        },
        {
            { 13,14,15 },
            { 16,17,18 },
            { 19,20,21 },
            { 22,23,24 }
        }
    };
    auto m = BMatrixXiR::fromArray(data);
    StridedMatrixInputIterator<BMatrixXiR> iter(m, thrust::make_tuple(3, 1, 12));
    DevicePointer<int> outMem(24);
    CUMAT_SAFE_CALL(cudaMemset(outMem.pointer(), 0, sizeof(int) * 24));
    FillWithIteratorKernel<<<24, 1>>>(iter, outMem.pointer());
    CUMAT_CHECK_ERROR();
    std::vector<int> hostMem(24);
    CUMAT_SAFE_CALL(cudaMemcpy(&hostMem[0], outMem.pointer(), sizeof(int) * 24, cudaMemcpyDeviceToHost));
    for (int i = 0; i < 24; ++i) REQUIRE(hostMem[i] == i + 1);
}
