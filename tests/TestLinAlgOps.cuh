#include <catch/catch.hpp>

#include <cuMat/Dense>
#include <cuMat/src/EigenInteropHelpers.h>
#include <vector>

#include <third-party/Eigen/Dense>

#include "Utils.h"

using namespace cuMat;

template<typename Scalar, int Flags, int Dims>
void testLinAlgOpsReal()
{
    INFO("Size=" << Dims << ", Flags=" << Flags);
    const int batches = 5;
    typedef Matrix<Scalar, Dims, Dims, Dynamic, Flags> mat_t;
    typedef typename mat_t::EigenMatrix_t emat_t;

    //create input matrices
    std::vector<emat_t> inputMatricesHost(batches);
    mat_t inputMatrixDevice(Dims, Dims, batches);
    for (int i=0; i<batches; ++i)
    {
        inputMatricesHost[i] = emat_t::Random();
        auto slice = Matrix<Scalar, Dims, Dims, 1, Flags>::fromEigen(inputMatricesHost[i]);
        inputMatrixDevice.template block<Dims, Dims, 1>(0, 0, i) = slice;
        INFO("input batch "<<i<<":\n" << inputMatricesHost[i]);
    }
    INFO("inputMatrixDevice: " << inputMatrixDevice);

    //1. Determinant
    {
        INFO("1. Test determinant");
        {
            INFO("a) direct in, direct out");
            auto determinantDevice = inputMatrixDevice.determinant().eval();
            REQUIRE(determinantDevice.rows() == 1);
            REQUIRE(determinantDevice.cols() == 1);
            REQUIRE(determinantDevice.batches() == batches);
            std::vector<Scalar> determinantHost(batches);
            determinantDevice.copyToHost(&determinantHost[0]);
            for (int i=0; i<batches; ++i)
            {
                INFO("batch " << i);
                INFO("input: \n" << inputMatricesHost[i]);
                REQUIRE(determinantHost[i] == Approx(inputMatricesHost[i].determinant()));
            }
        }
        {
            INFO("b) cwise in, direct out");
            auto determinantDevice = (inputMatrixDevice + 0).determinant().eval();
            REQUIRE(determinantDevice.rows() == 1);
            REQUIRE(determinantDevice.cols() == 1);
            REQUIRE(determinantDevice.batches() == batches);
            std::vector<Scalar> determinantHost(batches);
            determinantDevice.copyToHost(&determinantHost[0]);
            for (int i=0; i<batches; ++i)
            {
                INFO("batch " << i);
                INFO("input: \n" << inputMatricesHost[i]);
                REQUIRE(determinantHost[i] == Approx(inputMatricesHost[i].determinant()));
            }
        }
        {
            INFO("c) direct in, cwise out");
            auto determinantDevice = (inputMatrixDevice.determinant() + 0).eval();
            cudaDeviceSynchronize();
            REQUIRE(determinantDevice.rows() == 1);
            REQUIRE(determinantDevice.cols() == 1);
            REQUIRE(determinantDevice.batches() == batches);
            std::vector<Scalar> determinantHost(batches);
            determinantDevice.copyToHost(&determinantHost[0]);
            for (int i=0; i<batches; ++i)
            {
                INFO("batch " << i);
                INFO("input: \n" << inputMatricesHost[i]);
                REQUIRE(determinantHost[i] == Approx(inputMatricesHost[i].determinant()));
            }
        }
        {
            INFO("d) cwise in, cwise out");
            auto determinantDevice = ((inputMatrixDevice + 0).determinant() + 0).eval();
            REQUIRE(determinantDevice.rows() == 1);
            REQUIRE(determinantDevice.cols() == 1);
            REQUIRE(determinantDevice.batches() == batches);
            std::vector<Scalar> determinantHost(batches);
            determinantDevice.copyToHost(&determinantHost[0]);
            for (int i=0; i<batches; ++i)
            {
                INFO("batch " << i);
                INFO("input: \n" << inputMatricesHost[i]);
                REQUIRE(determinantHost[i] == Approx(inputMatricesHost[i].determinant()));
            }
        }
    }
}
template<int Dims>
void testlinAlgOps2()
{
    SECTION("float")
    {
        SECTION("row major")
        {
            testLinAlgOpsReal<float, RowMajor, Dims>();
        }
        SECTION("column major")
        {
            testLinAlgOpsReal<float, ColumnMajor, Dims>();
        }
    }
    SECTION("double")
    {
        SECTION("row major")
        {
            testLinAlgOpsReal<double, RowMajor, Dims>();
        }
        SECTION("column major")
        {
            testLinAlgOpsReal<double, ColumnMajor, Dims>();
        }
    }
}

