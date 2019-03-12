#include <catch/catch.hpp>

#include <cuMat/Dense>
#include <cuMat/src/EigenInteropHelpers.h>
#include <vector>

#include <third-party/Eigen/Dense>

#include "Utils.h"

using namespace cuMat;

template<typename Scalar, int Flags, int Dims, bool DetLeqFour>
struct TestInverseWithDetHelper;
template<typename Scalar, int Flags, int Dims>
struct TestInverseWithDetHelper<Scalar, Flags, Dims, true>
{
	typedef Matrix<Scalar, Dims, Dims, Dynamic, Flags> mat_t;
	typedef Matrix<Scalar, 1, 1, Dynamic, Flags> scalar_t;
	typedef typename mat_t::EigenMatrix_t emat_t;
	static void run(const std::vector<emat_t>& inputMatricesHost, const mat_t& inputMatrixDevice)
	{
		int batches = inputMatrixDevice.batches();
		INFO("3. Test 'computeInverseAndDet'");
		{
			INFO("a) direct in");
			mat_t inverseMatrixDevice(Dims, Dims, batches);
			scalar_t detMatrixDevice(1, 1, batches);
			inputMatrixDevice.computeInverseAndDet(inverseMatrixDevice, detMatrixDevice);
			std::vector<Scalar> determinantHost(batches);
			detMatrixDevice.copyToHost(&determinantHost[0]);
			for (int i = 0; i < batches; ++i)
			{
				INFO("batch " << i);
				INFO("input: \n" << inputMatricesHost[i]);

				emat_t expectedInverse = inputMatricesHost[i].inverse();
				INFO("expected inverse:\n" << expectedInverse);
				emat_t actualInverse = inverseMatrixDevice.slice(i).eval().toEigen();
				INFO("actual inverse:\n" << actualInverse);
				REQUIRE(expectedInverse.isApprox(actualInverse));

				Scalar expectedDet = inputMatricesHost[i].determinant();
				INFO("expected determinant: " << expectedDet);
				Scalar actualDet = determinantHost[i];
				INFO("actual determinant: " << actualDet);
				REQUIRE(expectedDet == Approx(actualDet));
			}
		}
		{
			INFO("b) cwise in");
			mat_t inverseMatrixDevice(Dims, Dims, batches);
			scalar_t detMatrixDevice(1, 1, batches);
			(inputMatrixDevice + 0.0f).computeInverseAndDet(inverseMatrixDevice, detMatrixDevice);
			std::vector<Scalar> determinantHost(batches);
			detMatrixDevice.copyToHost(&determinantHost[0]);
			for (int i = 0; i < batches; ++i)
			{
				INFO("batch " << i);
				INFO("input: \n" << inputMatricesHost[i]);

				emat_t expectedInverse = inputMatricesHost[i].inverse();
				INFO("expected inverse:\n" << expectedInverse);
				emat_t actualInverse = inverseMatrixDevice.slice(i).eval().toEigen();
				INFO("actual inverse:\n" << actualInverse);
				REQUIRE(expectedInverse.isApprox(actualInverse));

				Scalar expectedDet = inputMatricesHost[i].determinant();
				INFO("expected determinant: " << expectedDet);
				Scalar actualDet = determinantHost[i];
				INFO("actual determinant: " << actualDet);
				REQUIRE(expectedDet == Approx(actualDet));
			}
		}
	}
};
template<typename Scalar, int Flags, int Dims>
struct TestInverseWithDetHelper<Scalar, Flags, Dims, false>
{
	typedef Matrix<Scalar, Dims, Dims, Dynamic, Flags> mat_t;
	typedef typename mat_t::EigenMatrix_t emat_t;
	static void run(const std::vector<emat_t>& inputMatricesHost, const mat_t& inputMatrixDevice)
	{
	}
};

template<typename Scalar, int Flags, int Dims>
void testLinAlgOpsReal()
{
    INFO("Size=" << Dims << ", Flags=" << Flags);
    const int batches = 5;
    typedef Matrix<Scalar, Dims, Dims, Dynamic, Flags> mat_t;
	typedef Matrix<Scalar, 1, 1, Dynamic, Flags> scalar_t;
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
            auto determinantDevice = (inputMatrixDevice + 0.0f).determinant().eval();
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
            auto determinantDevice = (inputMatrixDevice.determinant() + 0.0f).eval();
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
            auto determinantDevice = ((inputMatrixDevice + 0.0f).determinant() + 0.0f).eval();
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

	//2. Inverse
	{
		INFO("2. Test inverse");
		{
			INFO("a) direct in");
			auto inverseDevice = inputMatrixDevice.inverse().eval();
			REQUIRE(inverseDevice.rows() == Dims);
			REQUIRE(inverseDevice.cols() == Dims);
			REQUIRE(inverseDevice.batches() == batches);
			for (int i = 0; i < batches; ++i)
			{
				INFO("batch " << i);
				INFO("input: \n" << inputMatricesHost[i]);
				emat_t expected = inputMatricesHost[i].inverse();
				INFO("expected:\n" << expected);
				emat_t actual = inverseDevice.slice(i).eval().toEigen();
				INFO("actual:\n" << actual);
				REQUIRE(expected.isApprox(actual));
			}
		}
		{
			INFO("b) cwise in");
			auto inverseDevice = (inputMatrixDevice + 0.0f).inverse().eval();
			REQUIRE(inverseDevice.rows() == Dims);
			REQUIRE(inverseDevice.cols() == Dims);
			REQUIRE(inverseDevice.batches() == batches);
			for (int i = 0; i < batches; ++i)
			{
				INFO("batch " << i);
				INFO("input: \n" << inputMatricesHost[i]);
				emat_t expected = inputMatricesHost[i].inverse();
				INFO("expected:\n" << expected);
				emat_t actual = inverseDevice.slice(i).eval().toEigen();
				INFO("actual:\n" << actual);
				REQUIRE(expected.isApprox(actual));
			}
		}
	}

	//3. Inverse with determinant
	TestInverseWithDetHelper<Scalar, Flags, Dims, (Dims <= 4) >::run(inputMatricesHost, inputMatrixDevice);
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

