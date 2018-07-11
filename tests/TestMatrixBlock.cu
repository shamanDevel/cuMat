#include <catch/catch.hpp>

#include <cuMat/Core>
#include "Utils.h"

TEST_CASE("fixed_block_rw", "[block]")
{
	const int rows = 10;
	const int cols = 16;
	const int batches = 8;

	Eigen::Matrix<int, rows, cols, Eigen::ColMajor> m_host1[batches];
	for (int i = 0; i < batches; ++i) m_host1[i].setRandom(rows, cols);

	cuMat::Matrix<int, rows, cols, batches, cuMat::ColumnMajor> m_device;
	for (int i = 0; i < batches; ++i) {
		auto slice = cuMat::Matrix<int, rows, cols, 1, cuMat::ColumnMajor>::fromEigen(m_host1[i]);
		m_device.block<rows, cols, 1>(0, 0, i) = slice;
	}

	Eigen::Matrix<int, rows, cols, Eigen::ColMajor> m_host2[batches];
	for (int i=0; i<batches; ++i)
	{
		m_host2[i] = m_device.block<rows, cols, 1>(0, 0, i).eval().toEigen();
	}

	const cuMat::Matrix<int, rows, cols, batches, cuMat::ColumnMajor> m_device_const = m_device;
	Eigen::Matrix<int, rows, cols, Eigen::ColMajor> m_host3[batches];
	for (int i = 0; i<batches; ++i)
	{
		m_host3[i] = m_device_const.block<rows, cols, 1>(0, 0, i).eval().toEigen();
	}

	for (int i = 0; i < batches; ++i) REQUIRE(m_host1[i] == m_host2[i]);
	for (int i = 0; i < batches; ++i) REQUIRE(m_host1[i] == m_host3[i]);
}

TEST_CASE("dynamic_block_rw", "[block]")
{
	const int rows = 10;
	const int cols = 16;
	const int batches = 8;

	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> m_host1[batches];
	for (int i = 0; i < batches; ++i) m_host1[i].setRandom(rows, cols);

	cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajor> m_device(rows, cols, batches);
	for (int i = 0; i < batches; ++i) {
		auto slice = cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, 1, cuMat::ColumnMajor>::fromEigen(m_host1[i]);
		m_device.block(0, 0, i, rows, cols, 1) = slice;
	}

	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> m_host2[batches];
	for (int i = 0; i<batches; ++i)
	{
		m_host2[i] = m_device.block(0, 0, i, rows, cols, 1).eval().toEigen();
	}

	const cuMat::Matrix<int, cuMat::Dynamic, cuMat::Dynamic, cuMat::Dynamic, cuMat::ColumnMajor> m_device_const = m_device;
	Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> m_host3[batches];
	for (int i = 0; i<batches; ++i)
	{
		m_host3[i] = m_device_const.block(0, 0, i, rows, cols, 1).eval().toEigen();
	}

	for (int i = 0; i < batches; ++i) REQUIRE(m_host1[i] == m_host2[i]);
	for (int i = 0; i < batches; ++i) REQUIRE(m_host1[i] == m_host3[i]);
}

TEST_CASE("cwise_r", "[block]")
{
    int data[2][2][2] {
        {
            {1, 2},
            {3, 4}
        },
        {
            {5, 6},
            {7, 8}
        }
    };
    cuMat::BMatrixXiR mat = cuMat::BMatrixXiR::fromArray(data);
    auto matCwise = (mat * 1);
    
    cuMat::Matrix2i batch1 = matCwise.block<2, 2, 1>(0, 0, 0);
    int batch1Expected[1][2][2]{
        {
            { 1, 2 },
            { 3, 4 }
        }
    };
    assertMatrixEquality(batch1Expected, batch1);
    
    cuMat::Matrix2i batch2 = matCwise.block<2, 2, 1>(0, 0, 1);
    int batch2Expected[1][2][2]{
        {
            { 5, 6 },
            { 7, 8 }
        }
    };
    assertMatrixEquality(batch2Expected, batch2);
}


TEST_CASE("cwise_convenients_r", "[block]")
{
    //convenience methods row, col, slice, segment

    int data[2][2][2] {
        {
            {1, 2},
            {3, 4}
        },
        {
            {5, 6},
            {7, 8}
        }
    };
    cuMat::BMatrixXiR mat = cuMat::BMatrixXiR::fromArray(data);
    auto matCwise = (mat * 1);

    cuMat::BRowVector2i row1 = matCwise.row(0);
    int row1Expected[2][1][2]{
        {
            { 1, 2 }
        },
        {
            { 5, 6 }
        }
    };
    assertMatrixEquality(row1Expected, row1);
    cuMat::BRowVector2i row2 = matCwise.row(1);
    int row2Expected[2][1][2]{
        {
            { 3, 4 }
        },
        {
            { 7, 8 }
        }
    };
    assertMatrixEquality(row2Expected, row2);
    
    cuMat::BVector2i col1 = matCwise.col(0);
    int col1Expected[2][2][1]{
        {
            { 1 },
            { 3 }
        },
        {
            { 5 },
            { 7 }
        }
    };
    assertMatrixEquality(col1Expected, col1);
    cuMat::BVector2i col2 = matCwise.col(1);
    int col2Expected[2][2][1]{
        {
            { 2 },
            { 4 }
        },
        {
            { 6 },
            { 8 }
        }
    };
    assertMatrixEquality(col2Expected, col2);

    cuMat::Matrix2i batch1 = matCwise.slice(0);
    int batch1Expected[1][2][2]{
        {
            { 1, 2 },
            { 3, 4 }
        }
    };
    assertMatrixEquality(batch1Expected, batch1);
    cuMat::Matrix2i batch2 = matCwise.slice(1);
    int batch2Expected[1][2][2]{
        {
            { 5, 6 },
            { 7, 8 }
        }
    };
    assertMatrixEquality(batch2Expected, batch2);
}

TEST_CASE("matrix_convenients_r", "[block]")
{
    //convenience methods row, col, slice, segment

    int data[2][2][2] {
        {
            {1, 2},
            {3, 4}
        },
        {
            {5, 6},
            {7, 8}
        }
    };
    const cuMat::BMatrixXiR mat = cuMat::BMatrixXiR::fromArray(data);

    cuMat::BRowVector2i row1 = mat.row(0);
    int row1Expected[2][1][2]{
        {
            { 1, 2 }
        },
        {
            { 5, 6 }
        }
    };
    assertMatrixEquality(row1Expected, row1);
    cuMat::BRowVector2i row2 = mat.row(1);
    int row2Expected[2][1][2]{
        {
            { 3, 4 }
        },
        {
            { 7, 8 }
        }
    };
    assertMatrixEquality(row2Expected, row2);
    
    cuMat::BVector2i col1 = mat.col(0);
    int col1Expected[2][2][1]{
        {
            { 1 },
            { 3 }
        },
        {
            { 5 },
            { 7 }
        }
    };
    assertMatrixEquality(col1Expected, col1);
    cuMat::BVector2i col2 = mat.col(1);
    int col2Expected[2][2][1]{
        {
            { 2 },
            { 4 }
        },
        {
            { 6 },
            { 8 }
        }
    };
    assertMatrixEquality(col2Expected, col2);

    cuMat::Matrix2i batch1 = mat.slice(0);
    int batch1Expected[1][2][2]{
        {
            { 1, 2 },
            { 3, 4 }
        }
    };
    assertMatrixEquality(batch1Expected, batch1);
    cuMat::Matrix2i batch2 = mat.slice(1);
    int batch2Expected[1][2][2]{
        {
            { 5, 6 },
            { 7, 8 }
        }
    };
    assertMatrixEquality(batch2Expected, batch2);
}

TEST_CASE("matrix_convenients_w", "[block]")
{
    //convenience methods row, col, slice, segment

    int data[2][2][2] {
        {
            {1, 2},
            {3, 4}
        },
        {
            {5, 6},
            {7, 8}
        }
    };
    const cuMat::BMatrixXiR mat1 = cuMat::BMatrixXiR::fromArray(data);
    cuMat::BMatrixXi mat2;

    mat2 = cuMat::BMatrixXi::Zero(2, 2, 2);
    mat2.row(0) = mat1.row(0);
    cuMat::BRowVector2i row1 = mat2.row(0);
    int row1Expected[2][1][2]{
        {
            { 1, 2 }
        },
        {
            { 5, 6 }
        }
    };
    assertMatrixEquality(row1Expected, row1);
    REQUIRE(mat2.row(1).cwiseAbs2().sum() == 0);

    mat2 = cuMat::BMatrixXi::Zero(2, 2, 2);
    mat2.row(1) = mat1.row(1);
    cuMat::BRowVector2i row2 = mat2.row(1);
    int row2Expected[2][1][2]{
        {
            { 3, 4 }
        },
        {
            { 7, 8 }
        }
    };
    assertMatrixEquality(row2Expected, row2);
    REQUIRE(mat2.row(0).cwiseAbs2().sum() == 0);

    mat2 = cuMat::BMatrixXi::Zero(2, 2, 2);
    mat2.col(0) = mat1.col(0);
    cuMat::BVector2i col1 = mat2.col(0);
    int col1Expected[2][2][1]{
        {
            { 1 },
            { 3 }
        },
        {
            { 5 },
            { 7 }
        }
    };
    assertMatrixEquality(col1Expected, col1);
    REQUIRE(mat2.col(1).cwiseAbs2().sum() == 0);

    mat2 = cuMat::BMatrixXi::Zero(2, 2, 2);
    mat2.col(1) = mat1.col(1);
    cuMat::BVector2i col2 = mat2.col(1);
    int col2Expected[2][2][1]{
        {
            { 2 },
            { 4 }
        },
        {
            { 6 },
            { 8 }
        }
    };
    assertMatrixEquality(col2Expected, col2);
    REQUIRE(mat2.col(0).cwiseAbs2().sum() == 0);

    mat2 = cuMat::BMatrixXi::Zero(2, 2, 2);
    mat2.slice(0) = mat1.slice(0);
    cuMat::Matrix2i batch1 = mat2.slice(0);
    int batch1Expected[1][2][2]{
        {
            { 1, 2 },
            { 3, 4 }
        }
    };
    assertMatrixEquality(batch1Expected, batch1);
    REQUIRE(mat2.slice(1).squaredNorm() == 0);

    mat2 = cuMat::BMatrixXi::Zero(2, 2, 2);
    mat2.slice(1) = mat1.slice(1);
    cuMat::Matrix2i batch2 = mat2.slice(1);
    int batch2Expected[1][2][2]{
        {
            { 5, 6 },
            { 7, 8 }
        }
    };
    assertMatrixEquality(batch2Expected, batch2);
    REQUIRE(mat2.slice(0).squaredNorm() == 0);
}

TEST_CASE("cwise_vector", "[block]")
{
    SECTION("Row Vector") {
        int dataRow[1][1][5]{ {
            {1, 2, 3, 4, 5}
        } };
        cuMat::RowVectorXi v = cuMat::RowVectorXi::fromArray(dataRow);
        auto vCwise = (v * 1);

        auto fixedSegmentExpr = vCwise.segment<2>(1);
        INFO(typeid(fixedSegmentExpr).name());
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::BatchesAtCompileTime == 1);
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::RowsAtCompileTime == 1);
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::ColsAtCompileTime == 2);
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::Flags == cuMat::internal::traits<cuMat::VectorXi>::Flags);

        auto dynamicSegmentExpr = vCwise.segment(1, 2);
        INFO(typeid(dynamicSegmentExpr).name());
        REQUIRE(cuMat::internal::traits<decltype(dynamicSegmentExpr)>::BatchesAtCompileTime == 1);
        REQUIRE(cuMat::internal::traits<decltype(dynamicSegmentExpr)>::RowsAtCompileTime == 1);
        REQUIRE(cuMat::internal::traits<decltype(dynamicSegmentExpr)>::ColsAtCompileTime == cuMat::Dynamic);
        REQUIRE(cuMat::internal::traits<decltype(dynamicSegmentExpr)>::Flags == cuMat::internal::traits<cuMat::VectorXi>::Flags);

        cuMat::RowVector2i v1a = vCwise.segment<2>(0);
        cuMat::RowVector2i v1b = vCwise.head<2>();
        cuMat::RowVectorXi v1c = vCwise.segment(0, 2);
        cuMat::RowVectorXi v1d = vCwise.head(2);
        int v1Expected[1][1][2]{ { { 1,2 } } };
        assertMatrixEquality(v1Expected, v1a);
        assertMatrixEquality(v1Expected, v1b);
        assertMatrixEquality(v1Expected, v1c);
        assertMatrixEquality(v1Expected, v1d);

        cuMat::RowVector3i v2a = vCwise.segment<3>(1);
        cuMat::RowVectorXi v2b = vCwise.segment(1, 3);
        int v2Expected[1][1][3]{ { { 2, 3, 4 } } };
        assertMatrixEquality(v2Expected, v2a);
        assertMatrixEquality(v2Expected, v2b);

        cuMat::RowVector2i v3a = vCwise.segment<2>(3);
        cuMat::RowVector2i v3b = vCwise.segment(3, 2);
        cuMat::RowVector2i v3c = vCwise.tail<2>();
        cuMat::RowVector2i v3d = vCwise.tail(2);
        int v3Expected[1][1][2]{ { { 4,5 } } };
        assertMatrixEquality(v3Expected, v3a);
        assertMatrixEquality(v3Expected, v3b);
        assertMatrixEquality(v3Expected, v3c);
        assertMatrixEquality(v3Expected, v3d);
    }

    SECTION("Column Vector") {
        int dataCol[1][5][1]{ {
            {1}, {2}, {3}, {4}, {5}
            } };
        cuMat::VectorXi v = cuMat::VectorXi::fromArray(dataCol);
        auto vCwise = (v * 1);

        auto fixedSegmentExpr = vCwise.segment<2>(1);
        INFO(typeid(fixedSegmentExpr).name());
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::BatchesAtCompileTime == 1);
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::RowsAtCompileTime == 2);
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::ColsAtCompileTime == 1);
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::Flags == cuMat::internal::traits<cuMat::VectorXi>::Flags);

        cuMat::Vector2i v1a = vCwise.segment<2>(0);
        cuMat::Vector2i v1b = vCwise.head<2>();
        cuMat::VectorXi v1c = vCwise.segment(0, 2);
        cuMat::VectorXi v1d = vCwise.head(2);
        int v1Expected[1][2][1]{ { {1},{2} }};
        assertMatrixEquality(v1Expected, v1a);
        assertMatrixEquality(v1Expected, v1b);
        assertMatrixEquality(v1Expected, v1c);
        assertMatrixEquality(v1Expected, v1d);

        cuMat::Vector3i v2a = vCwise.segment<3>(1);
        cuMat::VectorXi v2b = vCwise.segment(1, 3);
        int v2Expected[1][3][1]{ { { 2 },{ 3 }, { 4 } } };
        assertMatrixEquality(v2Expected, v2a);
        assertMatrixEquality(v2Expected, v2b);

        cuMat::Vector2i v3a = vCwise.segment<2>(3);
        cuMat::Vector2i v3b = vCwise.segment(3, 2);
        cuMat::Vector2i v3c = vCwise.tail<2>();
        cuMat::Vector2i v3d = vCwise.tail(2);
        int v3Expected[1][2][1]{ { { 4 },{ 5 } } };
        assertMatrixEquality(v3Expected, v3a);
        assertMatrixEquality(v3Expected, v3b);
        assertMatrixEquality(v3Expected, v3c);
        assertMatrixEquality(v3Expected, v3d);
    }
}

TEST_CASE("matrix_vector_r", "[block]")
{
    SECTION("Row Vector") {
        int dataRow[1][1][5]{ {
            { 1, 2, 3, 4, 5 }
            } };
        const cuMat::RowVectorXi v = cuMat::RowVectorXi::fromArray(dataRow);

        auto fixedSegmentExpr = v.segment<2>(1);
        INFO(typeid(fixedSegmentExpr).name());
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::BatchesAtCompileTime == 1);
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::RowsAtCompileTime == 1);
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::ColsAtCompileTime == 2);
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::Flags == cuMat::internal::traits<cuMat::VectorXi>::Flags);

        auto dynamicSegmentExpr = v.segment(1, 2);
        INFO(typeid(dynamicSegmentExpr).name());
        REQUIRE(cuMat::internal::traits<decltype(dynamicSegmentExpr)>::BatchesAtCompileTime == 1);
        REQUIRE(cuMat::internal::traits<decltype(dynamicSegmentExpr)>::RowsAtCompileTime == 1);
        REQUIRE(cuMat::internal::traits<decltype(dynamicSegmentExpr)>::ColsAtCompileTime == cuMat::Dynamic);
        REQUIRE(cuMat::internal::traits<decltype(dynamicSegmentExpr)>::Flags == cuMat::internal::traits<cuMat::VectorXi>::Flags);

        cuMat::RowVector2i v1a = v.segment<2>(0);
        cuMat::RowVector2i v1b = v.head<2>();
        cuMat::RowVectorXi v1c = v.segment(0, 2);
        cuMat::RowVectorXi v1d = v.head(2);
        int v1Expected[1][1][2]{ { { 1,2 } } };
        assertMatrixEquality(v1Expected, v1a);
        assertMatrixEquality(v1Expected, v1b);
        assertMatrixEquality(v1Expected, v1c);
        assertMatrixEquality(v1Expected, v1d);

        cuMat::RowVector3i v2a = v.segment<3>(1);
        cuMat::RowVectorXi v2b = v.segment(1, 3);
        int v2Expected[1][1][3]{ { { 2, 3, 4 } } };
        assertMatrixEquality(v2Expected, v2a);
        assertMatrixEquality(v2Expected, v2b);

        cuMat::RowVector2i v3a = v.segment<2>(3);
        cuMat::RowVector2i v3b = v.segment(3, 2);
        cuMat::RowVector2i v3c = v.tail<2>();
        cuMat::RowVector2i v3d = v.tail(2);
        int v3Expected[1][1][2]{ { { 4,5 } } };
        assertMatrixEquality(v3Expected, v3a);
        assertMatrixEquality(v3Expected, v3b);
        assertMatrixEquality(v3Expected, v3c);
        assertMatrixEquality(v3Expected, v3d);
    }

    SECTION("Column Vector") {
        int dataCol[1][5][1]{ {
            { 1 },{ 2 },{ 3 },{ 4 },{ 5 }
            } };
        const cuMat::VectorXi v = cuMat::VectorXi::fromArray(dataCol);

        auto fixedSegmentExpr = v.segment<2>(1);
        INFO(typeid(fixedSegmentExpr).name());
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::BatchesAtCompileTime == 1);
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::RowsAtCompileTime == 2);
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::ColsAtCompileTime == 1);
        REQUIRE(cuMat::internal::traits<decltype(fixedSegmentExpr)>::Flags == cuMat::internal::traits<cuMat::VectorXi>::Flags);

        cuMat::Vector2i v1a = v.segment<2>(0);
        cuMat::Vector2i v1b = v.head<2>();
        cuMat::VectorXi v1c = v.segment(0, 2);
        cuMat::VectorXi v1d = v.head(2);
        int v1Expected[1][2][1]{ { { 1 },{ 2 } } };
        assertMatrixEquality(v1Expected, v1a);
        assertMatrixEquality(v1Expected, v1b);
        assertMatrixEquality(v1Expected, v1c);
        assertMatrixEquality(v1Expected, v1d);

        cuMat::Vector3i v2a = v.segment<3>(1);
        cuMat::VectorXi v2b = v.segment(1, 3);
        int v2Expected[1][3][1]{ { { 2 },{ 3 },{ 4 } } };
        assertMatrixEquality(v2Expected, v2a);
        assertMatrixEquality(v2Expected, v2b);

        cuMat::Vector2i v3a = v.segment<2>(3);
        cuMat::Vector2i v3b = v.segment(3, 2);
        cuMat::Vector2i v3c = v.tail<2>();
        cuMat::Vector2i v3d = v.tail(2);
        int v3Expected[1][2][1]{ { { 4 },{ 5 } } };
        assertMatrixEquality(v3Expected, v3a);
        assertMatrixEquality(v3Expected, v3b);
        assertMatrixEquality(v3Expected, v3c);
        assertMatrixEquality(v3Expected, v3d);
    }
}

TEST_CASE("matrix_vector_w", "[block]")
{
    SECTION("Row Vector") {
        int dataRow[1][1][5]{ { { 1, 2, 3, 4, 5 } } };
        const cuMat::RowVectorXi v1 = cuMat::RowVectorXi::fromArray(dataRow);
        cuMat::RowVectorXi v2;

        v2 = cuMat::RowVectorXi::Zero(5);
        v2.segment<2>(1) = v1.segment<2>(1);
        int expected1[1][1][5]{ { { 0, 2, 3, 0, 0 } } };
        assertMatrixEquality(expected1, v2);

        v2 = cuMat::RowVectorXi::Zero(5);
        v2.segment(1, 2) = v1.segment<2>(1);
        int expected2[1][1][5]{ { { 0, 2, 3, 0, 0 } } };
        assertMatrixEquality(expected2, v2);

        v2 = cuMat::RowVectorXi::Zero(5);
        v2.head<3>() = v1.segment(0, 3);
        int expected3[1][1][5]{ { { 1, 2, 3, 0, 0 } } };
        assertMatrixEquality(expected3, v2);

        v2 = cuMat::RowVectorXi::Zero(5);
        v2.head(3) = v1.segment<3>(0);
        int expected4[1][1][5]{ { { 1, 2, 3, 0, 0 } } };
        assertMatrixEquality(expected4, v2);

        v2 = cuMat::RowVectorXi::Zero(5);
        v2.tail<3>() = v1.segment(2, 3);
        int expected5[1][1][5]{ { { 0, 0, 3, 4, 5 } } };
        assertMatrixEquality(expected5, v2);

        v2 = cuMat::RowVectorXi::Zero(5);
        v2.tail(3) = v1.segment<3>(2);
        int expected6[1][1][5]{ { { 0, 0, 3, 4, 5 } } };
        assertMatrixEquality(expected6, v2);
    }

    SECTION("Column Vector") {
        int dataRow[1][5][1]{ { { 1}, {2}, {3}, {4}, {5 } } };
        const cuMat::VectorXi v1 = cuMat::VectorXi::fromArray(dataRow);
        cuMat::VectorXi v2;

        v2 = cuMat::VectorXi::Zero(5);
        v2.segment<2>(1) = v1.segment<2>(1);
        int expected1[1][5][1]{ { { 0}, {2}, {3}, {0}, {0 } } };
        assertMatrixEquality(expected1, v2);

        v2 = cuMat::VectorXi::Zero(5);
        v2.segment(1, 2) = v1.segment<2>(1);
        int expected2[1][5][1]{ { { 0}, {2}, {3}, {0}, {0 } } };
        assertMatrixEquality(expected2, v2);

        v2 = cuMat::VectorXi::Zero(5);
        v2.head<3>() = v1.segment(0, 3);
        int expected3[1][5][1]{ { { 1}, {2}, {3}, {0}, {0 } } };
        assertMatrixEquality(expected3, v2);

        v2 = cuMat::VectorXi::Zero(5);
        v2.head(3) = v1.segment<3>(0);
        int expected4[1][5][1]{ { { 1}, {2}, {3}, {0}, {0 } } };
        assertMatrixEquality(expected4, v2);

        v2 = cuMat::VectorXi::Zero(5);
        v2.tail<3>() = v1.segment(2, 3);
        int expected5[1][5][1]{ { { 0}, {0}, {3}, {4}, {5 } } };
        assertMatrixEquality(expected5, v2);

        v2 = cuMat::VectorXi::Zero(5);
        v2.tail(3) = v1.segment<3>(2);
        int expected6[1][5][1]{ { { 0}, {0}, {3}, {4}, {5 } } };
        assertMatrixEquality(expected6, v2);
    }
}

//Utils assertMatrixEquality is based on matrix blocks, so test that also here
TEST_CASE("test_utils", "[block]")
{
    int data[2][2][2] {
        {
            {1, 2},
            {3, 4}
        },
        {
            {5, 6},
            {7, 8}
        }
    };
    cuMat::BMatrixXiR matR = cuMat::BMatrixXiR::fromArray(data);
    cuMat::BMatrixXiC matC = matR+0;
    
    assertMatrixEquality(data, matR);
    assertMatrixEquality(data, matC);
    
    assertMatrixEquality(matR, matC);
    assertMatrixEquality(matC, matR);
    
    assertMatrixEquality(matR*2, matC+matC);
    assertMatrixEquality(matC*2, matR+matR);
}


