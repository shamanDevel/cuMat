#include "TestNoCUDA.h"

#include <cuMat/Core>

int cudaSumAll(const cuMat::BMatrixXiR& mat)
{
	return static_cast<int>(mat.sum());
}