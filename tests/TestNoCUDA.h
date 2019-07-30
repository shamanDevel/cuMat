#ifndef __TEST_NO_CUDA_H__
#define __TEST_NO_CUDA_H__

#include <cuMat/Core>

int cudaSumAll(const cuMat::BMatrixXiR& mat);

#endif