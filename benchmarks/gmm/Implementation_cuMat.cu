#include "benchmark.h"

#include <cuMat/Core>
#include <cuMat/Dense>
#include <iostream>
#include <fstream>

namespace
{
	class Gaussian
	{
		static const double log2pi;

		size_t n_;
		cuMat::VectorXd mean_;
		cuMat::MatrixXd covariance_;
		cuMat::MatrixXd invCovariance_;
		double logdet_;
		cuMat::CholeskyDecomposition<cuMat::MatrixXd> cholesky_;

		cuMat::BVectorXd tmp_;

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	public:
		Gaussian(const cuMat::VectorXd& mean, const cuMat::MatrixXd& cov)
			: n_(mean.size())
			, mean_(mean)
			, covariance_(cov)
			, invCovariance_(cov.rows(), cov.cols())
			, logdet_(0)
			, cholesky_(cov.rows())
		{
			computeDetInv();
		}
		void computeDetInv()
		{
			cuMat::Scalard det;
			if (n_ == 1) {
				covariance_.block<1, 1, 1>(0, 0, 0).computeInverseAndDet(invCovariance_, det);
				logdet_ = std::log(static_cast<double>(det));
			}
			else if (n_ == 2) {
				covariance_.block<2, 2, 1>(0, 0, 0).computeInverseAndDet(invCovariance_, det);
				logdet_ = std::log(static_cast<double>(det));
			}
			else if (n_ == 3) {
				covariance_.block<3, 3, 1>(0, 0, 0).computeInverseAndDet(invCovariance_, det);
				logdet_ = std::log(static_cast<double>(det));
			}
			else if (n_ == 4) {
				covariance_.block<4, 4, 1>(0, 0, 0).computeInverseAndDet(invCovariance_, det);
				logdet_ = std::log(static_cast<double>(det));
			}
			else {
				cholesky_.compute(covariance_);
				logdet_ = static_cast<double>(cholesky_.logDeterminant());
			}
			if (logdet_ < -20) // det(A) < ~2e-9
			{
				std::cerr << "Covariance is singular, re-initialize with the identity matrix" << std::endl;
				covariance_ = cuMat::MatrixXd::Identity(n_, n_);
				invCovariance_.cuMat::MatrixXd::Identity(n_, n_);
				logdet_ = 0; // log(det(In))=log(1)=0
			}
		}
		//Returns the log-probability of x in this gaussian
		cuMat::BScalard logP(const cuMat::BVectorXd& x)
		{
			tmp_ = x - mean_;
			if (n_ <= 4) {
				//auto alpha = tmp_.dot(invCovariance_*tmp_) //slower
				//this version is faster with cuBLAS by a factor of 4
				//I think, the batched GEMM reads the matrix multiple times for all batches of right hand sides,
				//while if I reinterpret the batches as columns of a RHS matrix,
				//the LHS matrix invCovariance_ is only read once.
				auto alpha = tmp_.dot(
					(invCovariance_*tmp_.swapAxis<cuMat::Row, cuMat::Batch, cuMat::NoAxis>())
					.swapAxis<cuMat::Row, cuMat::NoAxis, cuMat::Column>());
				return -0.5 * (alpha + (n_ * log2pi + logdet_));
			} 
			else {
				auto alpha = tmp_.dot(
					cholesky_.solve(
						tmp_.swapAxis<cuMat::Row, cuMat::Batch, cuMat::NoAxis>())
					.swapAxis<cuMat::Row, cuMat::NoAxis, cuMat::Column>());
				return -0.5 * (alpha + (n_ * log2pi + logdet_));
			}
		}
		//getter+setter
		const cuMat::VectorXd& mean() const { return mean_; }
		cuMat::VectorXd& mean() { return mean_; }
		const cuMat::MatrixXd& cov() const { return covariance_; }
		cuMat::MatrixXd& cov() { return covariance_; }

		friend std::ostream& operator<<(std::ostream& o, const Gaussian& g)
		{
			o << "Mean: " << g.mean().transpose().eval() << "\n"
				<< "Cov:\n" << g.cov();
			return o;
		}
	};
	const double Gaussian::log2pi = 1.8378770664093454835606594728112f;

	//Log-Sum-Exp, reduction per row
	template<typename Derived, typename Scalar = typename Derived::Scalar>
	cuMat::Matrix<Scalar, cuMat::Dynamic, 1, 1, cuMat::ColumnMajor> 
	LSE_row(const cuMat::MatrixBase<Derived>& vec)
	{
		auto m = vec.template maxCoeff<cuMat::Axis::Column>().eval();
		return m + (vec - m)
			.cwiseExp()
			.template sum<cuMat::Axis::Column>()
			.cwiseLog();
	}
	//Log-Sum-Exp, reduction per column
	template<typename Derived, typename Scalar = typename Derived::Scalar>
	cuMat::Matrix<Scalar, 1, cuMat::Dynamic, 1, cuMat::ColumnMajor>
	LSE_col(const cuMat::MatrixBase<Derived>& vec)
	{
		auto m = vec.template maxCoeff<cuMat::Axis::Row>().eval();
		return m + (vec - m)
			.cwiseExp()
			.template sum<cuMat::Axis::Row>()
			.cwiseLog();
	}
}

void benchmark_cuMat(
	const std::string& pointsFile,
	const std::string& settingsFile,
	int numIterations,
	Json::Object& returnValues)
{
	//load settings and points
	int dimension, components, numPoints;
	cuMat::BVectorXd points;
	cuMat::BScalard logWeights;
	std::vector<Gaussian> gaussians;
	{ //points
		std::ifstream in(pointsFile);
		in >> dimension >> components >> numPoints;
		float dummy; //skip ground truth
		for (int i = 0; i < components * (1 + dimension + dimension * dimension); ++i) in >> dummy;
		std::vector<double> pointsHost(dimension * numPoints);
		size_t pointIndex = 0;
		for (int i = 0; i < numPoints; ++i) { //read points
			for (int d = 0; d < dimension; ++d) in >> pointsHost[pointIndex++];
		}
		points = cuMat::BVectorXd(dimension, 1, numPoints);
		points.copyFromHost(pointsHost.data());
	}
	{ //initial settings
		std::ifstream in(settingsFile);
		int dummy;
		in >> dimension >> components >> dummy;
		std::vector<double> logWeightsHost(components);
		gaussians.reserve(components);
		Eigen::VectorXd mean(dimension);
		Eigen::MatrixXd cov(dimension, dimension);
		for (int i = 0; i < components; ++i)
		{
			double weight;
			in >> weight;
			logWeightsHost[i] = std::log(weight);
			for (int x = 0; x < dimension; ++x) in >> mean[x];
			for (int x = 0; x < dimension; ++x)
				for (int y = 0; y < dimension; ++y)
					in >> cov(x, y);
			gaussians.emplace_back(
				cuMat::VectorXd::fromEigen(mean),
				cuMat::MatrixXd::fromEigen(cov));
		}
		logWeights = cuMat::BScalard(1, 1, components);
		logWeights.copyFromHost(logWeightsHost.data());
	}

#ifndef NDEBUG
	for (int k = 0; k < components; ++k)
	{
		std::cout << "\nComponent " << k
			<< "\nWeight: " << std::exp(static_cast<double>(logWeights.slice(k)))
			<< "\n" << gaussians[k] << std::endl;
	}
#endif

	//temporary memory
	cuMat::MatrixXd logW(numPoints, components);

	//run EM (fixed number of iterations)
	std::cout << "Run EM algorithm" << std::endl;
	std::chrono::time_point<std::chrono::steady_clock> start;
	double logLikeliehoodAccum;
	for (int iter = 0; iter < numIterations; ++iter)
	{
		//half of the iterations for warm up
		if (iter == numIterations / 2) start = std::chrono::steady_clock::now();

#ifndef NDEBUG
		std::cout << "    Iteration " << iter << std::endl;
#endif

		//Precomputation
		for (int i = 0; i < components; ++i)
			gaussians[i].computeDetInv();

		//E-Step
		logLikeliehoodAccum = 0;
		logW.setZero();
		for (int k = 0; k < components; ++k)
		{
			logW.col(k) = gaussians[k].logP(points).swapAxis<cuMat::Batch, cuMat::NoAxis, cuMat::NoAxis>()
			            + logWeights.slice(k);
		}
		auto lseR = LSE_row(logW);
		logW = logW - lseR;
		logLikeliehoodAccum += static_cast<double>(lseR.sum());
#ifndef NDEBUG
		auto logWHost = logW.toEigen();
		for (int i = 0; i < numPoints; ++i)
		{
			std::cout << "Point " << i << ":";
			for (int k = 0; k < components; ++k)
				std::cout << " " << std::exp(logWHost(i, k));
			std::cout << std::endl;
		}
#endif

		//M-Step
		auto lseC = LSE_col(logW);
		Eigen::RowVectorXd lseCHost = lseC.toEigen();
		logWeights.inplace() = lseC.swapAxis<cuMat::NoAxis, cuMat::NoAxis, cuMat::Column>()
			- std::log(numPoints);
		for (int k = 0; k < components; ++k)
		{
#ifndef NDEBUG
			std::cout << "Component " << k << " pre-update:"
				<< "\nWeight: " << std::exp(static_cast<double>(logWeights.slice(k)))
				<< "\n" << gaussians[k] << std::endl;
#endif

			double divNk = 1.0f / std::exp(lseCHost[k]);
			gaussians[k].mean().inplace() = (
				logW.col(k)
				.cwiseExp()
				.swapAxis<cuMat::NoAxis, cuMat::NoAxis, cuMat::Row>()
				.cwiseMul(points)
				).sum<cuMat::Batch>()
				* divNk;
			gaussians[k].cov().inplace() = (
				logW.col(k)
				.cwiseExp()
				.swapAxis<cuMat::NoAxis, cuMat::NoAxis, cuMat::Row>()
				.cwiseMul(
					(points - gaussians[k].mean()) * 
					(points - gaussians[k].mean()).transpose()
					)
				).sum<cuMat::Batch>()
				* divNk
				+ cuMat::MatrixXd::Identity(dimension)*1e-1;

#ifndef NDEBUG
			std::cout << "Component " << k << " post-update:"
				<< "\nWeight: " << std::exp(static_cast<double>(logWeights.slice(k)))
				<< "\n" << gaussians[k] << std::endl;
#endif
		}

#ifndef NDEBUG
		std::cout << " -> log-likelihood: " << logLikeliehoodAccum << "\n";
#endif
	}
#ifdef NDEBUG
	std::cout << " -> log-likelihood: " << logLikeliehoodAccum << "\n";
#endif

	auto finish = std::chrono::steady_clock::now();
	double elapsed = std::chrono::duration_cast<
		std::chrono::duration<double>>(finish - start).count() * 1000 * 2; //*2 for warm up
	std::cout << "    Done in " << elapsed << "ms" << std::endl;

	//save results
	returnValues.Insert(std::make_pair("Time", elapsed));
	returnValues.Insert(std::make_pair("LogLikelihood", logLikeliehoodAccum));
	Json::Array comData;
	for (int k = 0; k < components; ++k)
	{
		Json::Object com;
		com.Insert(std::make_pair("Weight", std::exp(static_cast<double>(logWeights.slice(k)))));
		Json::Array m, c;
		auto mean = gaussians[k].mean().toEigen();
		auto cov = gaussians[k].cov().toEigen();
		for (int i = 0; i < dimension; ++i) m.PushBack(mean[i]);
		for (int i = 0; i < dimension; ++i)
			for (int j = 0; j < dimension; ++j)
				c.PushBack(cov(i, j));
		com.Insert(std::make_pair("Mean", m));
		com.Insert(std::make_pair("Cov", c));
		comData.PushBack(com);
	}
	returnValues.Insert(std::make_pair("Components", comData));
}