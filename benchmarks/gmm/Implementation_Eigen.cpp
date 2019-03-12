#include "benchmark.h"

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <fstream>
#include <chrono>
#include <iostream>
#include <cstdlib>

namespace
{
	class Gaussian
	{
		static const double log2pi;

		size_t n_;
		Eigen::VectorXd mean_;
		Eigen::MatrixXd covariance_;
		Eigen::MatrixXd invCovariance_;
		double logdet_;
		Eigen::LLT<Eigen::MatrixXd> cholesky_;

		Eigen::VectorXd tmp_;

		EIGEN_MAKE_ALIGNED_OPERATOR_NEW
		
	public:
		Gaussian(const Eigen::VectorXd& mean, const Eigen::MatrixXd& cov)
			: n_(mean.size()), mean_(mean), covariance_(cov), logdet_(0)
		{
			invCovariance_.resizeLike(covariance_);
			computeDetInv();
		}
		void computeDetInv()
		{
			bool invertible = true;
			double det;
			if (n_ == 1) {
				covariance_.block<1, 1>(0, 0).computeInverseAndDetWithCheck(invCovariance_, det, invertible);
				logdet_ = std::log(det);
			}
			else if (n_ == 2) {
				covariance_.block<2, 2>(0, 0).computeInverseAndDetWithCheck(invCovariance_, det, invertible);
				logdet_ = std::log(det);
			}
			else if (n_ == 3) {
				covariance_.block<3, 3>(0, 0).computeInverseAndDetWithCheck(invCovariance_, det, invertible);
				logdet_ = std::log(det);
			}
			else if (n_ == 4) {
				covariance_.block<4, 4>(0, 0).computeInverseAndDetWithCheck(invCovariance_, det, invertible);
				logdet_ = std::log(det);
			}
			else {
				cholesky_.compute(covariance_);
				invertible = cholesky_.info() == Eigen::Success;
				//https://gist.github.com/redpony/fc8a0db6b20f7b1a3f23#gistcomment-2277286
				logdet_ = cholesky_.matrixL().toDenseMatrix().diagonal().array().log().sum();
			}
			if (!invertible)
			{
				std::cerr << "Covariance is singular, re-initialize with the identity matrix" << std::endl;
				covariance_.setIdentity(n_, n_);
				invCovariance_.setIdentity(n_, n_);
				logdet_ = 0; // log(det(In))=log(1)=0
			}
		}
		//Returns the log-probability of x in this gaussian
		double logP(const Eigen::VectorXd& x)
		{
			tmp_ = x - mean_;
			double alpha;
			if (n_ <= 4)
				alpha = tmp_.dot(invCovariance_*tmp_);
			else
				alpha = tmp_.dot(cholesky_.solve(tmp_));
			return -0.5 * (alpha + n_*log2pi + logdet_);
		}
		//getter+setter
		const Eigen::VectorXd& mean() const { return mean_; }
		Eigen::VectorXd& mean() { return mean_; }
		const Eigen::MatrixXd& cov() const { return covariance_; }
		Eigen::MatrixXd& cov() { return covariance_; }

		friend std::ostream& operator<<(std::ostream& o, const Gaussian& g)
		{
			o << "Mean: " << g.mean().transpose() << "\n"
				<< "Cov:\n" << g.cov();
			return o;
		}
	};
	const double Gaussian::log2pi = 1.8378770664093454835606594728112f;

	template<typename Derived, typename Scalar = typename Derived::Scalar>
	Scalar LSE(const Eigen::MatrixBase<Derived>& vec)
	{
		Scalar m = vec.maxCoeff();
		return m + std::log((vec.array() - m).exp().sum());
	}

}

void benchmark_Eigen(
	const std::string& pointsFile,
	const std::string& settingsFile,
	int numIterations,
    Json::Object& returnValues)
{
    //load settings and points
	int dimension, components, numPoints;
	Eigen::MatrixXd points;
	std::vector<double> logWeights;
	std::vector<Gaussian> gaussians;
	{ //points
		std::ifstream in(pointsFile);
		in >> dimension >> components >> numPoints;
		points.resize(dimension, numPoints);
		float dummy; //skip ground truth
		for (int i = 0; i < components * (1 + dimension + dimension * dimension); ++i) in >> dummy;
		for (int i = 0; i < numPoints; ++i) { //read points
			Eigen::VectorXd p(dimension);
			for (int d = 0; d < dimension; ++d) in >> p[d];
			points.col(i) = p;
		}
	}
	{ //initial settings
		std::ifstream in(settingsFile);
		int dummy;
		in >> dimension >> components >> dummy;
		logWeights.resize(components);
		gaussians.reserve(components);
		Eigen::VectorXd mean(dimension);
		Eigen::MatrixXd cov(dimension, dimension);
		for (int i=0; i<components; ++i)
		{
			double weight;
			in >> weight;
			logWeights[i] = std::log(weight);
			for (int x = 0; x < dimension; ++x) in >> mean[x];
			for (int x = 0; x < dimension; ++x)
				for (int y = 0; y < dimension; ++y)
					in >> cov(x, y);
			gaussians.emplace_back(mean, cov);
		}
	}

#ifndef NDEBUG
	for (int k = 0; k < components; ++k)
	{
		std::cout << "\nComponent " << k
			<< "\nWeight: " << std::exp(logWeights[k])
			<< "\n" << gaussians[k] << std::endl;
	}
#endif

	//temporary memory
	Eigen::MatrixXd logW(numPoints, components);

	//run EM (fixed number of iterations)
	std::cout << "Run EM algorithm" << std::endl;
	std::chrono::time_point<std::chrono::steady_clock> start;
	double logLikeliehoodAccum;
	for (int iter=0; iter < numIterations; ++iter)
	{
		//half of the iterations for warm up
		if (iter==numIterations/2) start = std::chrono::steady_clock::now();

#ifndef NDEBUG
		std::cout << "    Iteration " << iter << std::endl;
#endif

		//Precomputation
#pragma omp parallel for
		for (int i = 0; i < components; ++i)
			gaussians[i].computeDetInv();

		//E-Step
		logLikeliehoodAccum = 0;
		for (int i=0; i<numPoints; ++i)
		{
			//compute membership weight w_ik
			for (int k=0; k<components; ++k)
			{
				double lw = gaussians[k].logP(points.col(i)) + logWeights[k];
				logW(i, k) = lw;
			}
			double lse = LSE(logW.row(i));
			logW.row(i) -= Eigen::RowVectorXd::Constant(components, lse);
			logLikeliehoodAccum += lse;

#ifndef NDEBUG
			std::cout << "Point " << i << ":";
			for (int k = 0; k < components; ++k)
				std::cout << " " << std::exp(logW(i, k));
			std::cout << std::endl;
#endif
		}

		//M-Step
		for (int k=0; k<components; ++k)
		{
#ifndef NDEBUG
			std::cout << "\nComponent " << k << " pre-update:"
				<< "\nWeight: " << std::exp(logWeights[k])
				<< "\n" << gaussians[k] << std::endl;
#endif

			double logNk = LSE(logW.col(k));
			double divNk = 1.0f / std::exp(logNk);
			logWeights[k] = logNk - std::log(numPoints);
			//Eigen does not support batches, so I have to write it explicitly as a loop here
			//(and broadcasting is very ugly)
			gaussians[k].mean().setZero();
			gaussians[k].cov().setZero();
			for (int i = 0; i < numPoints; ++i)
				gaussians[k].mean() += std::exp(logW(i, k)) * points.col(i);
			gaussians[k].mean() *= divNk;
			for (int i = 0; i < numPoints; ++i)
				gaussians[k].cov() += std::exp(logW(i, k)) *
					(points.col(i) - gaussians[k].mean()) * (points.col(i) - gaussians[k].mean()).transpose();
			gaussians[k].cov() = gaussians[k].cov()*divNk 
			+ Eigen::MatrixXd::Identity(dimension, dimension) * 1e-1;

#ifndef NDEBUG
			std::cout << "Component " << k << " post-update:"
				<< "\nWeight: " << std::exp(logWeights[k])
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
	for (int k=0; k<components; ++k)
	{
		Json::Object com;
		com.Insert(std::make_pair("Weight", std::exp(logWeights[k])));
		Json::Array m, c;
		for (int i = 0; i < dimension; ++i) m.PushBack(gaussians[k].mean()[i]);
		for (int i = 0; i < dimension; ++i)
			for (int j = 0; j < dimension; ++j)
				c.PushBack(gaussians[k].cov()(i, j));
		com.Insert(std::make_pair("Mean", m));
		com.Insert(std::make_pair("Cov", c));
		comData.PushBack(com);
	}
	returnValues.Insert(std::make_pair("Components", comData));
}