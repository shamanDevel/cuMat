#ifndef __CUMAT_REDUCTION_ALGORITHM_SELECTION_H__
#define __CUMAT_REDUCTION_ALGORITHM_SELECTION_H__

#include <tuple>
#include <array>

#include "Macros.h"
#include "ForwardDeclarations.h"

CUMAT_NAMESPACE_BEGIN

namespace internal
{
	/**
	 * \brief Algorithms possible during dynamic selection.
	 * These are a subset of the tags from namespace ReductionAlg.
	 */
	enum class ReductionAlgorithm
	{
		Segmented,
		Thread,
		Warp,
		Block256,
		Device1,
		Device2,
		Device4
	};

	/**
	 * \brief Selects the best reduction algorithm dynamically given
	 * the reduction axis (inner, middle, outer).
	 * Given a matrix in ColumnMajor order, these axis
	 * correspond to the following: inner=Row, middle=Column, outer=Batch.
	 * 
	 * The timings from which those selections were determined were evaluated 
	 * on a Nvidia RTX 2070.
	 * For a different architecture, you might need to tweak the conditions
	 * in the source code.
	 */
	struct ReductionAlgorithmSelection
	{
	private:
		typedef std::tuple<double, double, double> condition;
		static constexpr int MAX_CONDITION = 3;
		typedef std::array<condition, MAX_CONDITION> conditions;
		typedef std::tuple<ReductionAlgorithm, int, conditions> choice;

		template<int N>
		static ReductionAlgorithm select(
			const choice(&conditions)[N], ReductionAlgorithm def,
			Index numBatches, Index batchSize)
		{
			const double nb = std::log2(numBatches);
			const double bs = std::log2(batchSize);
			for (const choice& c : conditions)
			{
				bool success = true;
				for (int i = 0; i < std::get<1>(c); ++i) {
					const auto& cond = std::get<2>(c)[i];
					if (std::get<0>(cond)*nb + std::get<1>(cond)*bs < std::get<2>(cond))
						success = false;
				}
				if (success)
					return std::get<0>(c);
			}
			return def;
		}

	public:
		static ReductionAlgorithm inner(Index numBatches, Index batchSize)
		{
			//adopt to new architectures
			static const choice CONDITONS[] = {
				choice{ReductionAlgorithm::Device1, 2, {condition{1.2, 1.0, 19.5}, condition{-1,0,-2.5}}},
				choice{ReductionAlgorithm::Device2, 3, {condition{0.42857142857142855,1,17.821428571428573}, condition{-1,0,-4.25}, condition{1,0,2.5}}},
				choice{ReductionAlgorithm::Device4, 3, {condition{0,1,16.25}, condition{-1,0,-5.5}, condition{1,0,4.25}}},
				choice{ReductionAlgorithm::Block256,2, {condition{-1.6, 1, 8}, condition{-1,0,-5}}},
				choice{ReductionAlgorithm::Thread,  2, {condition{0.475, -1, 2.01875}, condition{0, -1, -4.75}}}
			};
			static const ReductionAlgorithm DEFAULT = ReductionAlgorithm::Warp;
			return select(CONDITONS, DEFAULT, numBatches, batchSize);
		}

		static ReductionAlgorithm middle(Index numBatches, Index batchSize)
		{
			//adopt to new architectures
			static const choice CONDITONS[] = {
				choice{ReductionAlgorithm::Device1, 2, {condition{1.5,1,19.5}, condition{-1,0,-2.5}}},
				choice{ReductionAlgorithm::Device2, 3, {condition{0,1,15.5}, condition{1,0,2.5}, condition{-1,0,-4}}},
				choice{ReductionAlgorithm::Device4, 3, {condition{0,1,15.75}, condition{1,0,4}, condition{-1,0,-5.75}}},
				choice{ReductionAlgorithm::Block256,2, {condition{0,1,9}, condition{-1,0,-2.5}}},
				choice{ReductionAlgorithm::Warp,  2, {condition{0,1,4}, condition{-1,0,-11.75}}}
			};
			static const ReductionAlgorithm DEFAULT = ReductionAlgorithm::Thread;
			return select(CONDITONS, DEFAULT, numBatches, batchSize);
		}

		static ReductionAlgorithm outer(Index numBatches, Index batchSize)
		{
			//adopt to new architectures
			static const choice CONDITONS[] = {
				choice{ReductionAlgorithm::Device1, 2, {condition{-1,0,-2}, condition{1.875,1,19}}},
				choice{ReductionAlgorithm::Device4, 3, {condition{1,0,2}, condition{-1,0,-4.25}, condition{10, 9, 184.25}}},
				choice{ReductionAlgorithm::Device2, 3, {condition{1,0,2}, condition{-1,0,-4.25}, condition{-0.22222, 1, 14.085555}}},
				choice{ReductionAlgorithm::Segmented, 3, {condition{1,0,4}, condition{0,1,11.5}, condition{-1,0,-8.5}}},
				choice{ReductionAlgorithm::Block256,2, {condition{0,1,8}, condition{-1,0,-2}}},
				choice{ReductionAlgorithm::Warp,  2, {condition{0,1,2.75}, condition{-1,0,-11.75}}}
			};
			static const ReductionAlgorithm DEFAULT = ReductionAlgorithm::Thread;
			return select(CONDITONS, DEFAULT, numBatches, batchSize);
		}
	};
}

CUMAT_NAMESPACE_END

#endif