//Included inside MatrixBase, define the accessors

/**
 * \brief Computes the sum of all elements along the specified reduction axis
 * \tparam axis the reduction axis, by default, reduction is performed among all axis
 */
template<int axis = ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch>
ReductionOp_StaticSwitched<_Derived, functor::Sum<Scalar>, axis> sum()
{
    return ReductionOp_StaticSwitched<_Derived, functor::Sum<Scalar>, axis>(derived(), functor::Sum<Scalar>(), 0);
}

/**
* \brief Computes the sum of all elements along the specified reduction axis
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
ReductionOp_DynamicSwitched<_Derived, functor::Sum<Scalar>> sum(int axis)
{
    return ReductionOp_DynamicSwitched<_Derived, functor::Sum<Scalar>>(derived(), axis, functor::Sum<Scalar>(), 0);
}