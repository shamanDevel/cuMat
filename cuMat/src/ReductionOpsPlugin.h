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

/**
* \brief Computes the product of all elements along the specified reduction axis
* \tparam axis the reduction axis, by default, reduction is performed among all axis
*/
template<int axis = ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch>
ReductionOp_StaticSwitched<_Derived, functor::Prod<Scalar>, axis> prod()
{
    return ReductionOp_StaticSwitched<_Derived, functor::Prod<Scalar>, axis>(derived(), functor::Prod<Scalar>(), 1);
}

/**
* \brief Computes the product of all elements along the specified reduction axis
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
ReductionOp_DynamicSwitched<_Derived, functor::Prod<Scalar>> prod(int axis)
{
    return ReductionOp_DynamicSwitched<_Derived, functor::Prod<Scalar>>(derived(), axis, functor::Prod<Scalar>(), 1);
}

/**
* \brief Computes the minimum value among all elements along the specified reduction axis
* \tparam axis the reduction axis, by default, reduction is performed among all axis
*/
template<int axis = ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch>
ReductionOp_StaticSwitched<_Derived, functor::Min<Scalar>, axis> minCoeff()
{
    return ReductionOp_StaticSwitched<_Derived, functor::Min<Scalar>, axis>(derived(), functor::Min<Scalar>(), std::numeric_limits<Scalar>::max());
}

/**
* \brief Computes the minimum value among all elements along the specified reduction axis
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
ReductionOp_DynamicSwitched<_Derived, functor::Min<Scalar>> minCoeff(int axis)
{
    return ReductionOp_DynamicSwitched<_Derived, functor::Min<Scalar>>(derived(), axis, functor::Min<Scalar>(), std::numeric_limits<Scalar>::max());
}

/**
* \brief Computes the maximum value among all elements along the specified reduction axis
* \tparam axis the reduction axis, by default, reduction is performed among all axis
*/
template<int axis = ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch>
ReductionOp_StaticSwitched<_Derived, functor::Max<Scalar>, axis> maxCoeff()
{
    return ReductionOp_StaticSwitched<_Derived, functor::Max<Scalar>, axis>(derived(), functor::Max<Scalar>(), std::numeric_limits<Scalar>::lowest());
}

/**
* \brief Computes the maximum value among all elements along the specified reduction axis
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
ReductionOp_DynamicSwitched<_Derived, functor::Max<Scalar>> maxCoeff(int axis)
{
    return ReductionOp_DynamicSwitched<_Derived, functor::Max<Scalar>>(derived(), axis, functor::Max<Scalar>(), std::numeric_limits<Scalar>::lowest());
}

/**
* \brief Computes the locical AND of all elements along the specified reduction axis,
* i.e. <b>all</b> values must be true for the result to be true.
* This is only defined for boolean matrices.
* \tparam axis the reduction axis, by default, reduction is performed among all axis
*/
template<int axis = ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch>
ReductionOp_StaticSwitched<_Derived, functor::LogicalAnd<Scalar>, axis> all()
{
    CUMAT_STATIC_ASSERT((std::is_same<Scalar, bool>::value), "'all' can only be applied to boolean matrices");
    return ReductionOp_StaticSwitched<_Derived, functor::LogicalAnd<Scalar>, axis>(derived(), functor::LogicalAnd<Scalar>(), true);
}

/**
* \brief Computes the logical AND of all elements along the specified reduction axis,
* i.e. <b>all</b> values must be true for the result to be true.
* This is only defined for boolean matrices.
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
ReductionOp_DynamicSwitched<_Derived, functor::LogicalAnd<Scalar>> all(int axis)
{
    CUMAT_STATIC_ASSERT((std::is_same<Scalar, bool>::value), "'all' can only be applied to boolean matrices");
    return ReductionOp_DynamicSwitched<_Derived, functor::LogicalAnd<Scalar>>(derived(), axis, functor::LogicalAnd<Scalar>(), true);
}

/**
* \brief Computes the locical OR of all elements along the specified reduction axis,
* i.e. <b>any</b> value must be true for the result to be true.
* This is only defined for boolean matrices.
* \tparam axis the reduction axis, by default, reduction is performed among all axis
*/
template<int axis = ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch>
ReductionOp_StaticSwitched<_Derived, functor::LogicalOr<Scalar>, axis> any()
{
    CUMAT_STATIC_ASSERT((std::is_same<Scalar, bool>::value), "'any' can only be applied to boolean matrices");
    return ReductionOp_StaticSwitched<_Derived, functor::LogicalOr<Scalar>, axis>(derived(), functor::LogicalOr<Scalar>(), false);
}

/**
* \brief Computes the logical OR of all elements along the specified reduction axis,
* i.e. <b>any</b> values must be true for the result to be true.
* This is only defined for boolean matrices.
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
ReductionOp_DynamicSwitched<_Derived, functor::LogicalOr<Scalar>> any(int axis)
{
    CUMAT_STATIC_ASSERT((std::is_same<Scalar, bool>::value), "'any' can only be applied to boolean matrices");
    return ReductionOp_DynamicSwitched<_Derived, functor::LogicalOr<Scalar>>(derived(), axis, functor::LogicalOr<Scalar>(), false);
}

/**
* \brief Computes the bitwise AND of all elements along the specified reduction axis.
* This is only defined for matrices of integer types.
* \tparam axis the reduction axis, by default, reduction is performed among all axis
*/
template<int axis = ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch>
ReductionOp_StaticSwitched<_Derived, functor::BitwiseAnd<Scalar>, axis> bitwiseAnd()
{
    CUMAT_STATIC_ASSERT(std::is_integral<Scalar>::value, "'bitwiseAnd' can only be applied to integral matrices");
    return ReductionOp_StaticSwitched<_Derived, functor::BitwiseAnd<Scalar>, axis>(derived(), functor::BitwiseAnd<Scalar>(), ~(Scalar(0)));
}

/**
* \brief Computes the logical AND of all elements along the specified reduction axis.
* This is only defined for matrices of integer types.
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
ReductionOp_DynamicSwitched<_Derived, functor::BitwiseAnd<Scalar>> bitwiseAnd(int axis)
{
    CUMAT_STATIC_ASSERT(std::is_integral<Scalar>::value, "'bitwiseAnd' can only be applied to integral matrices");
    return ReductionOp_DynamicSwitched<_Derived, functor::BitwiseAnd<Scalar>>(derived(), axis, functor::BitwiseAnd<Scalar>(), ~(Scalar(0)));
}

/**
* \brief Computes the bitwise OR of all elements along the specified reduction axis.
* This is only defined for matrices of integer types.
* \tparam axis the reduction axis, by default, reduction is performed among all axis
*/
template<int axis = ReductionAxis::Row | ReductionAxis::Column | ReductionAxis::Batch>
ReductionOp_StaticSwitched<_Derived, functor::BitwiseOr<Scalar>, axis> bitwiseOr()
{
    CUMAT_STATIC_ASSERT(std::is_integral<Scalar>::value, "'bitwiseOr' can only be applied to integral matrices");
    return ReductionOp_StaticSwitched<_Derived, functor::BitwiseOr<Scalar>, axis>(derived(), functor::BitwiseOr<Scalar>(), Scalar(0));
}

/**
* \brief Computes the logical OR of all elements along the specified reduction axis.
* This is only defined for matrices of integer types.
* \param axis the reduction axis, by default, reduction is performed among all axis
*/
ReductionOp_DynamicSwitched<_Derived, functor::BitwiseOr<Scalar>> bitwiseOr(int axis)
{
    CUMAT_STATIC_ASSERT(std::is_integral<Scalar>::value, "'bitwiseOr' can only be applied to integral matrices");
    return ReductionOp_DynamicSwitched<_Derived, functor::BitwiseOr<Scalar>>(derived(), axis, functor::BitwiseOr<Scalar>(), Scalar(0));
}