//included inside of Matrix




template<typename _NullaryFunctor>
using NullaryOp_t = NullaryOp<Scalar, Rows, Columns, Batches, Flags, _NullaryFunctor >;

/**
* \brief Creates a new matrix with all entries set to a constant value
* \param rows the number of rows
* \param cols the number of columns
* \param batches the number of batches
* \param value the value to fill
* \return the expression creating that matrix
*/
static NullaryOp_t<functor::ConstantFunctor<Scalar> >
Constant(Index rows, Index cols, Index batches, const Scalar& value)
{
    if (Rows != Dynamic) CUMAT_ASSERT_ARGUMENT(Rows == rows && "runtime row count does not match compile time row count");
    if (Columns != Dynamic) CUMAT_ASSERT_ARGUMENT(Columns == cols && "runtime row count does not match compile time row count");
    if (Batches != Dynamic) CUMAT_ASSERT_ARGUMENT(Batches == batches && "runtime row count does not match compile time row count");
    return NullaryOp_t<functor::ConstantFunctor<Scalar> >(
        rows, cols, batches, functor::ConstantFunctor<Scalar>(value));
}
//Specialization for some often used cases

/**
* \brief Creates a new matrix with all entries set to a constant value.
* This version is only available if the number of batches is fixed on compile-time.
* \param rows the number of rows
* \param cols the number of columns
* \param value the value to fill
* \return the expression creating that matrix
*/
static NullaryOp_t<functor::ConstantFunctor<Scalar> >
Constant(Index rows, Index cols, const Scalar& value)
{
    CUMAT_STATIC_ASSERT(Batches != Dynamic, "Number of batches must be fixed on compile-time");
    if (Rows != Dynamic) CUMAT_ASSERT_ARGUMENT(Rows == rows && "runtime row count does not match compile time row count");
    if (Columns != Dynamic) CUMAT_ASSERT_ARGUMENT(Columns == cols && "runtime row count does not match compile time row count");
    return NullaryOp_t<functor::ConstantFunctor<Scalar> >(
        rows, cols, Batches, functor::ConstantFunctor<Scalar>(value));
}

/**
* \brief Creates a new vector with all entries set to a constant value.
* This version is only available if the number of batches is fixed on compile-time,
* and either rows or columns are fixed on compile time.
* \param size the size of the matrix along the free dimension
* \param value the value to fill
* \return the expression creating that matrix
*/
static NullaryOp_t<functor::ConstantFunctor<Scalar> >
Constant(Index size, const Scalar& value)
{
    CUMAT_STATIC_ASSERT(Batches != Dynamic
        && ((Rows == Dynamic && Columns != Dynamic) || (Rows != Dynamic && Columns == Dynamic)),
        "Matrix must have a fixed compile-time batch size and compile-time row or column count (a vector)");
    return NullaryOp_t<functor::ConstantFunctor<Scalar> >(
        Rows == Dynamic ? size : Rows,
        Columns == Dynamic ? size : Columns,
        Batches, functor::ConstantFunctor<Scalar>(value));
}

/**
* \brief Creates a new matrix with all entries set to a constant value.
* This version is only available if all sized (row, column, batch) are fixed on compile-time.
* \param value the value to fill
* \return the expression creating that matrix
*/
static NullaryOp_t<functor::ConstantFunctor<Scalar> > 
Constant(const Scalar& value)
{
    CUMAT_STATIC_ASSERT(Batches != Dynamic && Rows != Dynamic && Columns != Dynamic,
        "All dimensions must be fixed on compile-time for this function to be available");
    return NullaryOp_t<functor::ConstantFunctor<Scalar> >(
        Rows, Columns, Batches, functor::ConstantFunctor<Scalar>(value));
}



/**
* \brief generalized identity matrix.
* This matrix contains ones along the main diagonal and zeros everywhere else.
* The matrix must not necessarily be square.
* \param rows the number of rows
* \param cols the number of columns
* \param batches the number of batches
* \return the operation that computes the identity matrix
*/
static NullaryOp_t<functor::IdentityFunctor<Scalar> >
Identity(Index rows, Index cols, Index batches)
{
    if (Rows != Dynamic) CUMAT_ASSERT_ARGUMENT(Rows == rows && "runtime row count does not match compile time row count");
    if (Columns != Dynamic) CUMAT_ASSERT_ARGUMENT(Columns == cols && "runtime row count does not match compile time row count");
    if (Batches != Dynamic) CUMAT_ASSERT_ARGUMENT(Batches == batches && "runtime row count does not match compile time row count");
    return NullaryOp_t<functor::IdentityFunctor<Scalar> >(
        rows, cols, batches, functor::IdentityFunctor<Scalar>());
}

/**
* \brief Generalized identity matrix.
* This version is only available if the number of batches is known on compile-time and rows and columns are dynamic.
* \param rows the number of rows
* \param cols the number of columns.
* \return  the operation that computes the identity matrix
*/
static NullaryOp_t<functor::IdentityFunctor<Scalar> >
Identity(Index rows, Index cols)
{
    CUMAT_STATIC_ASSERT(Batches != Dynamic, "Number of batches must be fixed on compile-time");
    if (Rows != Dynamic) CUMAT_ASSERT_ARGUMENT(Rows == rows && "runtime row count does not match compile time row count");
    if (Columns != Dynamic) CUMAT_ASSERT_ARGUMENT(Columns == cols && "runtime row count does not match compile time row count");
    return NullaryOp_t<functor::IdentityFunctor<Scalar> >(
        rows, cols, Batches, functor::IdentityFunctor<Scalar>());
}
/**
* \brief Creates a square identity matrix.
* This version is only available if the number of batches is known on compile-time and rows and columns are dynamic.
* \param size the size of the matrix
* \return the operation that computes the identity matrix
*/
static NullaryOp_t<functor::IdentityFunctor<Scalar> >
Identity(Index size)
{
    CUMAT_STATIC_ASSERT(Batches != Dynamic
        && (Rows == Dynamic && Columns == Dynamic),
        "This function can only be called on dynamic sized matrices with a fixed batch size");
    return NullaryOp_t<functor::IdentityFunctor<Scalar> >(
        size, size, Batches, functor::IdentityFunctor<Scalar>());
}
/**
* \brief Creates the identity matrix.
* This version is only available if the number of rows, columns and batches are available at compile-time.
* Note that the matrix must not necessarily be square.
* \return  the operation that computes the identity matrix
*/
static NullaryOp_t<functor::IdentityFunctor<Scalar> >
Identity()
{
    CUMAT_STATIC_ASSERT(Batches != Dynamic && Rows != Dynamic && Columns != Dynamic,
        "This function can only be called on matrices with all dimensions fixed on compile-time");
    return NullaryOp_t<functor::IdentityFunctor<Scalar> >(
        Rows, Columns, Batches, functor::IdentityFunctor<Scalar>());
}


/**
* \brief Creates a new matrix expression with all entries set to zero
* \param rows the number of rows
* \param cols the number of columns
* \param batches the number of batches
* \return the expression creating that matrix
*/
static NullaryOp_t<functor::ConstantFunctor<Scalar> >
Zero(Index rows, Index cols, Index batches)
{
    if (Rows != Dynamic) CUMAT_ASSERT_ARGUMENT(Rows == rows && "runtime row count does not match compile time row count");
    if (Columns != Dynamic) CUMAT_ASSERT_ARGUMENT(Columns == cols && "runtime row count does not match compile time row count");
    if (Batches != Dynamic) CUMAT_ASSERT_ARGUMENT(Batches == batches && "runtime row count does not match compile time row count");
    return NullaryOp_t<functor::ConstantFunctor<Scalar> >(
        rows, cols, batches, functor::ConstantFunctor<Scalar>(_Scalar(0)));
}
//Specialization for some often used cases

/**
* \brief Creates a new matrix expression with all entries set to zero.
* This version is only available if the number of batches is fixed on compile-time.
* \param rows the number of rows
* \param cols the number of columns
* \return the expression creating that matrix
*/
static NullaryOp_t<functor::ConstantFunctor<Scalar> >
Zero(Index rows, Index cols)
{
    CUMAT_STATIC_ASSERT(Batches != Dynamic, "Number of batches must be fixed on compile-time");
    if (Rows != Dynamic) CUMAT_ASSERT_ARGUMENT(Rows == rows && "runtime row count does not match compile time row count");
    if (Columns != Dynamic) CUMAT_ASSERT_ARGUMENT(Columns == cols && "runtime row count does not match compile time row count");
    return NullaryOp_t<functor::ConstantFunctor<Scalar> >(
        rows, cols, _Batches, functor::ConstantFunctor<Scalar>(Scalar(0)));
}

/**
* \brief Creates a new vector with all entries set to zero.
* This version is only available if the number of batches is fixed on compile-time,
* and either rows or columns are fixed on compile time.
* \param size the size of the matrix along the free dimension
* \return the expression creating that matrix
*/
static NullaryOp_t<functor::ConstantFunctor<Scalar> >
Zero(Index size)
{
    CUMAT_STATIC_ASSERT(Batches != Dynamic
        && ((Rows == Dynamic && Columns != Dynamic) || (Rows != Dynamic && Columns == Dynamic)),
        "Matrix must have a fixed compile-time batch size and compile-time row or column count (a vector)");
    return NullaryOp_t<functor::ConstantFunctor<Scalar> >(
        Rows == Dynamic ? size : Rows,
        Columns == Dynamic ? size : Columns,
        Batches, functor::ConstantFunctor<Scalar>(Scalar(0)));
}

/**
* \brief Creates a new matrix with all entries set to zero.
* This version is only available if all sized (row, column, batch) are fixed on compile-time.
* \param value the value to fill
* \return the expression creating that matrix
*/
static NullaryOp_t<functor::ConstantFunctor<Scalar> >
Zero()
{
    CUMAT_STATIC_ASSERT(Batches != Dynamic && Rows != Dynamic && Columns != Dynamic,
        "All dimensions must be fixed on compile-time for this function to be available");
    return NullaryOp_t<functor::ConstantFunctor<Scalar> >(
        Rows, Columns, Batches, functor::ConstantFunctor<Scalar>(Scalar(0)));
}

/**
 * \brief Custom nullary expression.
 * The nullary functor must look as follow:
 * \code
 * struct MyFunctor
 * {
 *     typedef OutputType ReturnType;
 *     __device__ CUMAT_STRONG_INLINE ReturnType operator()(Index row, Index col, Index batch) const
 *     {
 *         return ...
 *     }
 * };
 * \endcode
 * with \c OutputType being the type of the matrix on which this nullary op is called.
 */
template<typename Functor>
static NullaryOp_t<Functor>
NullaryExpr(Index rows, Index cols, Index batches, const Functor& functor = Functor())
{
	if (Rows != Dynamic) CUMAT_ASSERT_ARGUMENT(Rows == rows && "runtime row count does not match compile time row count");
	if (Columns != Dynamic) CUMAT_ASSERT_ARGUMENT(Columns == cols && "runtime row count does not match compile time row count");
	if (Batches != Dynamic) CUMAT_ASSERT_ARGUMENT(Batches == batches && "runtime row count does not match compile time row count");
	CUMAT_STATIC_ASSERT((std::is_same<typename Functor::ReturnType, Scalar>::value), "Functor must return the same type as the matrix it is called on");
	return NullaryOp_t<Functor>(
		rows, cols, batches, functor);
}
