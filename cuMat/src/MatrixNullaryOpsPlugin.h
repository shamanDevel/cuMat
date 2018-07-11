//included inside of Matrix




template<typename _NullaryFunctor>
using NullaryOp_t = NullaryOp<_Scalar, _Rows, _Columns, _Batches, _Flags, _NullaryFunctor >;

/**
* \brief Creates a new matrix with all entries set to a constant value
* \param rows the number of rows
* \param cols the number of columns
* \param batches the number of batches
* \param value the value to fill
* \return the expression creating that matrix
*/
static NullaryOp_t<functor::ConstantFunctor<_Scalar> >
Constant(Index rows, Index cols, Index batches, const _Scalar& value)
{
    if (_Rows != Dynamic) CUMAT_ASSERT_ARGUMENT(_Rows == rows && "runtime row count does not match compile time row count");
    if (_Columns != Dynamic) CUMAT_ASSERT_ARGUMENT(_Columns == cols && "runtime row count does not match compile time row count");
    if (_Batches != Dynamic) CUMAT_ASSERT_ARGUMENT(_Batches == batches && "runtime row count does not match compile time row count");
    return NullaryOp_t<functor::ConstantFunctor<_Scalar> >(
        rows, cols, batches, functor::ConstantFunctor<_Scalar>(value));
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
template<typename T = std::enable_if<_Batches != Dynamic && _Rows == Dynamic && _Columns == Dynamic,
    NullaryOp_t<functor::ConstantFunctor<_Scalar> > > >
    static typename T::type Constant(Index rows, Index cols, const _Scalar& value)
{
    return NullaryOp_t<functor::ConstantFunctor<_Scalar> >(
        rows, cols, _Batches, functor::ConstantFunctor<_Scalar>(value));
}

/**
* \brief Creates a new vector with all entries set to a constant value.
* This version is only available if the number of batches is fixed on compile-time,
* and either rows or columns are fixed on compile time.
* \param size the size of the matrix along the free dimension
* \param value the value to fill
* \return the expression creating that matrix
*/
template<typename T = std::enable_if<_Batches != Dynamic
    && ((_Rows == Dynamic && _Columns != Dynamic) || (_Rows != Dynamic && _Columns == Dynamic)),
    NullaryOp_t<functor::ConstantFunctor<_Scalar> > > >
    static typename T::type Constant(Index size, const _Scalar& value)
{
    return NullaryOp_t<functor::ConstantFunctor<_Scalar> >(
        _Rows == Dynamic ? size : _Rows,
        _Columns == Dynamic ? size : _Columns,
        _Batches, functor::ConstantFunctor<_Scalar>(value));
}

/**
* \brief Creates a new matrix with all entries set to a constant value.
* This version is only available if all sized (row, column, batch) are fixed on compile-time.
* \param value the value to fill
* \return the expression creating that matrix
*/
template<typename T = std::enable_if<_Batches != Dynamic && _Rows != Dynamic && _Columns != Dynamic,
    NullaryOp_t<functor::ConstantFunctor<_Scalar> > > >
    static typename T::type Constant(const _Scalar& value)
{
    return NullaryOp_t<functor::ConstantFunctor<_Scalar> >(
        _Rows, _Columns, _Batches, functor::ConstantFunctor<_Scalar>(value));
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
static NullaryOp_t<functor::IdentityFunctor<_Scalar> >
Identity(Index rows, Index cols, Index batches)
{
    if (_Rows != Dynamic) CUMAT_ASSERT_ARGUMENT(_Rows == rows && "runtime row count does not match compile time row count");
    if (_Columns != Dynamic) CUMAT_ASSERT_ARGUMENT(_Columns == cols && "runtime row count does not match compile time row count");
    if (_Batches != Dynamic) CUMAT_ASSERT_ARGUMENT(_Batches == batches && "runtime row count does not match compile time row count");
    return NullaryOp_t<functor::IdentityFunctor<_Scalar> >(
        rows, cols, batches, functor::IdentityFunctor<_Scalar>());
}

/**
* \brief Generalized identity matrix.
* This version is only available if the number of batches is known on compile-time and rows and columns are dynamic.
* \param rows the number of rows
* \param cols the number of columns.
* \return  the operation that computes the identity matrix
*/
template<typename T = std::enable_if<_Batches != Dynamic && _Rows == Dynamic && _Columns == Dynamic,
    NullaryOp_t<functor::IdentityFunctor<_Scalar> > > >
    static typename T::type Identity(Index rows, Index cols)
{
    return NullaryOp_t<functor::IdentityFunctor<_Scalar> >(
        rows, cols, _Batches, functor::IdentityFunctor<_Scalar>());
}
/**
* \brief Creates a square identity matrix.
* This version is only available if the number of batches is known on compile-time and rows and columns are dynamic.
* \param size the size of the matrix
* \return the operation that computes the identity matrix
*/
template<typename T = std::enable_if<_Batches != Dynamic
    && (_Rows == Dynamic && _Columns == Dynamic),
    NullaryOp_t<functor::IdentityFunctor<_Scalar> > > >
    static typename T::type Identity(Index size)
{
    return NullaryOp_t<functor::IdentityFunctor<_Scalar> >(
        size, size, _Batches, functor::IdentityFunctor<_Scalar>());
}
/**
* \brief Creates the identity matrix.
* This version is only available if the number of rows, columns and batches are available at compile-time.
* Note that the matrix must not necessarily be square.
* \return  the operation that computes the identity matrix
*/
template<typename T = std::enable_if<_Batches != Dynamic && _Rows != Dynamic && _Columns != Dynamic,
    NullaryOp_t<functor::IdentityFunctor<_Scalar> > > >
static typename T::type Identity()
{
    return NullaryOp_t<functor::IdentityFunctor<_Scalar> >(
        _Rows, _Columns, _Batches, functor::IdentityFunctor<_Scalar>());
}


/**
* \brief Creates a new matrix expression with all entries set to zero
* \param rows the number of rows
* \param cols the number of columns
* \param batches the number of batches
* \return the expression creating that matrix
*/
static NullaryOp_t<functor::ConstantFunctor<_Scalar> >
Zero(Index rows, Index cols, Index batches)
{
    if (_Rows != Dynamic) CUMAT_ASSERT_ARGUMENT(_Rows == rows && "runtime row count does not match compile time row count");
    if (_Columns != Dynamic) CUMAT_ASSERT_ARGUMENT(_Columns == cols && "runtime row count does not match compile time row count");
    if (_Batches != Dynamic) CUMAT_ASSERT_ARGUMENT(_Batches == batches && "runtime row count does not match compile time row count");
    return NullaryOp_t<functor::ConstantFunctor<_Scalar> >(
        rows, cols, batches, functor::ConstantFunctor<_Scalar>(_Scalar(0)));
}
//Specialization for some often used cases

/**
* \brief Creates a new matrix expression with all entries set to zero.
* This version is only available if the number of batches is fixed on compile-time.
* \param rows the number of rows
* \param cols the number of columns
* \return the expression creating that matrix
*/
template<typename T = std::enable_if<_Batches != Dynamic && _Rows == Dynamic && _Columns == Dynamic,
    NullaryOp_t<functor::ConstantFunctor<_Scalar> > > >
static typename T::type Zero(Index rows, Index cols)
{
    return NullaryOp_t<functor::ConstantFunctor<_Scalar> >(
        rows, cols, _Batches, functor::ConstantFunctor<_Scalar>(_Scalar(0)));
}

/**
* \brief Creates a new vector with all entries set to zero.
* This version is only available if the number of batches is fixed on compile-time,
* and either rows or columns are fixed on compile time.
* \param size the size of the matrix along the free dimension
* \return the expression creating that matrix
*/
template<typename T = std::enable_if<_Batches != Dynamic
    && ((_Rows == Dynamic && _Columns != Dynamic) || (_Rows != Dynamic && _Columns == Dynamic)),
    NullaryOp_t<functor::ConstantFunctor<_Scalar> > > >
static typename T::type Zero(Index size)
{
    return NullaryOp_t<functor::ConstantFunctor<_Scalar> >(
        _Rows == Dynamic ? size : _Rows,
        _Columns == Dynamic ? size : _Columns,
        _Batches, functor::ConstantFunctor<_Scalar>(_Scalar(0)));
}

/**
* \brief Creates a new matrix with all entries set to zero.
* This version is only available if all sized (row, column, batch) are fixed on compile-time.
* \param value the value to fill
* \return the expression creating that matrix
*/
template<typename T = std::enable_if<_Batches != Dynamic && _Rows != Dynamic && _Columns != Dynamic,
    NullaryOp_t<functor::ConstantFunctor<_Scalar> > > >
static typename T::type Zero()
{
    return NullaryOp_t<functor::ConstantFunctor<_Scalar> >(
        _Rows, _Columns, _Batches, functor::ConstantFunctor<_Scalar>(_Scalar(0)));
}