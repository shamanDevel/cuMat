#ifndef __CUMAT_IO_H__
#define __CUMAT_IO_H__

#include "Macros.h"
#include "ForwardDeclarations.h"

#include <vector>
#include <ostream>

CUMAT_NAMESPACE_BEGIN

//This code is copied from Eigen's IO.h

namespace io
{
    enum { DontAlignCols = 1 };
    enum {
        StreamPrecision = -1,
        FullPrecision = -2
    };

    /**
  * \brief Stores a set of parameters controlling the way matrices are printed
  *
  * List of available parameters:
  *  - \b precision number of digits for floating point values, or one of the special constants \c StreamPrecision and \c FullPrecision.
  *                 The default is the special value \c StreamPrecision which means to use the
  *                 stream's own precision setting, as set for instance using \c cout.precision(3). The other special value
  *                 \c FullPrecision means that the number of digits will be computed to match the full precision of each floating-point
  *                 type.
  *  - \b flags an OR-ed combination of flags, the default value is 0, the only currently available flag is \c DontAlignCols which
  *             allows to disable the alignment of columns, resulting in faster code.
  *  - \b coeffSeparator string printed between two coefficients of the same row
  *  - \b rowSeparator string printed between two rows
  *  - \b rowPrefix string printed at the beginning of each row
  *  - \b rowSuffix string printed at the end of each row
  *  - \b matPrefix string printed at the beginning of the matrix
  *  - \b matSuffix string printed at the end of the matrix
  *  - \b fill character printed to fill the empty space in aligned columns

  */
    struct IOFormat
    {
        /** Default constructor, see class IOFormat for the meaning of the parameters */
        IOFormat(int _precision = StreamPrecision, int _flags = 0,
            const std::string& _coeffSeparator = " ",
            const std::string& _rowSeparator = "\n", const std::string& _rowPrefix = "", const std::string& _rowSuffix = "",
            const std::string& _matPrefix = "", const std::string& _matSuffix = "", const char _fill = ' ',
            const std::string& _batchPrefix = "[\n", const std::string& _batchSuffix="\n]", const std::string& _batchSeparator = "\n],[\n")
            : matPrefix(_matPrefix), matSuffix(_matSuffix), rowPrefix(_rowPrefix), rowSuffix(_rowSuffix), rowSeparator(_rowSeparator),
            rowSpacer(""), coeffSeparator(_coeffSeparator), fill(_fill), precision(_precision), flags(_flags),
    	    batchPrefix(_batchPrefix), batchSuffix(_batchSuffix), batchSeparator(_batchSeparator)
        {
            // TODO check if rowPrefix, rowSuffix or rowSeparator contains a newline
            // don't add rowSpacer if columns are not to be aligned
            if ((flags & DontAlignCols))
                return;
            int i = int(matSuffix.length()) - 1;
            while (i >= 0 && matSuffix[i] != '\n')
            {
                rowSpacer += ' ';
                i--;
            }
        }
        std::string matPrefix, matSuffix;
        std::string rowPrefix, rowSuffix, rowSeparator, rowSpacer;
        std::string coeffSeparator;
        char fill;
        int precision;
        int flags;
        std::string batchPrefix;
        std::string batchSuffix;
        std::string batchSeparator;
    };

    namespace internal
    {
    	template<typename T>
	    struct TypeCast
    	{
            typedef T Type;
    	};
    	template<>
    	struct TypeCast<bool>
    	{
            typedef uint8_t Type;
    	};
    }
	
	/**
     * \brief Prints the matrix _m into stream s.
     * This code is copied from Eigen's IO.h
     * to avoid an Eigen-dependency when printing the matrices.
     * 
     * \param s 
     * \param _m 
     * \param transposed 
     * \return 
     */
    template <typename _Scalar, int _Rows, int _Columns, int _Batches, int _Flags>
    __host__ std::ostream& print_matrix(
        std::ostream& s, 
        const Matrix<_Scalar, _Rows, _Columns, _Batches, _Flags>& _m, 
        const IOFormat& fmt, bool transposed = false)
    {
        if (_m.size() == 0)
        {
            s << fmt.matPrefix << fmt.matSuffix;
            return s;
        }

        typedef typename internal::TypeCast<_Scalar>::Type Scalar;

        Index batches = _m.batches();
        Index rows = _m.rows();
        Index cols = _m.cols();
        std::vector<Scalar> data(rows * cols * batches);
        _m.copyToHost(reinterpret_cast<_Scalar*>(&data[0]));

        Index width = 0;
        std::streamsize explicit_precision;
        if (fmt.precision == StreamPrecision)
        {
            explicit_precision = 0;
        }
        else if (fmt.precision == FullPrecision)
        {
            if (std::numeric_limits<_Scalar>::is_integer)
            {
                explicit_precision = 0;
            }
            else
            {
                explicit_precision = std::numeric_limits<_Scalar>::max_digits10;
            }
        }
        else
        {
            explicit_precision = fmt.precision;
        }

        std::streamsize old_precision = 0;
        if (explicit_precision) old_precision = s.precision(explicit_precision);

        bool align_cols = !(fmt.flags & DontAlignCols);
        if (align_cols)
        {
            // compute the largest width
        	for (Index k = 0; k < batches; ++k)
	            for (Index j = 0; j < cols; ++j)
	                for (Index i = 0; i < rows; ++i)
	                {
	                    std::stringstream sstr;
	                    sstr.copyfmt(s);
	                    sstr << data[_m.index(i, j, k)];
	                    width = std::max<Index>(width, Index(sstr.str().length()));
	                }
        }
        std::streamsize old_width = s.width();
        char old_fill_character = s.fill();

    	//actual printing
        if (batches > 1) s << fmt.batchPrefix;
        for (Index k = 0; k < batches; ++k)
        {
            if (k > 0) s << fmt.batchSeparator;
            s << fmt.matPrefix;
            if (transposed)
            {
                for (Index i = 0; i < cols; ++i)
                {
                    if (i)
                        s << fmt.rowSpacer;
                    s << fmt.rowPrefix;
                    if (width) {
                        s.fill(fmt.fill);
                        s.width(width);
                    }
                    s << data[_m.index(0, i, k)];
                    for (Index j = 1; j < rows; ++j)
                    {
                        s << fmt.coeffSeparator;
                        if (width) {
                            s.fill(fmt.fill);
                            s.width(width);
                        }
                        s << data[_m.index(j, i, k)];
                    }
                    s << fmt.rowSuffix;
                    if (i < cols - 1)
                        s << fmt.rowSeparator;
                }
            }
            else
            {
                for (Index i = 0; i < rows; ++i)
                {
                    if (i)
                        s << fmt.rowSpacer;
                    s << fmt.rowPrefix;
                    if (width) {
                        s.fill(fmt.fill);
                        s.width(width);
                    }
                    s << data[_m.index(i, 0, k)];
                    for (Index j = 1; j < cols; ++j)
                    {
                        s << fmt.coeffSeparator;
                        if (width) {
                            s.fill(fmt.fill);
                            s.width(width);
                        }
                        s << data[_m.index(i, j, k)];
                    }
                    s << fmt.rowSuffix;
                    if (i < rows - 1)
                        s << fmt.rowSeparator;
                }
            }
            s << fmt.matSuffix;
        }
        if (batches > 1) s << fmt.batchSuffix;
        if (explicit_precision) s.precision(old_precision);
        if (width) {
            s.fill(old_fill_character);
            s.width(old_width);
        }
        return s;
    }
}

CUMAT_NAMESPACE_END

#endif