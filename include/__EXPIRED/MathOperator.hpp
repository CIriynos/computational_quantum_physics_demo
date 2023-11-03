#ifndef __MATH_OPERATOR_H__
#define __MATH_OPERATOR_H__

#include "NumericalGrid.hpp"
#include "Wavefunction.hpp"
#include "Eigen/Sparse"
#include "Eigen/Dense"

namespace CQP {

typedef Eigen::SparseMatrix< std::complex<double> > SpMat;

template<unsigned N>
class MathOperator
{
public:
	virtual Wavefunction<N> operator*(const Wavefunction<N>&) const = 0;
protected:
	NumericalGrid<N> grid;
};

template<unsigned N>
class MathOperatorMatrix : public MathOperator<N>
{
public:
	using MathOperator<N>::grid;

	MathOperatorMatrix() {}

	MathOperatorMatrix(const NumericalGrid<N>& grid_)
		: matrix() { grid = grid_; }

	MathOperatorMatrix(const NumericalGrid<N>& grid_, const SpMat& matrix_)
		: matrix(matrix_) 
	{
		assert(grid_.getTotalCount() == matrix_.rows());
		assert(matrix_.rows() == matrix_.cols());
		grid = grid_;
	}

	MathOperatorMatrix(const MathOperatorMatrix<N>& op)
		: matrix(op.matrix) { grid = op.grid; }

	MathOperatorMatrix(MathOperatorMatrix<N>&& op) noexcept
		: matrix(std::move(op.matrix)) { grid = op.grid; }

	MathOperatorMatrix<N>& operator=(const MathOperatorMatrix<N>& op) {
		grid = op.grid;
		matrix = op.matrix;
		return *this;
	}

	MathOperatorMatrix<N>& operator=(MathOperatorMatrix<N>&& op) noexcept {
		grid = std::move(op.grid);
		matrix = std::move(op.matrix);
		return *this;
	}

	virtual Wavefunction<N> operator*(const Wavefunction<N>& wave) const {
		assert(grid == wave.getGrid());
		return Wavefunction<N>(grid, matrix * wave.getSamplesView());
	}

	MathOperatorMatrix<N> operator*(const MathOperatorMatrix<N>& op) const {
		assert(grid == op.grid);
		return MathOperatorMatrix<N>(grid, matrix * op.matrix);
	}

	MathOperatorMatrix<N> operator*(std::complex<double> scaler) const {
		return MathOperatorMatrix<N>(grid, matrix * scaler);
	}

	MathOperatorMatrix<N> operator+(const MathOperatorMatrix<N>& op) const {
		assert(grid == op.grid);
		return MathOperatorMatrix<N>(grid, matrix + op.matrix);
	}

	MathOperatorMatrix<N> operator-(const MathOperatorMatrix<N>& op) const {
		assert(grid == op.grid);
		return MathOperatorMatrix<N>(grid, matrix - op.matrix);
	}

	MathOperatorMatrix<N> transpose() const { return MathOperatorMatrix<N>(grid, matrix.transpose()); }

	MathOperatorMatrix<N> adjoint() const { return MathOperatorMatrix<N>(grid, matrix.adjoint()); }

	MathOperatorMatrix<N> conjugate() const { return MathOperatorMatrix<N>(grid, matrix.conjugate()); }

	const NumericalGrid<N>& getGrid() const { return grid; }

	const SpMat& getMatrix() const { return matrix; }


protected:
	SpMat matrix;
};

typedef MathOperatorMatrix<1> MathOperatorMatrix1D;
typedef MathOperatorMatrix<2> MathOperatorMatrix2D;
typedef MathOperatorMatrix<3> MathOperatorMatrix3D;

template<unsigned N>
inline MathOperatorMatrix<N> createIdentityOperator(const NumericalGrid<N>& grid)
{
	typedef Eigen::Triplet< std::complex<double> > T;
	int cnt = grid.getTotalCount();
	SpMat mat(cnt, cnt);
	std::vector<T> tripletList;

	for (int i = 0; i < cnt; i++) {
		tripletList.push_back(T(i, i, 1.0));
	}
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	MathOperatorMatrix1D op(grid, mat);
	return op;
}

}

#endif // !__MATH_OPERATOR_H__
