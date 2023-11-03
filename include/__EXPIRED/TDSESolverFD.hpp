#ifndef __TDSE_SOLVER_FD_H__
#define __TDSE_SOLVER_FD_H__

#include "TDSESolver.hpp"
#include "MathOperator.hpp"
#include "Tools.hpp"

namespace CQP{

template<unsigned N>
class TDSESolverFD : public TDSESolver<N>
{
public:
	using TDSESolver<N>::total_steps;
	using TDSESolver<N>::crt_step;
	using TDSESolver<N>::crt_state;
	using TDSESolver<N>::delta_t;
	using TDSESolver<N>::boundary_cond;
	using TDSESolver<N>::initial_state;
	using TDSESolver<N>::potiential_func;
	
	TDSESolverFD(const Wavefunction<N>& po_func, Wavefunction<N>& is, double dt, double ts, int cond);

	virtual bool execute(int ex_times) override;

	virtual void update_hamiltonian();

	virtual const MathOperatorMatrix<N>& getHamiltonian() const override 
		{ return infact_hamiltonian; }

protected:
	MathOperatorMatrix<N> infact_hamiltonian;
	MathOperatorMatrix<N> A_positive;
	MathOperatorMatrix<N> A_negative;
};


inline void eliminationProcess(SpMat& A, Eigen::VectorXcd& X, Eigen::VectorXcd& B, int cnt)
{
	assert(A.cols() == X.rows() && A.cols() == B.rows());

	for (int m = 1; m < cnt; m++) {
		for (int l = m; l >= m - 1; l--) {
			A.coeffRef(m, l) -= (A.coeffRef(m, m - 1) / A.coeffRef(m - 1, m - 1)) * A.coeffRef(m - 1, l);
			B(m) -= A.coeffRef(m, m - 1) / A.coeffRef(m - 1, m - 1) * B(m - 1);
		}
	}
	X(cnt - 1) = B(cnt - 1) / A.coeffRef(cnt - 1, cnt - 1);
	for (int m = cnt - 2; m >= 0; m--) {
		X(m) = (B(m) - A.coeffRef(m, m + 1) * X(m + 1)) / A.coeffRef(m, m);
	}
}

inline void solveLinearEquationsProblemInFD1D(const MathOperatorMatrix<1>& matrix, Wavefunction<1>& X, const Wavefunction<1>& B, int cond)
{
	assert(matrix.getGrid() == B.getGrid() && X.getGrid() == B.getGrid());

	int cnt = matrix.getGrid().getCount();

	if (cond == REFLECTING_BOUNDARY_COND 
		|| cond == IMAG_TIME_PROPAGATION_COND
		|| cond == IMAG_TIME_PROPAGATION_MINUS_COND)
	{
		SpMat mat(matrix.getMatrix());
		Eigen::VectorXcd bdata(B.getSamplesView());
		Eigen::VectorXcd xdata(cnt);
		eliminationProcess(mat, X.getSamplesHandler(), bdata, cnt);
	}
	else if (cond == PERIODIC_BOUNDARY_COND)
	{
		SpMat mat1(matrix.getMatrix());
		auto corner_num = mat1.coeffRef(0, cnt - 1);

		Eigen::VectorXcd udata = Eigen::VectorXcd::Zero(cnt);
		Eigen::VectorXcd vdata = Eigen::VectorXcd::Zero(cnt);
		udata(0) = 1;
		udata(cnt - 1) = 1;
		vdata(0) = corner_num;
		vdata(cnt - 1) = corner_num;

		mat1.coeffRef(0, 0) -= corner_num;
		mat1.coeffRef(0, cnt - 1) -= corner_num;
		mat1.coeffRef(cnt - 1, 0) -= corner_num;
		mat1.coeffRef(cnt - 1, cnt - 1) -= corner_num;

		SpMat mat2(mat1);

		Eigen::VectorXcd bdata1(B.getSamplesView());
		Eigen::VectorXcd xdata1 = Eigen::VectorXcd::Zero(cnt);
		Eigen::VectorXcd bdata2(udata);
		Eigen::VectorXcd xdata2 = Eigen::VectorXcd::Zero(cnt);

		eliminationProcess(mat1, xdata1, bdata1, cnt);
		eliminationProcess(mat2, xdata2, bdata2, cnt);

		std::complex<double> tmp1 = vdata.transpose() * xdata2;
		std::complex<double> tmp2 = vdata.transpose() * xdata1;
		auto next_func = xdata1 - xdata2 * (tmp2 / (1.0 + tmp1));

		X.getSamplesHandler() = next_func;
	}
}

inline void CrankNicolsonMethod1D(Wavefunction1D& crt_state, const MathOperatorMatrix1D& A_pos, const MathOperatorMatrix1D& A_neg, int ex_times, int bond)
{
	Wavefunction1D half_state(crt_state.getGrid());
	for (int i = 1; i <= ex_times; i++) {
		half_state = A_pos * crt_state;
		solveLinearEquationsProblemInFD1D(A_neg, crt_state, half_state, bond);
	}
}

template<>
inline void TDSESolverFD<1>::update_hamiltonian()
{
	using namespace std::literals;

	auto grid = initial_state.getGrid();
	auto delta_x = grid.getDelta(0);
	auto hamiltonian = createHamiltonianFD1D(grid, boundary_cond);
	auto I = createIdentityOperator(grid);
	auto V = createMatrixV1D(grid, potiential_func);
	infact_hamiltonian = hamiltonian + V;

	auto delta_t_infact = (boundary_cond == IMAG_TIME_PROPAGATION_COND) ? (-1i * delta_t) \
		: ((boundary_cond == IMAG_TIME_PROPAGATION_MINUS_COND) ? (1i * delta_t) : delta_t);

	auto D = hamiltonian * (-2.0);
	auto M = I + D * (delta_x * delta_x / 12);
	A_positive = M - (D * (-0.5) + M * V) * (0.5i * delta_t_infact);
	A_negative = M + (D * (-0.5) + M * V) * (0.5i * delta_t_infact);
}


template<>
inline TDSESolverFD<1>::TDSESolverFD(const Wavefunction<1>& po_func, Wavefunction<1>& is, double dt, double ts, int cond)
	: TDSESolver<1>(po_func, is, dt, ts, cond),
	A_positive(is.getGrid()), A_negative(is.getGrid())
{
	assert(is.getGrid() == po_func.getGrid());
	update_hamiltonian();
}


template<>
inline bool TDSESolverFD<1>::execute(int ex_times)
{
	assert(ex_times == -1 || (ex_times >= 0 && ex_times <= total_steps));

	if (total_steps == crt_step) {
		return false;
	}
	if (ex_times == -1 || ex_times > (total_steps - crt_step)) {
		crt_step += total_steps - crt_step;
	}
	else {
		crt_step += ex_times;
	}

	CrankNicolsonMethod1D(crt_state, A_positive, A_negative, ex_times, boundary_cond);

	return true;
}

}

#endif // !__TDSE_SOLVER_FD_H__
