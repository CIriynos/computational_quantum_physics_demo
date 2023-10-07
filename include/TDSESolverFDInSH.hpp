#ifndef __TDSE_SOLVER_FD_IN_SH_H__
#define __TDSE_SOLVER_FD_IN_SH_H__

#include "TDSESolverFD.hpp"
#include "MathOperator.hpp"
#include "Wavefunction.hpp"
#include "Tools.hpp"

namespace CQP{

class TDSESolverFDInSH : public TDSESolverFD<1>
{
public:

	TDSESolverFDInSH(const Wavefunction<1>& po_func, Wavefunction<1>& is, double dt, double ts, int cond, int l)
		: l_number(l),
		TDSESolverFD<1>(po_func, is, dt, ts, cond) 
	{
		assert(boundary_cond == REFLECTING_BOUNDARY_COND
			|| boundary_cond == IMAG_TIME_PROPAGATION_COND);
		update_hamiltonian();
	}

	virtual void update_hamiltonian() 
	{
		using namespace std::complex_literals;

		const NumericalGrid1D& grid = initial_state.getGrid();
		auto hamiltonian = createHamiltonianFD(grid, boundary_cond);
		auto I = createIdentityOperator(grid);
		double l = (double)l_number;
		auto po_appendix = create1DWaveByExpression(grid, [l](double r) { return l * (l + 1) / (2 * r * r); });
		
		auto Vl = createMatrixV1D(grid, potiential_func + po_appendix);
		auto D = hamiltonian * (-2.0);
		auto M2 = createM2(grid);
		infact_hamiltonian = hamiltonian + Vl;
		forward_h = D + M2 * Vl;

		if (boundary_cond == REFLECTING_BOUNDARY_COND) {
			A_positive = M2 - (D + M2 * Vl) * (0.5i * delta_t);
			A_negative = M2 + (D + M2 * Vl) * (0.5i * delta_t);
		}
		else if (boundary_cond == IMAG_TIME_PROPAGATION_COND) {
			A_positive = M2 - (D + M2 * Vl) * (0.5 * delta_t);
			A_negative = M2 + (D + M2 * Vl) * (0.5 * delta_t);
		}
	}

	MathOperatorMatrix1D createM2(const NumericalGrid<1>& grid)
	{
		typedef Eigen::Triplet< std::complex<double> > T;

		int cnt = grid.getTotalCount();
		std::vector<T> tripletList;
		tripletList.reserve((size_t)cnt * 4);
		double scaler = -1.0 / 6.0;
		SpMat matrix(cnt, cnt);

		for (int i = 0; i < cnt; i++) {
			if (i - 1 >= 0) {
				tripletList.push_back(T(i, i - 1, 1 * scaler));
			}
			tripletList.push_back(T(i, i, 10 * scaler));
			if (i + 1 < cnt) {
				tripletList.push_back(T(i, i + 1, 1 * scaler));
			}
		}
		matrix.setFromTriplets(tripletList.begin(), tripletList.end());
		return MathOperatorMatrix<1>(grid, matrix);
	}

protected:
	int l_number;
	MathOperatorMatrix<1> forward_h;
};


/*
class TDSESolverSH : public TDSESolver<3>
{
public:
	TDSESolverSH(const Wavefunction3D& po_func, const Wavefunction3D& is, double dt, double ts, int l, int cond)
		: TDSESolver<3>(po_func, is, dt, ts, l),
		l_max_number(l), total_solvers_number(l* l),
		grid(initial_state.getGrid()),
		buffer(l * l),
		solvers(l * l)
	{
		//There is an ASSUMPTION that the po_func is spherically symmetric.
		auto r_po_func = potiential_func.getSlice<1>({ std::make_pair(THETA_DEGREE, 0), std::make_pair(PHI_DEGREE, 0) });
		//std::cout << r_po_func.getSamplesHandler() << std::endl;
		auto bases = expandInSHBases(is, total_solvers_number);

		for (int i = 0; i < total_solvers_number; i++) {
			solvers[i] = TDSESolverFDInSH(r_po_func, bases[i], dt, ts, l, cond);
		}
	}

	virtual TDSESolverResult<3> execute(int steps)
	{
		int state = 0;
		for (auto& s : solvers) {
			state = std::get<0>(s.execute(steps));
			//if (mode == IMAG_TIME_SH) {
			//	std::cout << s.checkResidual() << std::endl;
			//}
		}

		int i = 0;
		for (auto& s : solvers) {
			buffer[i] = s.getCurrentState();
			i++;
		}
		//crt_state = convergeSHBasesToWave(grid, buffer, l_max_number);
		std::cout << "crt_step = " << crt_step << std::endl;
		crt_step += steps;
		return TDSESolverResult<3>(state, crt_state, 0);
	}

private:
	int l_max_number;
	int total_solvers_number;
	const NumericalGrid3D& grid;

	std::vector< TDSESolverFDInSH > solvers;
	std::vector< Wavefunction1D > buffer;
};

*/
}

#endif // !__TDSE_SOLVER_FD_IN_SH_H__