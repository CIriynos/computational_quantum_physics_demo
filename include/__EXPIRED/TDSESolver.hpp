#ifndef __TDSE_SOLVER_H__
#define __TDSE_SOLVER_H__

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "Wavefunction.hpp"
#include "MathOperator.hpp"

namespace CQP{

typedef Eigen::SparseMatrix< std::complex<double> > SpMat;

constexpr int REFLECTING_BOUNDARY_COND = 1;
constexpr int PERIODIC_BOUNDARY_COND = 2;
constexpr int IMAG_TIME_PROPAGATION_COND = 3;
constexpr int IMAG_TIME_PROPAGATION_MINUS_COND = 4;


template<unsigned N>
class TDSESolver
{
public:
	constexpr static int _N_number = N;

	TDSESolver(const Wavefunction<N>& po_func, Wavefunction<N>& is, double dt, double ts, int cond)
		: potiential_func(po_func), initial_state(is),
		crt_state(is), delta_t(dt), time_span(ts), boundary_cond(cond),
		crt_step(0), total_steps(static_cast<int>(std::floor(ts / dt)))
	{
		assert(po_func.getGrid() == is.getGrid());
		assert(dt >= 0.0 && ts >= 0.0);
		assert(cond == REFLECTING_BOUNDARY_COND 
			|| cond == PERIODIC_BOUNDARY_COND 
			|| cond == IMAG_TIME_PROPAGATION_COND
			|| cond == IMAG_TIME_PROPAGATION_MINUS_COND);
	}

	virtual bool execute(int) = 0;

	virtual const MathOperatorMatrix<N> & getHamiltonian() const = 0;

	int getCurrentStep() { return crt_step; }

	Wavefunction<N>& getCurrentState() { return crt_state; }

protected:

	Wavefunction<N> potiential_func;
	Wavefunction<N> initial_state;
	Wavefunction<N>& crt_state;
	double delta_t;
	double time_span;

	//for step-by-step calculation
	int total_steps;
	int crt_step;

	int boundary_cond;
};

}

#endif //__TDSE_SOLVER_H__!
