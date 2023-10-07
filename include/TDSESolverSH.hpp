#ifndef __TDSE_SOLVER_SH_H__
#define __TDSE_SOLVER_SH_H__

#include "WavefunctionSH.hpp"
#include "TDSESolver.hpp"

namespace CQP {

class TDSESolverSH
{
public:

    TDSESolverSH(const Wavefunction1D& po_func_r, WavefunctionSH& is, double dt, double ts, int cond);

    bool execute(int);

    const MathOperatorMatrix1D & getHamiltonianInRfunc(int index) const 
        { return r_solvers[index].getHamiltonian(); }

	int getCurrentStep() { return crt_step; }

	WavefunctionSH& getCurrentState() { return crt_state; }

protected:
    Wavefunction1D potiential_func_r;
    WavefunctionSH initial_state;
    WavefunctionSH& crt_state;
    std::vector< TDSESolverFDInSH > r_solvers;

    double delta_t;
    double time_span;
	int total_steps;
	int crt_step;
	int boundary_cond;
};


TDSESolverSH::TDSESolverSH(const Wavefunction1D& po_func_r, WavefunctionSH& is, double dt, double ts, int cond)
    : potiential_func_r(po_func_r), initial_state(is), crt_state(is),
    delta_t(dt), time_span(ts), boundary_cond(cond),
    crt_step(0), total_steps(static_cast<int>(std::floor(ts / dt)))
{
    assert(po_func_r.getGrid() == is.getRGrid());
	assert(dt >= 0.0 && ts >= 0.0);
	assert(cond == REFLECTING_BOUNDARY_COND 
		|| cond == PERIODIC_BOUNDARY_COND 
		|| cond == IMAG_TIME_PROPAGATION_COND
		|| cond == IMAG_TIME_PROPAGATION_MINUS_COND);

    for (int i = 0; i < is.size(); i++) {
        int l = WavefunctionSH::getNumberL(i);
        r_solvers.emplace_back(po_func_r, crt_state[i], dt, ts, cond, l);
    }
}


bool TDSESolverSH::execute(int ex_times)
{
    assert(ex_times == -1 || (ex_times >= 0 && ex_times <= total_steps));

	if (total_steps == crt_step) {
		return false;
	}
	if (ex_times == -1 || ex_times > (total_steps - crt_step)) {
		crt_step += total_steps - crt_step;
	} else {
        crt_step += ex_times;
    }

	for (auto &s: r_solvers) {
        s.execute(ex_times);
    }

	return true;
}


}



#endif //__TDSE_SOLVER_SH_H__