#ifndef __TDSE_SOLVER_FFT_H__
#define __TDSE_SOLVER_FFT_H__

#include "TDSESolver.hpp"
#include "MathOperator.hpp"
#include "Tools.hpp"
#include <functional>

namespace CQP{

template<unsigned N>
class TDSESolverFFT : public TDSESolver<N>
{
public:
	using TDSESolver<N>::total_steps;
	using TDSESolver<N>::crt_step;
	using TDSESolver<N>::crt_state;
	using TDSESolver<N>::delta_t;
	using TDSESolver<N>::boundary_cond;
	using TDSESolver<N>::initial_state;
	using TDSESolver<N>::potiential_func;

	TDSESolverFFT(const Wavefunction<N>& po_func, Wavefunction<N>& is, double dt, double ts, int cond);

	virtual bool execute(int ex_times) override;

	virtual const MathOperatorMatrix<N>& getHamiltonian() const override
	{
		return infact_hamiltonian;
	}

private:
	template<unsigned _N>
	void update_phase_factor();
 
	MathOperatorMatrix<N> infact_hamiltonian;
	Wavefunction<N> phase_factor_1;
	Wavefunction<N> phase_factor_2;
};


template<unsigned N>
inline TDSESolverFFT<N>::TDSESolverFFT(const Wavefunction<N>& po_func, Wavefunction<N>& is, double dt, double ts, int cond)
	: TDSESolver<N>(po_func, is, dt, ts, cond),
	infact_hamiltonian(is.getGrid())
{
	using namespace std::literals;
	assert(po_func.getGrid() == is.getGrid());
	assert(boundary_cond == PERIODIC_BOUNDARY_COND 
		|| boundary_cond == IMAG_TIME_PROPAGATION_COND
		|| boundary_cond == IMAG_TIME_PROPAGATION_MINUS_COND);

	auto grid = initial_state.getGrid();
	auto hamiltonian = createHamiltonianFD(grid, boundary_cond);     
	auto V_matrix = createMatrixV(potiential_func);
	infact_hamiltonian = hamiltonian + V_matrix;
	update_phase_factor<N>();
	
	std::cout << "init over." << std::endl;
}


template<unsigned N>
template<unsigned _N>
inline void TDSESolverFFT<N>::update_phase_factor()
{
	using namespace std::literals;

	auto grid = initial_state.getGrid();
	auto initial_state_fft = fft(initial_state, std::vector<double>(N, 0));
	auto fft_grid = initial_state_fft.getGrid();
	auto delta_t_infact = (boundary_cond == IMAG_TIME_PROPAGATION_COND) ? (-1i * delta_t) \
		: ((boundary_cond == IMAG_TIME_PROPAGATION_MINUS_COND) ? (1i * delta_t) : delta_t);
	Wavefunction<N>& po_func = potiential_func;

	std::function<std::complex<double>(int)> func2;
	
	func2 = [delta_t_infact, &po_func](int i) {
		return std::exp(-1i * delta_t_infact * po_func.getValueByIndex(i));
	};

	phase_factor_2 = createWaveByExpressionWithIndex(grid, func2);
	typename MathFunc<N>::_type func1;

	if constexpr (N == 1) {
		func1 = [delta_t_infact](double k) { 
			return std::exp(-0.25i * delta_t_infact * std::pow(k, 2));
		};
	} else if constexpr (N == 2) {
		func1 = [delta_t_infact](double kx, double ky) {
			return std::exp(-0.25i * delta_t_infact * std::pow(kx, 2)) \
				 * std::exp(-0.25i * delta_t_infact * std::pow(ky, 2));
		};
	} else if constexpr (N == 3) {
		func1 = [delta_t_infact](double kx, double ky, double kz) {
			return std::exp(-0.25i * delta_t_infact * std::pow(kx, 2)) \
				* std::exp(-0.25i * delta_t_infact * std::pow(ky, 2)) \
				* std::exp(-0.25i * delta_t_infact * std::pow(kz, 2));
		};
	} else {
		static_assert(!(N >= 1 && N <= 3), "Error! the N must be 1, 2 or 3.");
	}

	phase_factor_1 = createWaveByExpression(fft_grid, func1);
}


template<unsigned N>
inline bool TDSESolverFFT<N>::execute(int ex_times)
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

	Wavefunction<N> x_space_state(crt_state);
	Wavefunction<N> k_space_state = fft(x_space_state, std::vector<double>(N, 0));

	for (int i = 1; i <= ex_times; i++) {
		k_space_state = k_space_state * phase_factor_1;
		x_space_state = ifft(k_space_state, std::vector<double>(N, 0));
		x_space_state = x_space_state * phase_factor_2;
		k_space_state = fft(x_space_state, std::vector<double>(N, 0));
		k_space_state = k_space_state * phase_factor_1;
	}
	x_space_state = ifft(k_space_state, std::vector<double>(N, 0));
	crt_state = x_space_state;

	return true;
}


/*
template<unsigned N>
inline TDSESolverResult<N> TDSESolverFFT<N>::execute(int ex_times)
{
	assert(ex_times == -1 || (ex_times >= 0 && ex_times <= total_steps));

	int state = TDSE_ONGOING;
	if (total_steps == crt_step) {
		return TDSESolverResult<N>(TDSE_REACHED_END, crt_state, 0.0);
	}
	if (ex_times == -1 || ex_times > (total_steps - crt_step)) {
		ex_times = total_steps - crt_step;
		state = TDSE_END;
	}

	std::clock_t c_start = std::clock();
	Wavefunction<N> x_space_state(crt_state);
	Wavefunction<N> k_space_state = fft(x_space_state, std::vector<double>(N, 0));

	for (int i = 1; i <= ex_times; i++) {
		k_space_state = k_space_state * phase_factor_1;
		x_space_state = ifft(k_space_state, std::vector<double>(N, 0));
		x_space_state = x_space_state * phase_factor_2;
		k_space_state = fft(x_space_state, std::vector<double>(N, 0));
		k_space_state = k_space_state * phase_factor_1;
	}
	x_space_state = ifft(k_space_state, std::vector<double>(N, 0));
	crt_state = x_space_state;

	std::clock_t c_end = std::clock();
	double duration = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;
	crt_step += ex_times;

	return TDSESolverResult<N>(state, crt_state, duration);
}
*/

}

#endif //__TDSE_SOLVER_FFT_H__