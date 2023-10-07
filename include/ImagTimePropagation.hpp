#ifndef __IMAG_TIME_PROPAGATION_H__
#define __IMAG_TIME_PROPAGATION_H__

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "Wavefunction.hpp"
#include "MathOperator.hpp"
#include "TDSESolverFD.hpp"
#include "TDSESolverFFT.hpp"
#include "TDSESolverFDInSH.hpp"
#include "TDSESolverSH.hpp"
#include "Tools.hpp"
#include "ExportDataToFile.hpp"
#include "Util.hpp"

namespace CQP{

constexpr double DEFAULT_MAX_T_SPAN = 10000.0;
constexpr double MINIMAL_RESIDUAL_ERROR = 1e-4;

//ITP MODE:
constexpr int ITP_SCHMIDT = 1;
constexpr int ITP_ONE_BY_ONE = 2;

//ENERGE MODE:
constexpr int POS_ENERGE_MODE = 1;
constexpr int NEG_ENERGE_MODE = 2;

template<unsigned N>
using InitWaveGenerator = std::function<Wavefunction<N>(const NumericalGrid<N>&, int)>;

template<unsigned N, unsigned M = 1, typename _Solver_T = TDSESolverFD<N>, unsigned _ITP_MODE = ITP_SCHMIDT>
class ImagTimePropagationSolver
{
public:
	static_assert((_Solver_T::_N_number == N), "the dimension of Solver must be equal to N.");
	static_assert(std::is_base_of_v<TDSESolver<N>, _Solver_T>, "The type of Solver must be derived from TDSESolver<N>.");

	ImagTimePropagationSolver(const Wavefunction<N>& po_func, InitWaveGenerator<N> initalWavePickupStrategy, double dt, double m_err);

	std::complex<double> checkResidual(int order);

	void execute();

	Wavefunction<N>& getResult(int order);

	double getEnergy(int order);

private:
	std::vector< _Solver_T > solvers;
	MathOperatorMatrix<N> H;
	std::vector< Wavefunction<N> > eigen_waves;
	double minimal_error_of_residual;
};



template<unsigned N, unsigned M, typename _Solver_T, unsigned _ITP_MODE>
ImagTimePropagationSolver<N, M, _Solver_T, _ITP_MODE>::ImagTimePropagationSolver
	(const Wavefunction<N>& po_func, InitWaveGenerator<N> initalWavePickupStrategy, double dt, double m_err)
	: minimal_error_of_residual(m_err), eigen_waves(M)
{
//#ifdef _USING_OPENMP_IN_ITP
//	#pragma omp parallel for num_threads(M) schedule(static)
//#endif
	for (int i = 0; i < M; i++) {
		//MathFuncType<N> ifunc = makeGaussPkgND<N>(std::vector<double>(N, 1), std::vector<double>(N, 0.1 * i), std::vector<double>(N, 0.1 * i));
		//Wavefunction<N> iwave = createWaveByExpression(po_func.getGrid(), ifunc);
		//Wavefunction<N> iwave = createRandomWave(po_func.getGrid(), i);
		eigen_waves[i] = initalWavePickupStrategy(po_func.getGrid(), i);
	}

	for (int i = 0; i < M; i++) {
		solvers.emplace_back(po_func, eigen_waves[i], dt, DEFAULT_MAX_T_SPAN, IMAG_TIME_PROPAGATION_COND);
	}

	H = solvers.front().getHamiltonian();
}


template<unsigned N, unsigned M, typename _Solver_T, unsigned _ITP_MODE>
inline std::complex<double> ImagTimePropagationSolver<N, M, _Solver_T, _ITP_MODE>::checkResidual(int order)
{
	auto tmp = H * solvers[order].getCurrentState();
	//std::cout << "energy distribution : " << tmp.getSamplesHandler() << std::endl;
	auto en = tmp.innerProduct(solvers[order].getCurrentState());
	auto residual_state = H * solvers[order].getCurrentState() - solvers[order].getCurrentState() * en;
	return residual_state.norm();
}


template<unsigned N, unsigned M, typename _Solver_T, unsigned _ITP_MODE>
inline void ImagTimePropagationSolver<N, M, _Solver_T, _ITP_MODE>::execute()
{
	if constexpr (_ITP_MODE == ITP_ONE_BY_ONE)
	{
		for (int i = 0; i < M; i++) {
			double max_err = 0.0;
			double last_err = 10000.0;
			double delta_err = 0.0;
			do {
				for (int j = 0; j < i; j++) {
					//project out the eigenstates solved before.
					eigen_waves[i] = eigen_waves[i] - eigen_waves[j] * eigen_waves[i].innerProduct(eigen_waves[j]);
				}
				solvers[i].execute(1);
				eigen_waves[i].normalize();
				max_err = checkResidual(i).real();
				delta_err = last_err - max_err;
				last_err = max_err;
				//std::cout << delta_err << std::endl;
			} while (abs(delta_err) > minimal_error_of_residual);
			//std::cout << "finished " << i << std::endl;
			eigen_waves[i].normalize();
		}
	}
	else if constexpr (_ITP_MODE == ITP_SCHMIDT)
	{
		double maxerr = 0.0;
		double last_err = 1000000.0;
		double delta_err = 0.0;
		do {
			for (int i = 0; i < M; i++) {
				solvers[i].execute(1);
				eigen_waves[i].normalize();
			}
			schmidtOrthon(eigen_waves);
			
			maxerr = 0; //set max
			for (int i = 0; i < M; i++) {
				maxerr = std::max(maxerr, checkResidual(i).real());
			}
			delta_err = last_err - maxerr;
			last_err = maxerr;

			for (int i = 0; i < M; i++) {
				std::cout << checkResidual(i).real() << " ";
			}
			std::cout << delta_err << std::endl;
			//getchar();

		} while (abs(delta_err) > minimal_error_of_residual);
	}
}


template<unsigned N, unsigned M, typename _Solver_T, unsigned _ITP_MODE>
inline Wavefunction<N>& ImagTimePropagationSolver<N, M, _Solver_T, _ITP_MODE>::getResult(int order)
{
	return eigen_waves[order];
}


template<unsigned N, unsigned M, typename _Solver_T, unsigned _ITP_MODE>
inline double ImagTimePropagationSolver<N, M, _Solver_T, _ITP_MODE>::getEnergy(int order) {
	return (H * eigen_waves[order]).innerProduct(eigen_waves[order]).real();
}




typedef std::function< WavefunctionSH(const NumericalGrid1D&, int, int) > InitWaveGeneratorSH;

template<unsigned M>
class ImagTimePropagationSolverSH
{
public:
	ImagTimePropagationSolverSH(const Wavefunction<3>&, InitWaveGeneratorSH, double, int, double);

	std::complex<double> checkResidualSum(int order);

	void execute();

	double getEnergy(int order);

	void checkDistribution(int order);

	void debug(int order) {
		exportWaveToFile(eigen_waves[0][0], "rfunc_0", XY, 1, 0);
		exportWaveToFile(eigen_waves[1][0], "rfunc_1", XY, 1, 0);
	}

	WavefunctionSH& getResult(int order) { return eigen_waves[order]; }

private:
	Wavefunction1D init(const NumericalGrid1D& grid, int order, int l) {
		double t = (1.0 + sqrt((double)order + 1.0));
		//std::cout << order << " " << l << " " << t << std::endl;
		return createWaveByExpression(grid, [t, l](double r) { return r * std::exp(-r) * 0.1 / (std::pow((double)l + 1.0, 2) / t); });
	}

	Wavefunction1D r_potiential;
	NumericalGrid3D sh_grid;
	std::vector< WavefunctionSH > eigen_waves;
	std::vector< TDSESolverSH > solvers;

	double delta_t;
	int maxl;
	
	double m_err;
};


template<unsigned M>
ImagTimePropagationSolverSH<M>::ImagTimePropagationSolverSH
	(const Wavefunction<3>& potiential_func_, InitWaveGeneratorSH initalWavePickupStrategy, double dt, int maxl, double m_err)
	: r_potiential(potiential_func_.getSlice<1>({ std::make_pair(THETA_DEGREE, 1), std::make_pair(PHI_DEGREE, 1) })),
	sh_grid(potiential_func_.getGrid()),
	eigen_waves(M),
	delta_t(dt), maxl(maxl), m_err(m_err)
{
	for(int i = 0; i < M; i++){
		//eigen_waves[i] = WavefunctionSH(r_potiential.getGrid(), maxl);
		//eigen_waves[i][0] = init(r_potiential.getGrid(), i, 0);
		eigen_waves[i] = initalWavePickupStrategy(r_potiential.getGrid(), maxl, i);
	}

	for(int i = 0; i < M; i++){
		solvers.emplace_back(r_potiential, eigen_waves[i], dt, DEFAULT_MAX_T_SPAN, IMAG_TIME_PROPAGATION_COND);
	}
}


template<unsigned M>
std::complex<double> ImagTimePropagationSolverSH<M>::checkResidualSum(int order)
{
	std::complex<double> res = 0.0;
	for(int i = 0; i < eigen_waves[order].size(); i++){
		auto tmp = solvers[order].getHamiltonianInRfunc(i) * eigen_waves[order][i];
		auto en = tmp.innerProduct(eigen_waves[order][i]);
		auto residual_state = tmp - eigen_waves[order][i] * en;
		res += residual_state.norm();
	}
	return res;
}

template<unsigned M>
double ImagTimePropagationSolverSH<M>::getEnergy(int order)
{
	double res = 0.0;
	for(int i = 0; i < eigen_waves[order].size(); i++){
		auto state = eigen_waves[order][i];
		auto tmp = solvers[order].getHamiltonianInRfunc(i) * state;
		res += tmp.innerProduct(state).real();
	}
	return res;
}

template<unsigned M>
void ImagTimePropagationSolverSH<M>::execute()
{
	double maxerr = 0.0;
	double last_err = 1000000.0;
	double delta_err = 0.0;

	do {
		for (int i = 0; i < M; i++) {
			solvers[i].execute(1);
			eigen_waves[i].normalize();
		}
		schmidtOrthon(eigen_waves);
		
		maxerr = 0; //set max
		for (int i = 0; i < M; i++) {
			maxerr = std::max(maxerr, checkResidualSum(i).real());
		}
		delta_err = last_err - maxerr;
		last_err = maxerr;
		
		for (int i = 0; i < M; i++) {
			std::cout << checkResidualSum(i).real() << " ";
		}
		std::cout << delta_err << std::endl;
		//getchar();

	} while (delta_err > m_err || delta_err < 0);
}

template <unsigned M>
inline void ImagTimePropagationSolverSH<M>::checkDistribution(int order)
{
	std::cout << "The distribution of " << order << " --> " << std::endl; 
	for(int i = 0; i < eigen_waves[order].size(); i++){
		std::cout << eigen_waves[order][i].norm().real() << std::endl;
	}
	std::cout << std::endl;
}

}

/*

template<unsigned M>
class ImagTimePropagationSolverSH
{
public:

	ImagTimePropagationSolverSH(const Wavefunction3D& potiential_func_, double dt, int lmax)
		: potiential_func(potiential_func_.getSlice<1>({ std::make_pair(THETA_DEGREE, 1), std::make_pair(PHI_DEGREE, 1) })),
		sh_grid(potiential_func_.getGrid()),
		bundle_of_result_eigen_state(M, Wavefunction3D(potiential_func_.getGrid())),
		bundle_of_tdse_solvers(M),
		bundle_of_r_states_p(M),
		r_rev_addition(create1DWaveByExpression(potiential_func.getGrid(), [](double r) { return 1.0 / r; })),
		r_addition(create1DWaveByExpression(potiential_func.getGrid(), [](double r) { return r; })),
		delta_t(dt), lmax(lmax)
	{
		int k = 0;
		for (int i = 0; i < M; i++) {
			for (int l = 0; l < lmax; l++) {
				for (int m = -l; m <= l; m++) {
					bundle_of_tdse_solvers[i].push_back(createSolverFDForImagTimeSH(potiential_func, init(potiential_func.getGrid(), i, l), l));
				}
			}
			k = 0;
			for (int l = 0; l < lmax; l++) {
				for (int m = -l; m <= l; m++) {
					bundle_of_r_states_p[i].push_back(&(bundle_of_tdse_solvers[i][k].getCurrentState()));
					k++;
				}
			}
		}
	}

	void normalize(int order) {
		for (auto rsp : bundle_of_r_states_p[order]) {
			(*rsp) = ((*rsp)) * (1.0 / get_norm_sum(order));
		}
	}

	void check_orthon() {
		int r_size = bundle_of_r_states_p[0].size();
		for (int i = 0; i < M; i++) {
			for (int j = 0; j < M; j++) {
				std::complex<double> tmp = 0;
				for (int k = 0; k < r_size; k++) {
					tmp += (*bundle_of_r_states_p[i][k]).innerProduct(*bundle_of_r_states_p[j][k]);
				}
				std::cout << i << ", " << j << " inner product : " << tmp << std::endl;
			}
		}
	}

	void check_distribution(int order) {
		int r_size = bundle_of_r_states_p[order].size();
		std::complex<double> tmp = 0;
		for (int k = 0; k < r_size; k++) {
			tmp = (*bundle_of_r_states_p[order][k]).norm();
			std::cout << order << " of norm : " << tmp << std::endl;
		}
		std::cout << std::endl;
	}

	void execute() {
		double maxerr = 0;
		for (int i = 0; i < M; i++) {
			normalize(i);
		}
		int n = 0;
		int start_index = 0;
		do {
			for (int i = start_index; i < M; i++) {
				for (auto& s : bundle_of_tdse_solvers[i]) {
					s.execute(1);
				}
				normalize(i);
			}
			schmidtOrthonForCouplingWavesInSH(bundle_of_r_states_p);
			for (int i = start_index; i < M; i++) {
				normalize(i);
			}
			
			if (0) {
				std::cout << "n = " << n << std::endl;
				for (int i = 0; i < M; i++) {
					std::cout << "energy of " << i << " = " << getEnergey(i) << std::endl;
					//std::cout << "distrution of " << i << " : " << std::endl;
					//std::cout << "total_norm : " << get_norm_sum(i) << std::endl;
					//check_distribution(i);
					std::cout << "err of " << i << " = " << checkResidualSum(i).real() << std::endl;
				}
				std::cout << std::endl;
				getchar();
				//check_orthon();
			}

			maxerr = 0;
			for (int i = start_index; i < M; i++) {
				double tmp = checkResidualSum(i).real();
				maxerr = std::max(maxerr, tmp);
				if (i == start_index && tmp < 3e-3) {
					start_index += 1;
				}
				//std::cout << "err of " << i << " = " << checkResidualSum(i).real() << " ";
			}
			//std::cout << std::endl << "maxerr: " << maxerr << std::endl;
			//getchar();
			//check_orthon();
			n++;
		} while (maxerr > 3e-3 && n <= 100000);
	}

	std::complex<double> get_norm_sum(int order) {
		std::complex<double> result = 0;
		for (auto rsp : bundle_of_r_states_p[order]) {
			result += (*rsp).norm();
		}
		return std::sqrt(result);
	}

	std::complex<double> checkResidualSum(int order) {
		std::complex<double> res = 0.0;
		for (auto& s : bundle_of_tdse_solvers[order]) {
			auto crt_s = s.getCurrentState();
			auto tmp = s.getHamiltonian() * crt_s;
			auto en = tmp.innerProduct(crt_s);
			auto residual_state = s.getHamiltonian() * crt_s - crt_s * en;
			res += residual_state.norm();
		}
		return res;
	}

	Wavefunction3D& getResult(int order) {
		bundle_of_result_eigen_state[order] = convergeSHBasesToWave(sh_grid, bundle_of_r_states_p[order], lmax);
		return bundle_of_result_eigen_state[order];
	}

	double getEnergey(int order) {
		double result = 0.0;
		for (auto& s : bundle_of_tdse_solvers[order]) {
			auto crt_s = s.getCurrentState();
			auto tmp = (s.getHamiltonian() * crt_s).innerProduct(crt_s).real();
			result += tmp;
		}
		return result;
	}

private:

	Wavefunction1D init(const NumericalGrid1D& grid, int order, int l) {
		double t = (1.0 + sqrt((double)order + 1.0));
		//std::cout << order << " " << l << " " << t << std::endl;
		return create1DWaveByExpression(grid, [t, l](double r) { return r * std::exp(-r) * 0.1 / (std::pow((double)l + 1.0, 2) / t); });
	}

	Wavefunction1D potiential_func;
	NumericalGrid3D sh_grid;

	std::vector< CouplingSolvers > bundle_of_tdse_solvers;
	std::vector< CouplingWavesRef > bundle_of_r_states_p;

	Wavefunction1D r_rev_addition;
	Wavefunction1D r_addition;

	std::vector<Wavefunction3D> bundle_of_result_eigen_state;
	double delta_t;
	int lmax;
};

}
*/
/*

template<>
class ImagTimePropagationSolverSH<1>
{
public:
	ImagTimePropagationSolverSH(const Wavefunction3D& potiential_func_, double dt, int lmax)
		: potiential_func(potiential_func_.getSlice<1>({ std::make_pair(THETA_DEGREE, 1), std::make_pair(PHI_DEGREE, 1) })),
		sh_grid(potiential_func_.getGrid()),
		result_eigen_state(potiential_func_.getGrid()),
		r_rev_addition(create1DWaveByExpression(potiential_func.getGrid(), [](double r) { return 1.0 / r; })),
		r_addition(create1DWaveByExpression(potiential_func.getGrid(), [](double r) { return r; })),
		delta_t(dt), lmax(lmax)
	{
		tdse_solvers.reserve(lmax * lmax);
		for (int l = 0; l < lmax; l++) {
			for (int m = -l; m <= l; m++) {
				tdse_solvers.push_back(createSolverFDForImagTimeSH(potiential_func, init(potiential_func.getGrid(), l), l));
				r_states_p.push_back(&tdse_solvers.back().getCurrentState());
			}
		}
		//r_rev_addition.normalize();
	}
	
	void normalize() {
		for (auto rsp : r_states_p) {
			(*rsp) = ((*rsp)) * (1.0 / get_norm_sum());
		}
	}

	void execute(){
		double err = 0;
		normalize();
		do {
			for (auto& s : tdse_solvers) {
				s.execute(1);
			}
			normalize();
			err = checkResidualSum().real();
			std::cout << err << std::endl;
		} while (err > 1e-5);
	}

	std::complex<double> get_norm_sum() {
		std::complex<double> result = 0;
		for (auto rsp : r_states_p) {
			result += (*rsp).norm();
		}
		return std::sqrt(result);
	}

	std::complex<double> checkResidualSum() {
		std::complex<double> res = 0.0;
		for (auto& s : tdse_solvers) {
			auto crt_s = s.getCurrentState();
			auto tmp = s.getHamiltonian() * crt_s;
			auto en = tmp.innerProduct(crt_s);
			auto residual_state = s.getHamiltonian() * crt_s - crt_s * en;
			res += residual_state.norm();
		}
		return res;
	}
 
	Wavefunction3D& getResult() { 
		result_eigen_state = convergeSHBasesToWave(sh_grid, r_states_p, lmax);
		return result_eigen_state;
	}

	double getEnergey() {
		double result = 0.0;
		for (auto& s : tdse_solvers) {
			auto crt_s = s.getCurrentState() * r_rev_addition;
			result += (s.getHamiltonian() * s.getCurrentState()).innerProduct(s.getCurrentState()).real();
		}
		return result;
	}

private:

	Wavefunction1D init(const NumericalGrid1D& grid, int l) {
		return create1DWaveByExpression(grid, [](double r) { return r * std::exp(-r); });
	}

	Wavefunction1D potiential_func;
	NumericalGrid3D sh_grid;
	std::vector< TDSESolverFDInSH > tdse_solvers;
	std::vector< Wavefunction1D* > r_states_p;
	Wavefunction1D r_rev_addition;
	Wavefunction1D r_addition;

	Wavefunction3D result_eigen_state;
	double delta_t;
	int lmax;
};

*/
#endif