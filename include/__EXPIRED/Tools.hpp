#ifndef __TOOLS_OF_QUANTUM_H__
#define __TOOLS_OF_QUANTUM_H__

#include "Wavefunction.hpp"
#include "WavefunctionSH.hpp"
#include "Util.hpp"
#include <cmath>
#include <cstdlib>
#include <ctime>

namespace CQP{

template<unsigned N>
inline Wavefunction<N> createRandomWave(const NumericalGrid<N>& grid, unsigned int order)
{
	std::vector< std::complex<double> > buffer(grid.getTotalCount(), 0);

#ifdef _USING_OPENMP_IN_TOOLS
	#pragma omp parallel
	{
		int tid = omp_get_thread_num();
		RandomNum r((long long)time(nullptr) + order + tid);

		#pragma omp for
		for (int i = 0; i < grid.getTotalCount(); i++) {
			buffer[i] = std::complex<double>(r.lcg(100) - 49, r.lcg(100) - 49);
		}
	}
#else
	RandomNum r((unsigned int)time(nullptr) + order);
	for (int i = 0; i < grid.getTotalCount(); i++) {
		buffer[i] = std::complex<double>(r.lcg(100) - 49, r.lcg(100) - 49);
	}
#endif

	Wavefunction<N> wave(grid, buffer);
	wave.normalize();
	return wave;
}


inline WavefunctionSH createRandomWaveSH(const NumericalGrid1D& rgrid, unsigned int maxl, unsigned int order)
{
	WavefunctionSH sh_wave(rgrid, maxl);
	for(int i = 0; i < sh_wave.size(); i++){
		int factor = (i + 1) * (order + 1);
		sh_wave[i] = createRandomWave(rgrid, factor);
	}
	return sh_wave;
}


template<unsigned N>
inline Wavefunction<N> createWaveByExpressionWithIndex(const NumericalGrid<N>& grid, std::function<std::complex<double>(int)> func)
{
	std::vector< std::complex<double> > buffer(grid.getTotalCount(), 0);
#ifdef _USING_OPENMP_IN_TOOLS
	#pragma omp parallel for
#endif
	for (int i = 0; i < grid.getTotalCount(); i++) {
		buffer[i] = func(i);
	}
	Wavefunction<N> wave(grid, buffer);
	return wave;
}


inline std::function<std::complex<double>(double)> makeGaussPkg1D(double omega_x, double x0, double p0)
{
	using namespace std::literals;
	std::complex<double> C = 1.0 / std::pow(2 * PI * omega_x * omega_x, 0.25);
	return [C, omega_x, x0, p0](double x) {
		return C * std::exp(-std::pow((x - x0) / (2 * omega_x), 2)) * std::exp(1i * p0 * x);
	};
}

inline std::function<std::complex<double>(double, double)> makeGaussPkg2D(double omega_x, double omega_y, double x0, double y0, double px, double py)
{
	using namespace std::literals;
	auto Cx = 1.0 / std::pow(2 * PI * omega_x * omega_x, 0.25);
	auto Cy = 1.0 / std::pow(2 * PI * omega_y * omega_y, 0.25);

	return [Cx, Cy, omega_x, omega_y, x0, y0, px, py](double x, double y) {
		auto xpart = Cx * std::exp(-std::pow((x - x0) / (2 * omega_x), 2)) * std::exp(1i * px * x);
		auto ypart = Cy * std::exp(-std::pow((y - y0) / (2 * omega_y), 2)) * std::exp(1i * py * y);
		return xpart * ypart;
	};
}

inline std::function<std::complex<double>(double, double, double)> makeGaussPkg3D(double omega_x, double omega_y, double omega_z, double x0, double y0, double z0, double px, double py, double pz)
{
	using namespace std::literals;
	auto Cx = 1.0 / std::pow(2 * PI * omega_x * omega_x, 0.25);
	auto Cy = 1.0 / std::pow(2 * PI * omega_y * omega_y, 0.25);
	auto Cz = 1.0 / std::pow(2 * PI * omega_z * omega_z, 0.25);

	return [Cx, Cy, Cz, omega_x, omega_y, omega_z, x0, y0, z0, px, py, pz](double x, double y, double z) {
		auto xpart = Cx * std::exp(-std::pow((x - x0) / (2 * omega_x), 2)) * std::exp(1i * px * x);
		auto ypart = Cy * std::exp(-std::pow((y - y0) / (2 * omega_y), 2)) * std::exp(1i * py * y);
		auto zpart = Cz * std::exp(-std::pow((z - z0) / (2 * omega_z), 2)) * std::exp(1i * pz * z);
		return xpart * ypart * zpart;
	};
}


template<unsigned N>
inline MathFuncType<N> makeGaussPkgND(const std::vector<double>& omegas, const std::vector<double>& x0s, const std::vector<double>& p0s)
{
	assert(omegas.size() == N && x0s.size() == N && p0s.size() == N);
	if constexpr (N == 1) {
		return makeGaussPkg1D(omegas[0], x0s[0], p0s[0]);
	} else if constexpr (N == 2) {
		return makeGaussPkg2D(omegas[0], omegas[1], x0s[0], x0s[1], p0s[0], p0s[1]);
	} else if constexpr (N == 3) {
		return makeGaussPkg3D(omegas[0], omegas[1], omegas[2], x0s[0], x0s[1], x0s[2], p0s[0], p0s[1], p0s[2]);
	} else {
		static_assert((N >= 1 && N <= 3), "Temporarily, the function does not support N which is bigger than 3.");
	}
}


inline MathOperatorMatrix<1> createHamiltonianFD1D(const NumericalGrid<1>& grid, int cond)
{
	typedef Eigen::Triplet< std::complex<double> > T;
	assert(cond == REFLECTING_BOUNDARY_COND
		|| cond == PERIODIC_BOUNDARY_COND
		|| cond == IMAG_TIME_PROPAGATION_COND);

	int cnt = grid.getTotalCount();
	double delta_x = grid.getDelta(0);
	std::vector<T> tripletList;
	tripletList.reserve(cnt * 4);
	double scaler = -1.0 / (2.0 * delta_x * delta_x);
	SpMat matrix(cnt, cnt);

	for (int i = 0; i < cnt; i++) {
		if (i - 1 >= 0) {
			tripletList.push_back(T(i, i - 1, scaler));
		}
		tripletList.push_back(T(i, i, -2.0 * scaler));
		if (i + 1 < cnt) {
			tripletList.push_back(T(i, i + 1, scaler));
		}
	}
	if (cond == PERIODIC_BOUNDARY_COND) {
		tripletList.push_back(T(0, cnt - 1, scaler * 1));
		tripletList.push_back(T(cnt - 1, 0, scaler * 1));
	}
	matrix.setFromTriplets(tripletList.begin(), tripletList.end());
	return MathOperatorMatrix<1>(grid, matrix);
}


template<unsigned N>
inline MathOperatorMatrix<N> createHamiltonianFD(const NumericalGrid<N>& grid, int cond)
{
	typedef Eigen::Triplet< std::complex<double> > T;
	assert(cond == REFLECTING_BOUNDARY_COND
		|| cond == PERIODIC_BOUNDARY_COND
		|| cond == IMAG_TIME_PROPAGATION_COND
		|| cond == IMAG_TIME_PROPAGATION_MINUS_COND);

	int cnt = grid.getTotalCount();
	double scalers[N] = { 0.0 };
	for (int i = 0; i < N; i++) {
		double delta = grid.getDelta(i);
		scalers[i] = -1.0 / (2.0 * delta * delta);
	}

	std::vector<T> tripletList;
	tripletList.reserve(cnt * (1 + 2 * N));
	SpMat matrix(cnt, cnt);

	GridIndice<N> indice;
	GridIndice<N> next_indice;

	double center_scaler = 0.0;
	for (int j = 0; j < N; j++) {
		center_scaler += scalers[j];
	}

	for (int i = 0; i < cnt; i++) {
		indice = grid.expand(i);
		for (int j = 0; j < N; j++) {
			for (int k = -1; k <= 1; k += 2) {
				next_indice = indice;
				next_indice[j] += k;
				if (next_indice[j] >= 0 && next_indice[j] < grid.getCount(j)) {
					int next_index = grid.shrink(next_indice);
					tripletList.push_back(T(i, next_index, scalers[j]));
				}
				else if (cond == PERIODIC_BOUNDARY_COND) {
					next_indice[j] = (next_indice[j] + grid.getCount(j)) % grid.getCount(j);
					int next_index = grid.shrink(next_indice);
					tripletList.push_back(T(i, next_index, scalers[j]));
				}
			}
		}
		tripletList.push_back(T(i, i, -2.0 * center_scaler));
	}
	matrix.setFromTriplets(tripletList.begin(), tripletList.end());
	return MathOperatorMatrix<N>(grid, matrix);
}

/*

template<unsigned N>
inline MathOperatorMatrix<N> createHamiltonianFD(const NumericalGrid<N>& grid, int cond)
{
	assert(cond == REFLECTING_BOUNDARY_COND
		|| cond == PERIODIC_BOUNDARY_COND
		|| cond == IMAG_TIME_PROPAGATION_COND
		|| cond == IMAG_TIME_PROPAGATION_MINUS_COND);

	int cnt = grid.getTotalCount();
	SpMat matrix(cnt, cnt);
	std::vector<int> indice;
	std::vector<int> next_indice;
	matrix.reserve(Eigen::VectorXi::Constant(cnt, 1 + 2 * N));

	double scalers[N] = { 0.0 };
	for (int i = 0; i < N; i++) {
		double delta = grid.getDelta(i);
		scalers[i] = -1.0 / (2.0 * delta * delta);
	}

	double center_scaler = 0.0;
	for (int j = 0; j < N; j++) {
		center_scaler += scalers[j];
	}

	for (int i = 0; i < cnt; i++) {
		indice = grid.expand(i);
		for (int j = 0; j < N; j++) {
			for (int k = -1; k <= 1; k += 2) {
				next_indice = indice;
				next_indice[j] += k;
				if (next_indice[j] >= 0 && next_indice[j] < grid.getCount(j)) {
					int next_index = grid.shrink(next_indice);
					matrix.insert(i, next_index, scalers[j]);
				}
				else if (cond == PERIODIC_BOUNDARY_COND) {
					next_indice[j] = (next_indice[j] + grid.getCount(j)) % grid.getCount(j);
					int next_index = grid.shrink(next_indice);
					//tripletList.push_back(T(i, next_index, scalers[j]));
					matrix.insert(i, next_index, scalers[j]);
				}
			}
		}
		//tripletList.push_back(T(i, i, -2.0 * center_scaler));
		matrix.insert(i, i, -2.0 * center_scaler);
	}

	return MathOperatorMatrix<N>(grid, matrix);
}
*/

inline MathOperatorMatrix<1> createMatrixV1D(const NumericalGrid<1>& grid, const Wavefunction<1>& potiential_func)
{
	typedef Eigen::Triplet< std::complex<double> > T;
	int cnt = grid.getCount(0);
	SpMat mat(cnt, cnt);
	std::vector<T> tripletList;

	for (int i = 0; i < cnt; i++) {
		tripletList.push_back(T(i, i, potiential_func.getValueByIndex(i)));
	}
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	MathOperatorMatrix1D op(grid, mat);
	return op;
}


template<unsigned N>
inline MathOperatorMatrix<N> createMatrixV(const Wavefunction<N>& potiential_func)
{
	typedef Eigen::Triplet< std::complex<double> > T;
	const NumericalGrid<N>& grid = potiential_func.getGrid();
	int cnt = grid.getTotalCount();
	SpMat mat(cnt, cnt);
	std::vector<T> tripletList(cnt);

#ifdef _USING_OPENMP_IN_TOOLS
	#pragma omp parallel for
#endif
	for (int i = 0; i < cnt; i++) {
		//tripletList.push_back(T(i, i, potiential_func.getValueByIndex(i)));
		tripletList[i] = T(i, i, potiential_func.getValueByIndex(i));
	}
	mat.setFromTriplets(tripletList.begin(), tripletList.end());
	MathOperatorMatrix<N> op(grid, mat);
	return op;
}





typedef NumericalGrid<3> SphericalGrid;
typedef Wavefunction<3> SphericalWave;

inline SphericalGrid createSphericalGrid(double rlength, int rN, int thetaN, int phiN)
{
	double offset_in_singular_point = (double)rlength / (double)rN;
	return NumericalGrid<3>(rN, rlength, rlength / 2 + offset_in_singular_point, thetaN, PI, PI / 2, phiN, 2 * PI, PI);
}

inline SphericalWave createSCWaveByExpr(const SphericalGrid& grid, MathFuncType<3> func)
{
	std::vector< std::complex<double> > buffer(grid.getTotalCount(), 0);

	for (int i = 0; i < grid.getTotalCount(); i++) {
		auto r = grid.index(i).x();
		auto theta = grid.index(i).y();
		auto phi = grid.index(i).z();
		buffer[i] = func(r, theta, phi);
	}
	SphericalWave wave(grid, buffer);
	return wave;
}

inline SphericalWave createSCWaveByCartesianExpr(const SphericalGrid& grid, MathFuncType<3> func)
{
	std::vector< std::complex<double> > buffer(grid.getTotalCount(), 0);

	for (int i = 0; i < grid.getTotalCount(); i++) {
		auto r = grid.index(i).x();
		auto theta = grid.index(i).y();
		auto phi = grid.index(i).z();
		auto x = r * std::sin(theta) * std::cos(phi);
		auto y = r * std::sin(theta) * std::sin(phi);
		auto z = r * std::cos(theta);
		buffer[i] = func(x, y, z);
	}
	SphericalWave wave(grid, buffer);
	return wave;
}

inline Wavefunction2D makeSHbase(const NumericalGrid2D& grid, int l, int m)
{
	using namespace std::literals;
	std::vector< std::complex<double> > buffer(grid.getTotalCount(), 0);

	auto C = std::pow(-1, m) * sqrt(((2 * (double)l + 1) / (4 * PI)) * (std::tgamma(l - m + 1) / std::tgamma(l + m + 1)));
	//(r) theta phi
	for (int i = 0; i < grid.getTotalCount(); i++) {
		auto theta = grid.index(i).x();
		auto phi = grid.index(i).y();
		buffer[i] = C * std::pow(-1, m) * (double)std::assoc_legendrel(l, abs(m), std::cos(theta)) * std::exp(1i * (double)m * phi);
	}
	Wavefunction2D wave(grid, buffer);
	wave.normalize();
	return wave;
}

inline WavefunctionSH expandToShwave(const Wavefunction3D& wave, int maxl)
{
	auto grid = wave.getGrid();
	auto r_grid = grid.subset<1>({ R_DEGREE });	
	WavefunctionSH shwave(r_grid, maxl);

	int k = 0;
	for (int l = 0; l <= maxl; l++) {
		for (int m = -l; m <= l; m++) {
			auto sh_base = makeSHbase(grid.subset<2>({THETA_DEGREE, PHI_DEGREE}), l, m);
			//sh_base.normalize();
			for (int r_index = 0; r_index < grid.getCount(0); r_index++) {
				auto phi_r = wave.getSlice<2>({std::make_pair(R_DEGREE, r_index)});
				auto r_value = r_grid.index(r_index).x();
				shwave[k].getSamplesHandler()(r_index) = phi_r.innerProduct(sh_base) * r_value;
			}
			k++;
		}
	}
	return shwave;
}

inline SphericalWave convergeShwaveToSCWave(const NumericalGrid<3>& sc_grid, WavefunctionSH& shwave)
{
	SphericalWave result(sc_grid);
	int Nr = sc_grid.getCount(R_DEGREE);
	double Lr = sc_grid.getLength(R_DEGREE);
	double Offr = sc_grid.getOffset(R_DEGREE);
	auto r_rev_f = [](double r, double theta, double phi) { return 1.0 / r; };
	auto r_rev_wave = createSCWaveByExpr(sc_grid, r_rev_f);

	for(int i = 0; i < shwave.size(); i++){
		int l = shwave.getNumberL(i);
		int m = shwave.getNumberM(i);
		auto sh_base = makeSHbase(sc_grid.subset<2>({THETA_DEGREE, PHI_DEGREE}), l, m);
		//sh_base.normalize();
		auto ex_sh_base = sh_base.upliftOnce(R_DEGREE, Nr, Lr, Offr);
		result = result + ex_sh_base * shwave[i].uplift(sc_grid) * r_rev_wave;
	}
	return result;
}

/*
inline Wavefunction3D convergeSHBasesToWave(const NumericalGrid<3>& sh_grid, const std::vector< Wavefunction1D >& bases, int maxNl)
{
	assert(bases.size() == (maxNl * maxNl));
	Wavefunction3D result(sh_grid);
	int Nr = sh_grid.getCount(R_DEGREE);
	double Lr = sh_grid.getLength(R_DEGREE);
	double Offr = sh_grid.getOffset(R_DEGREE);
	auto r_rev_f = [](double r, double theta, double phi) { return 1.0 / r; };
	auto r_rev_wave = create3DWaveByExpression(sh_grid, r_rev_f);

	int k = 0;
	for (int l = 0; l < maxNl; l++) {
		for (int m = -l; m <= l; m++) {
			auto sh_base = makeSHWave(sh_grid, l, m).getSlice<2>({ std::make_pair(R_DEGREE, 1) });
			sh_base.normalize();
			auto ex_sh_base = sh_base.upliftOnce(R_DEGREE, Nr, Lr, Offr);
			result = result + ex_sh_base * bases[k].uplift(sh_grid) * r_rev_wave;
			//cout << "l = " << l << " , m = " << m << " : " << bases[k].norm() << endl;
			k++;
		}
	}
	return result;
}
*/

template<typename _Wave_T>
inline void schmidtOrthon(std::vector< _Wave_T >& waves)
{
	for (int i = 0; i < waves.size(); i++) {
		auto tmp = waves[i];
		for (int j = 0; j < i; j++) {
			auto scaler = waves[i].innerProduct(waves[j]) / waves[j].innerProduct(waves[j]);
			tmp = tmp - waves[j] * scaler;
		}
		waves[i] = tmp;
	}
}

/*
inline void schmidtOrthon(std::vector< WavefunctionSH >& coupling_waves)
{
	auto r_grid = coupling_waves[0].getRfunc(0, 0).getGrid();
	//inner product for scaler -> yield a wavefunction! not a complex number.
	Wavefunction1D scaler1(r_grid);
	Wavefunction1D scaler2(r_grid);

	for (int i = 0; i < coupling_waves.size(); i++) {
		WavefunctionSH next_wave = coupling_waves[i];
		for (int j = 0; j < i; j++) {
			for (int k = 0; k < next_wave.size(); k++) {
				scaler1 = scaler1 + (coupling_waves[i][k]) * (coupling_waves[j][k]).conjugate();
				scaler2 = scaler2 + (coupling_waves[j][k]) * (coupling_waves[j][k]).conjugate();
			}
			for (int k = 0; k < next_wave.size(); k++) {
				next_wave[k] = next_wave[k] - coupling_waves[j][k] * (scaler1 / scaler2);
			}
		}
		coupling_waves[i] = next_wave;	//copy
	}
}
*/

inline Wavefunction2D convertToFullThetaWaveInPolar(const Wavefunction2D& wave)
{
	NumericalGrid2D grid = wave.getGrid();
	NumericalGrid2D new_grid(grid.getCount(0), grid.getLength(0), grid.getOffset(0), grid.getCount(1) * 2, PI * 2, PI);
	Wavefunction2D new_wave(new_grid);
	
	GridIndice<2> indice;
	GridIndice<2> old_indice;
	for(int i = 0; i < new_grid.getTotalCount(); i++){
		GridIndice<2> indice = new_grid.expand(i);
		old_indice = indice;
		old_indice[1] = (indice[1] >= grid.getCount(1)) ? (2 * grid.getCount(1) - 1 - indice[1]) : indice[1];
		int old_id = grid.shrink(old_indice);
		new_wave.getSamplesHandler()(i) = wave.getValueByIndex(old_id);
	}
	return new_wave;
}


/*
template<unsigned N>
inline void schmidtOrthon(const std::vector< Wavefunction<N>* >& waves)
{
	for (int i = 0; i < waves.size(); i++) {
		auto tmp = *waves[i];
		for (int j = 0; j < i; j++) {
			//auto scaler = (*(waves[i])).innerProduct(*(waves[j])) / (*(waves[j])).innerProduct(*(waves[j]));
			auto scaler1 = (*waves[i]).innerProduct(*waves[j]);
			auto scaler2 = (*waves[j]).innerProduct(*waves[j]);
			tmp = tmp - (*(waves[j])) * (scaler1 / scaler2);
		}
		*waves[i] = tmp;
		//for (int j = 0; j < waves.size(); j++) {
		//	std::cout << (*waves[j]).norm() << std::endl;
		//}
	}
}
*/

/*
inline void schmidtOrthonForCouplingWavesInSH(const std::vector< std::vector< Wavefunction1D* > >& coupling_waves)
{
	for (int i = 0; i < coupling_waves.size(); i++) {
		auto next_wave = coupling_waves[i];
		for (int j = 0; j < i; j++) {
			//inner product for scaler -> yield a wavefunction! not a complex number.
			Wavefunction1D scaler1(next_wave.front()->getGrid());
			Wavefunction1D scaler2(next_wave.front()->getGrid());
			for (int k = 0; k < next_wave.size(); k++) {
				scaler1 = scaler1 + (*coupling_waves[i][k]) * (*coupling_waves[j][k]).conjugate();
				scaler2 = scaler2 + (*coupling_waves[j][k]) * (*coupling_waves[j][k]).conjugate();
			}
			for (int k = 0; k < next_wave.size(); k++) {
				(*next_wave[k]) = (*next_wave[k]) - (*coupling_waves[j][k]) * (scaler1 / scaler2);
			}
		}

		//copy
		for (int k = 0; k < next_wave.size(); k++) {
			(*coupling_waves[i][k]) = (*next_wave[k]);
		}
	}
}*/

}

#endif // !1