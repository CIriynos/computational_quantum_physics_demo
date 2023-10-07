/*****************************************************************//**
 * \file   Wavefunction.h
 * \brief  Define the template class "Wavefunction", and other functions like fft.
 * 
 * \author twq_email@163.cc
 * \date   September 2023
 *********************************************************************/

#ifndef __WAVE_FUNCTION_H__
#define __WAVE_FUNCTION_H__

#include <vector>
#include <complex>
#include <cmath>
#include "NumericalGrid.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include <fftw3.h>
#include "Util.hpp"
#include <iostream>

#ifdef _USING_OMP

#include <omp.h>

#endif

/**
 *  the class Wavefunction<N> describes a "matter wave" in quantum physics, which can be
 *	operated like a function in mathmatics, such as Add, Minus, Scale Multiply, and 
 *	Inner Product. A Wavefunction is initialized by a NumercialGrid<N>, which indicates
 *	many parameters to store a discrete function, such as the number of the grid points, 
 *	the actual range for PDE (in our case it is Schrodinger Equation). 
 */

namespace CQP {

template<unsigned N>
class Wavefunction
{
public:
	constexpr static int N_plus_one = N + 1;
	constexpr static int N_number = N;
	
	Wavefunction();

	Wavefunction(const NumericalGrid<N>&);

	Wavefunction(const NumericalGrid<N>&, const std::vector<std::complex<double> >&);
	Wavefunction(const NumericalGrid<N>&, const Eigen::VectorXcd&);

	Wavefunction(const Wavefunction<N>&);
	Wavefunction(Wavefunction<N>&&) noexcept;
	Wavefunction<N>& operator=(const Wavefunction<N>&);
	Wavefunction<N>& operator=(Wavefunction<N>&&) noexcept;

	const NumericalGrid<N>& getGrid() const;
	//std::complex<double> getValue(const GridPoint<N>&) const;
	std::complex<double> getValueByIndex(int) const;
	const Eigen::VectorXcd& getSamplesView() const;
	Eigen::VectorXcd& getSamplesHandler();
	std::complex<double>& operator[](int);

	template<unsigned _N>
	Wavefunction<_N> getSlice(std::initializer_list< std::pair<int, int> >) const;

	template<unsigned _N>
	Wavefunction<_N> uplift(const NumericalGrid<_N>&) const;

	Wavefunction<Wavefunction<N>::N_plus_one> upliftOnce(int, int, double, double) const;

	Wavefunction<N> replaceGrid(const NumericalGrid<N>&);

	std::complex<double> norm() const;
	std::complex<double> innerProduct(const Wavefunction<N>&) const;
	Wavefunction<N> conjugate() const;
	Wavefunction<N> operator+(const Wavefunction<N>&) const;
	Wavefunction<N> operator-(const Wavefunction<N>&) const;
	Wavefunction<N> operator*(std::complex<double>) const;
	Wavefunction<N> operator*(const Wavefunction<N>&) const;
	Wavefunction<N> operator/(std::complex<double>) const;
	Wavefunction<N> operator/(const Wavefunction<N>&) const;
	
	void normalize();

private:
	NumericalGrid<N> grid;
	Eigen::VectorXcd samples;
};

typedef Wavefunction<1> Wavefunction1D;
typedef Wavefunction<2> Wavefunction2D;
typedef Wavefunction<3> Wavefunction3D;


template<unsigned N>
Wavefunction<N>::Wavefunction()
	: grid(NumericalGrid<N>())
{
}

template<unsigned N>
Wavefunction<N>::Wavefunction(const NumericalGrid<N>& grid_)
	: grid(grid_), samples(Eigen::VectorXcd::Zero(grid_.getTotalCount()))
{
}

template<unsigned N>
Wavefunction<N>::Wavefunction(const NumericalGrid<N>& grid_, const std::vector<std::complex<double>>& data)
	: grid(grid_), samples(Eigen::VectorXcd::Zero(grid_.getTotalCount()))
{
	assert(grid.getTotalCount() == data.size());
	for (int i = 0; i < samples.size(); i++) {
		samples(i) = data[i];
	}
}

template<unsigned N>
Wavefunction<N>::Wavefunction(const NumericalGrid<N>& grid_, const Eigen::VectorXcd& samples_)
	: grid(grid_), samples(samples_)
{
	assert(grid_.getTotalCount() == samples_.size());
}

template<unsigned N>
Wavefunction<N>::Wavefunction(const Wavefunction<N>& wave)
	: grid(wave.grid), samples(wave.samples)
{
}

template<unsigned N>
Wavefunction<N>::Wavefunction(Wavefunction<N>&& wave) noexcept
	: grid(std::move(wave.grid)), samples(std::move(wave.samples))
{
}

template<unsigned N>
Wavefunction<N>& Wavefunction<N>::operator=(const Wavefunction<N>& wave)
{
	grid = wave.grid;
	samples = wave.samples;
	return *this;
}

template<unsigned N>
inline Wavefunction<N>& Wavefunction<N>::operator=(Wavefunction<N>&& wave) noexcept
{
	grid = wave.grid;
	samples = std::move(wave.samples);
	return *this;
}

template<unsigned N>
inline const NumericalGrid<N>& Wavefunction<N>::getGrid() const
{
	return grid;
}

template<unsigned N>
inline std::complex<double> Wavefunction<N>::getValueByIndex(int id) const
{
	return samples(id);
}

template<unsigned N>
inline const Eigen::VectorXcd& Wavefunction<N>::getSamplesView() const
{
	return samples;
}

template<unsigned N>
inline Eigen::VectorXcd& Wavefunction<N>::getSamplesHandler()
{
	return samples;
}

template<unsigned N>
inline std::complex<double>& Wavefunction<N>::operator[](int id)
{
	std::complex<double>& ref = samples(id);
	return ref;
}

template<unsigned N>
inline std::complex<double> Wavefunction<N>::norm() const
{
	return innerProduct(*this);
}

template<unsigned N>
inline std::complex<double> Wavefunction<N>::innerProduct(const Wavefunction<N>& wave) const
{
	assert(grid == wave.getGrid());
	std::complex<double> res = wave.samples.adjoint() * samples;
	//std::complex<double> delta = 1;
	//for (int i = 0; i < N; i++) {
	//	delta *= grid.getDelta(i);
	//}
	return res;
}

template<unsigned N>
inline Wavefunction<N> Wavefunction<N>::operator+(const Wavefunction<N>& wave) const
{
	assert(grid == wave.grid);
	auto res = samples + wave.samples;
	Wavefunction<N> new_wave(grid, res);
	return new_wave;
}

template<unsigned N>
inline Wavefunction<N> Wavefunction<N>::operator-(const Wavefunction<N>& wave) const
{
	assert(grid == wave.grid);
	auto res = samples - wave.samples;
	Wavefunction<N> new_wave(grid, res);
	return new_wave;
}

template<unsigned N>
inline Wavefunction<N> Wavefunction<N>::operator*(std::complex<double> scaler) const
{
	auto res = samples * scaler;
	Wavefunction<N> new_wave(grid, res);
	return new_wave;
}

template<unsigned N>
inline Wavefunction<N> Wavefunction<N>::operator*(const Wavefunction<N>& wave) const
{
	assert(grid == wave.grid);
	auto res = (samples.array() * wave.samples.array()).matrix();
	Wavefunction<N> new_wave(grid, res);
	return new_wave;
}

template<unsigned N>
inline Wavefunction<N> Wavefunction<N>::operator/(std::complex<double> scaler) const
{
	auto res = samples / scaler;
	Wavefunction<N> new_wave(grid, res);
	return new_wave;
}

template<unsigned N>
inline Wavefunction<N> Wavefunction<N>::operator/(const Wavefunction<N>& wave) const
{
	assert(grid == wave.grid);
	auto res = (samples.array() / wave.samples.array()).matrix();
	Wavefunction<N> new_wave(grid, res);
	return new_wave;
}

template<unsigned N>
inline Wavefunction<N> Wavefunction<N>::conjugate() const
{
	return Wavefunction<N>(grid, samples.conjugate());
}

template<unsigned N>
inline void Wavefunction<N>::normalize()
{
	samples.normalize();
}


template<unsigned N>
template<unsigned _N>
inline Wavefunction<_N> Wavefunction<N>::getSlice(std::initializer_list< std::pair<int, int> > L) const
{
	assert(N - L.size() == _N);

	int sub_degrees[_N] = { 0 };
	int flags[N] = { 0 };
	int flags_index[N] = { 0 };
	int k = 0;

	for (auto tmp : L) {
		flags[tmp.first] = 1;  //means degree of "tmp" is unavailable.
		flags_index[tmp.first] = tmp.second;
	}
	for (int i = 0; i < N; i++) {
		if (flags[i] == 0) {
			sub_degrees[k] = i;
			k++;
		}
	}

	NumericalGrid<_N> sub_grid = grid.template subset<_N>(std::vector<int>(sub_degrees, sub_degrees + _N));
	Wavefunction<_N> sub_wave(sub_grid);

	int full_indice[N] = { 0 };

	for (int i = 0; i < sub_grid.getTotalCount(); i++) {
		auto indice = sub_grid.expand(i);
		int m = 0;
		for (int j = 0; j < N; j++) {
			if (flags[j] == 1) {
				full_indice[j] = flags_index[j];
			}
			else {
				full_indice[j] = indice[m];
				m++;
			}
		}
		sub_wave.getSamplesHandler()(i) = samples(grid.shrink(full_indice));

		for (int j = 0; j < N; j++) {	//clear
			full_indice[j] = 0;
		}
		m = 0;
	}

	return sub_wave;
}

template<unsigned N>
template<unsigned _N>
inline Wavefunction<_N> Wavefunction<N>::uplift(const NumericalGrid<_N>& new_grid) const
{
	assert(_N >= N);
	Wavefunction<_N> upwave(new_grid);

	for (int i = 0; i < new_grid.getTotalCount(); i++) {
		auto indice = new_grid.expand(i);
		auto sub_indice = indice.template sub<N>(0, N);
		auto id = grid.shrink(sub_indice);
		upwave[i] = samples(id);
	}
	return upwave;
}

template<unsigned N>
inline Wavefunction<Wavefunction<N>::N_plus_one> Wavefunction<N>::upliftOnce(int degree, int cnt, double L, double off) const
{
	NumericalGrid<N_plus_one> new_grid = grid.supersetOnce(degree, cnt, L, off);
	Wavefunction<N_plus_one> upwave(new_grid);

	for (int i = 0; i < new_grid.getTotalCount(); i++) {
		GridIndice<N_plus_one> indice = new_grid.expand(i);
		GridIndice<N> old_indice = indice.erase(degree);
		auto id = grid.shrink(old_indice);
		upwave.getSamplesHandler()(i) = samples(id);
	}
	return upwave;
}

template<unsigned N>
inline Wavefunction<N> Wavefunction<N>::replaceGrid(const NumericalGrid<N>&)
{
	return Wavefunction<N>();
}



inline Wavefunction1D create1DWaveByExpression(const NumericalGrid1D& grid, std::function<std::complex<double>(double)> func)
{
	std::vector< std::complex<double> > buffer(grid.getTotalCount(), 0);

	double x = 0;
#ifdef _USING_OPENMP_IN_TOOLS
	#pragma omp parallel for firstprivate(x)
#endif
	for (int i = 0; i < grid.getTotalCount(); i++) {
		x = grid.index(i).x();
		buffer[i] = func(x);
	}
	Wavefunction1D wave(grid, buffer);
	return wave;
}

inline Wavefunction2D create2DWaveByExpression(const NumericalGrid2D& grid, std::function<std::complex<double>(double, double)> func)
{
	std::vector< std::complex<double> > buffer(grid.getTotalCount(), 0);

	double x = 0, y = 0;
#ifdef _USING_OPENMP_IN_TOOLS
	#pragma omp parallel for firstprivate(x, y)
#endif
	for (int i = 0; i < grid.getTotalCount(); i++) {
		x = grid.index(i).x();
		y = grid.index(i).y();
		buffer[i] = func(x, y);
	}
	Wavefunction2D wave(grid, buffer);
	return wave;
}

inline Wavefunction3D create3DWaveByExpression(const NumericalGrid3D& grid, std::function<std::complex<double>(double, double, double)> func)
{
	std::vector< std::complex<double> > buffer(grid.getTotalCount(), 0);

	double x = 0, y = 0, z = 0;
#ifdef _USING_OPENMP_IN_TOOLS
	#pragma omp parallel for firstprivate(x, y, z)
#endif
	for (int i = 0; i < grid.getTotalCount(); i++) {
		x = grid.index(i).x();
		y = grid.index(i).y();
		z = grid.index(i).z();
		buffer[i] = func(x, y, z);
	}
	Wavefunction3D wave(grid, buffer);
	return wave;
}


template<unsigned N>
inline Wavefunction<N> createWaveByExpression(const NumericalGrid<N>& grid, MathFuncType<N> func)
{
	//static_assert(std::is_same_v<typename MathFunc<N>::_type, _Tf>, "the input math function is not the type MathFunc<N>.");

	if constexpr (N == 1) {
		return create1DWaveByExpression(grid, func);
	} else if constexpr (N == 2) {
		return create2DWaveByExpression(grid, func);
	} else if constexpr (N == 3) {
		return create3DWaveByExpression(grid, func);
	} else {
		static_assert((N >= 1 && N <= 3), "Temporarily, the function does not support N which is bigger than 3.");
	}
}



}


#endif // !__WAVE_FUNCTION_H__