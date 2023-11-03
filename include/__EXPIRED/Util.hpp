#ifndef __UTIL_H__
#define __UTIL_H__

#include <cmath>
#include <functional>
#include <omp.h>
#include <fftw3.h>
#include <Eigen/Dense>
#include <iostream>
#define PI acos(-1)

#define _USING_OMP

#ifdef _USING_OMP

#define THREAD_NUM_CQP 16
#define _USING_OPENMP_IN_FFT
#define _USING_OPENMP_IN_FD
#define _USING_OPENMP_IN_TOOLS
#define _USING_OPENMP_IN_ITP
//define _MEASURE_MODE_IN_FFT

#endif 

namespace CQP {

constexpr double MINIMAL_APPROX_EQUAL_CRITERION = 1e-8;

struct RandomNum
{
	long long seed;

	RandomNum(long long seed_): seed(seed_) {}

	unsigned int lcg(int mod) {
		seed = (25214903917LL * seed + 11LL) & ((1LL << 48) - 1LL);
		return (unsigned int)(seed % mod);
	}
};


template<typename T>
inline bool isApproxEqual(const T& a, const T& b)
{
	return abs(a - b) < MINIMAL_APPROX_EQUAL_CRITERION;
}


//some tricks... about template
template<unsigned N>
struct MathFunc {	
};

template<>
struct MathFunc<1> {
	typedef std::function<std::complex<double>(double)> _type;
};

template<>
struct MathFunc<2> {
	typedef std::function<std::complex<double>(double, double)> _type;
};

template<>
struct MathFunc<3> {
	typedef std::function<std::complex<double>(double, double, double)> _type;
};

template<unsigned N>
using MathFuncType = typename MathFunc<N>::_type;



inline void init_for_openmp()
{
#ifdef _USING_OMP
	omp_set_num_threads(THREAD_NUM_CQP);
	int n = Eigen::nbThreads();
	std::cout << "Num of threads: " << Eigen::nbThreads() << std::endl;

	std::cout << "In multi-thread mode." << std::endl;

#ifdef _USING_OPENMP_IN_FFT
	int res = fftw_init_threads();
	if(res == 0){
		std::cout << "There is a problem with multi-threads initialization." << std::endl;
	}
	fftw_plan_with_nthreads(THREAD_NUM_CQP);
#endif

#endif	
}


}

#endif // !__UTIL_H__
