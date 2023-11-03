#ifndef __CQP_UTIL_H__
#define __CQP_UTIL_H__

#include <cmath>
#include <functional>
#include <omp.h>
#include <fftw3.h>
#include <Eigen/Dense>
#include <iostream>

#define PI acos(-1)

#define _USING_OMP
#define _USING_TIME_TEST

// configuration on omp
#ifdef _USING_OMP

// here, set the num of threads used in openmp.
#define THREAD_NUM_CQP 64

#define _USING_OPENMP_IN_FFT
#define _USING_OPENMP_IN_FD
#define _USING_OPENMP_IN_ITP

#define _USING_OPENMP_IN_CREATE_SPMAT

#define _USING_OPENMP_IN_VFUNC
#define VFUNC_BLOCK_NUM (THREAD_NUM_CQP)

#define _USING_OPENMP_IN_CWISE
#define CWISE_BLOCK_NUM (THREAD_NUM_CQP)

//#define _USING_OPENMP_IN_DOT
//#define DOT_BLOCK_NUM (4)
#define MAX_BUFFER_SIZE_OF_INNER_PRODUCT_SH (200 * 200)

#define _USING_OPENMP_IN_SH
#define THREAD_NUM_SH (THREAD_NUM_CQP)

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
	//(double x, double t) -> double 
	typedef double (*_type)(double, double);
};

template<>
struct MathFunc<2> {
	//(double x, double y, double t) -> double 
	typedef double (*_type)(double, double, double);
};

template<>
struct MathFunc<3> {
	//(double x, double y, double z, double t) -> double 
	typedef double (*_type)(double, double, double, double);
};

template<unsigned N>
using MathFuncType = typename MathFunc<N>::_type;

typedef MathFuncType<1> MathFuncType1D;
typedef MathFuncType<2> MathFuncType2D;
typedef MathFuncType<3> MathFuncType3D;



inline void init_for_openmp()
{
#ifdef _USING_OMP
	omp_set_num_threads(THREAD_NUM_CQP);

	// make sure that Eigen is using openMP.
	//int n = Eigen::nbThreads();
	Eigen::setNbThreads(1);
	std::cout << "Num of threads: " << THREAD_NUM_CQP << std::endl;
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

#endif // !__CQP_UTIL_H__
