#ifndef __WAVE_FUNCTION_H__
#define __WAVE_FUNCTION_H__

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "numerical_grid.hpp"
#include "cqp_util.hpp"

namespace CQP {

typedef Eigen::VectorXcd WaveData;

template<unsigned N>
inline auto create_empty_wave(const NumericalGrid<N>& grid)
{
    return Eigen::VectorXcd::Zero(grid.getTotalCount());
}

inline auto create_empty_wave(int cnt)
{
    return Eigen::VectorXcd::Zero(cnt);
}

template<unsigned N>
inline auto create_random_wave(const NumericalGrid<N>& grid)
{
	return Eigen::VectorXcd::Random(grid.getTotalCount());
}

inline auto create_random_wave(int cnt)
{
	return Eigen::VectorXcd::Random(cnt);
}


template<unsigned N>
inline void update_wave_by_simple_expr(Eigen::Ref<Eigen::VectorXcd> data, const NumericalGrid<N>& grid, MathFuncType<N> expr, double t) 
{
	assert(data.rows() == grid.getTotalCount());

    if constexpr (N == 1) {
        for(int i = 0; i < grid.getTotalCount(); i++){
            data[i] = expr(grid.index(i).x(), t);
        }
    }
    else if constexpr (N == 2) {
        for(int i = 0; i < grid.getTotalCount(); i++){
            data[i] = expr(grid.index(i).x(), grid.index(i).y(), t);
        }
    }
    else if constexpr (N == 3) {
        for(int i = 0; i < grid.getTotalCount(); i++){
            data[i] = expr(grid.index(i).x(), grid.index(i).y(), grid.index(i).z(), t);
        }
    }
}

template<int N>
using LinSpaceData = Eigen::Array<double, Eigen::Dynamic, N>;
template<int N>
using LinSpaceData_f = Eigen::Array<float, Eigen::Dynamic, N>;

typedef LinSpaceData<1> LinSpaceData1D;
typedef LinSpaceData<2> LinSpaceData2D;
typedef LinSpaceData<3> LinSpaceData3D;

typedef LinSpaceData_f<1> LinSpaceData1D_f;
typedef LinSpaceData_f<2> LinSpaceData2D_f;
typedef LinSpaceData_f<3> LinSpaceData3D_f;

template<unsigned N>
inline LinSpaceData<N> create_linspace(const NumericalGrid<N>& grid)
{
	int total_cnt = grid.getTotalCount();
	LinSpaceData<N> data(total_cnt, N);

	// for x-fast indexing, the lower rank data changes faster.
	if constexpr (N == 1) {
		int cntx = grid.getCount(0);
		double x_lb = grid.get_value_by_index(0, X_DEGREE);
		double x_rb = grid.get_value_by_index(cntx - 1, X_DEGREE);
		data.col(0) = Eigen::ArrayXd::LinSpaced(cntx, x_lb, x_rb);
	}
	else if constexpr (N == 2) {
		int cntx = grid.getCount(0);
		int cnty = grid.getCount(1);
		double x_lb = grid.get_value_by_index(0, X_DEGREE);
		double x_rb = grid.get_value_by_index(cntx - 1, X_DEGREE);
		for(int i = 0; i < cnty; i++) {
			data(Eigen::seqN(i * cntx, cntx), 0) = Eigen::ArrayXd::LinSpaced(cntx, x_lb, x_rb);
			data(Eigen::seqN(i * cntx, cntx), 1) = Eigen::ArrayXd::Ones(cntx) * grid.get_value_by_index(i, Y_DEGREE);
		}
	}
	else if constexpr (N == 3) {
		int cntx = grid.getCount(0);
		int cnty = grid.getCount(1);
		int cntz = grid.getCount(2);
		double x_lb = grid.get_value_by_index(0, X_DEGREE);
		double x_rb = grid.get_value_by_index(cntx - 1, X_DEGREE);

#if 0
		#pragma omp single
		{
			int id = 0;
			for(int i = 0; i < cntz; i++){
				for(int j = 0; j < cnty; j++){
					#pragma omp task firstprivate(id)
					{
						id = i * cnty * cntx + j * cntx;
						data(Eigen::seqN(id, cntx), 0) = Eigen::ArrayXd::LinSpaced(cntx, x_lb, x_rb);
						data(Eigen::seqN(id, cntx), 1) = Eigen::ArrayXd::Ones(cntx) * grid.get_value_by_index(j, Y_DEGREE);
					}
				}
				#pragma omp task firstprivate(id)
				data(Eigen::seqN(i * cnty * cntx, cnty * cntx), 2) = Eigen::ArrayXd::Ones(cnty * cntx) * grid.get_value_by_index(i, Z_DEGREE);
			}
		}
	}
#else 
		for(int i = 0; i < cntz; i++){
			for(int j = 0; j < cnty; j++){
				int id = i * cnty * cntx + j * cntx;
				data(Eigen::seqN(id, cntx), 0) = Eigen::ArrayXd::LinSpaced(cntx, x_lb, x_rb);
				data(Eigen::seqN(id, cntx), 1) = Eigen::ArrayXd::Ones(cntx) * grid.get_value_by_index(j, Y_DEGREE);
			}
			data(Eigen::seqN(i * cnty * cntx, cnty * cntx), 2) = Eigen::ArrayXd::Ones(cnty * cntx) * grid.get_value_by_index(i, Z_DEGREE);
		}
	}
#endif //_USING_OMP


	return data;
}


template<unsigned N>
inline LinSpaceData_f<N> create_linspace_f(const NumericalGrid<N>& grid)
{
	int total_cnt = grid.getTotalCount();
	LinSpaceData_f<N> data(total_cnt, N);

	// for x-fast indexing, the lower rank data changes faster.
	if constexpr (N == 1) {
		int cntx = grid.getCount(0);
		float x_lb = static_cast<float>(grid.get_value_by_index(0, X_DEGREE));
		float x_rb = static_cast<float>(grid.get_value_by_index(cntx - 1, X_DEGREE));
		data.col(0) = Eigen::ArrayXf::LinSpaced(cntx, x_lb, x_rb);
	}
	else if constexpr (N == 2) {
		int cntx = grid.getCount(0);
		int cnty = grid.getCount(1);
		float x_lb = static_cast<float>(grid.get_value_by_index(0, X_DEGREE));
		float x_rb = static_cast<float>(grid.get_value_by_index(cntx - 1, X_DEGREE));
		for(int i = 0; i < cnty; i++) {
			data(Eigen::seqN(i * cntx, cntx), 0) = Eigen::ArrayXf::LinSpaced(cntx, x_lb, x_rb);
			data(Eigen::seqN(i * cntx, cntx), 1) = Eigen::ArrayXf::Ones(cntx) * grid.get_value_by_index(i, Y_DEGREE);
		}
	}
	else if constexpr (N == 3) {
		int cntx = grid.getCount(0);
		int cnty = grid.getCount(1);
		int cntz = grid.getCount(2);
		float x_lb = static_cast<float>(grid.get_value_by_index(0, X_DEGREE));
		float x_rb = static_cast<float>(grid.get_value_by_index(cntx - 1, X_DEGREE));
		
		for(int i = 0; i < cntz; i++){
			for(int j = 0; j < cnty; j++){
				int id = i * cnty * cntx + j * cntx;
				data(Eigen::seqN(id, cntx), 0) = Eigen::ArrayXf::LinSpaced(cntx, x_lb, x_rb);
				data(Eigen::seqN(id, cntx), 1) = Eigen::ArrayXf::Ones(cntx) * grid.get_value_by_index(j, Y_DEGREE);
			}
			data(Eigen::seqN(i * cnty * cntx, cnty * cntx), 2) = Eigen::ArrayXf::Ones(cnty * cntx) * grid.get_value_by_index(i, Z_DEGREE);
		}
	}

	return data;
}


template<unsigned N>
inline LinSpaceData<N> create_linspace_in_kspace(const NumericalGrid<N>& grid)
{
	int total_cnt = grid.getTotalCount();
	LinSpaceData<N> data(total_cnt, N);
	
	int cntx = grid.getCount(0);
	double x_lb = grid.get_value_by_index(0, X_DEGREE);
	double x_rb = grid.get_value_by_index(cntx - 1, X_DEGREE);
	double x_mid = grid.get_value_by_index(cntx / 2, X_DEGREE);
	double x_mid_left = grid.get_value_by_index(cntx / 2 - 1, X_DEGREE);

	// for x-fast indexing, the lower rank data changes faster.
	if constexpr (N == 1) {
		data(Eigen::seq(0, (cntx - 1) / 2), 0) = Eigen::ArrayXd::LinSpaced(cntx - cntx / 2, x_mid, x_rb);
		data(Eigen::seq((cntx - 1) / 2 + 1, cntx - 1), 0) = Eigen::ArrayXd::LinSpaced(cntx / 2, x_lb, x_mid_left);
	}
	else if constexpr (N == 2) {
		int cnty = grid.getCount(1);
		
		for(int i = 0; i < cnty; i++) {
			int crt_id = (i + cnty / 2) % cnty;
			int offset = i * cntx;
			data(Eigen::seq(offset + 0, offset + (cntx - 1) / 2), 0) = Eigen::ArrayXd::LinSpaced(cntx - cntx / 2, x_mid, x_rb);
			data(Eigen::seq(offset + (cntx - 1) / 2 + 1, offset + cntx - 1), 0) = Eigen::ArrayXd::LinSpaced(cntx / 2, x_lb, x_mid_left);
			data(Eigen::seqN(offset, cntx), 1) = Eigen::ArrayXd::Ones(cntx) * grid.get_value_by_index(crt_id, Y_DEGREE);
		}
	}
	else if constexpr (N == 3) {
		int cnty = grid.getCount(1);
		int cntz = grid.getCount(2);

		for(int i = 0; i < cntz; i++){
			int crt_z_id = (i + cntz / 2) % cntz;
			for(int j = 0; j < cnty; j++){
				int crt_y_id = (j + cnty / 2) % cnty;
				int offset = i * cnty * cntx + j * cntx;
				data(Eigen::seq(offset + 0, offset + (cntx - 1) / 2), 0) = Eigen::ArrayXd::LinSpaced(cntx - cntx / 2, x_mid, x_rb);
				data(Eigen::seq(offset + (cntx - 1) / 2 + 1, offset + cntx - 1), 0) = Eigen::ArrayXd::LinSpaced(cntx / 2, x_lb, x_mid_left);
				data(Eigen::seqN(offset, cntx), 1) = Eigen::ArrayXd::Ones(cntx) * grid.get_value_by_index(crt_y_id, Y_DEGREE);
			}
			data(Eigen::seqN(i * cnty * cntx, cnty * cntx), 2) = Eigen::ArrayXd::Ones(cnty * cntx) * grid.get_value_by_index(crt_z_id, Z_DEGREE);
		}
	}

	return data;
}

//#define INPUT_VEC_TYPE const Eigen::Ref<Eigen::ArrayXd>&
#define INPUT_VEC_TYPE const Eigen::Ref<Eigen::ArrayXd>&
#define INPUT_VEC_TYPE_F const Eigen::Ref<Eigen::ArrayXf>&
#define COMPLEX_T std::complex<double>
#define COMPLEX_T_F std::complex<float>

// All content below are real-array expression.
#define CC(content) 	((content).template cast<COMPLEX_T>())
#define EXP(content) 	((content).exp())
#define POW2(content)	((content).square())
#define COS(content) 	((content).cos())
#define SIN(content) 	((content).sin())
#define CIEXP(content)	(CC(COS(content)) + CC(SIN(content)) * 1i)
#define SQRT(content)	((content).sqrt())
#define RSQRT(content)	((content).rsqrt())
#define INV(content)	((content).inverse())

// add one-edge absorbing layer for potiential function
#define ABSORB(border, W) (((XS - (border)).sign() + 1) * (W) * -0.5i)


//usage: CREATE_VFUNC(EXP(XS * 2))
#define XS xs
#define YS ys
#define ZS zs
#define T t

// pre-defined macros for lambda expression, which is used to create vfunc.
#define CREATE_1D_VFUNC(expr) ([](INPUT_VEC_TYPE xs, double t) \
	{ using namespace std::literals; return expr; })

#define CREATE_1D_VFUNC_C(expr, ...) ([__VA_ARGS__](INPUT_VEC_TYPE xs, double t) \
	{ using namespace std::literals; return expr; })

#define CREATE_2D_VFUNC(expr) ([](INPUT_VEC_TYPE xs, INPUT_VEC_TYPE ys, double t) \
	{ using namespace std::literals; return expr; })

#define CREATE_2D_VFUNC_C(expr, ...) ([__VA_ARGS__](INPUT_VEC_TYPE xs, INPUT_VEC_TYPE ys, double t) \
	{ using namespace std::literals; return expr; })

#define CREATE_3D_VFUNC(expr) ([](INPUT_VEC_TYPE xs, INPUT_VEC_TYPE ys, INPUT_VEC_TYPE zs, double t) \
	{ using namespace std::literals; return expr; })

#define CREATE_3D_VFUNC_C(expr, ...) ([__VA_ARGS__](INPUT_VEC_TYPE xs, INPUT_VEC_TYPE ys, INPUT_VEC_TYPE zs, double t) \
	{ using namespace std::literals; return expr; })


inline auto gauss_vfunc_1d(double omega_x, double x0, double p0)
{
	using namespace std::literals;
	std::complex<double> x_scaler = 1.0 / std::pow(2 * PI * omega_x * omega_x, 0.25);

	return [=](INPUT_VEC_TYPE xs, double t) {
		return CC(EXP(POW2((xs - x0) / (2 * omega_x)) * -1.0))
			* (CIEXP(p0 * xs)) * x_scaler;
	};
}

inline auto gauss_vfunc_2d(
	double omega_x, double x0, double px0,
	double omega_y, double y0, double py0
)
{
	using namespace std::literals;
	std::complex<double> Cx = 1.0 / std::pow(2 * PI * omega_x * omega_x, 0.25);
	std::complex<double> Cy = 1.0 / std::pow(2 * PI * omega_y * omega_y, 0.25);

	return [=](const Eigen::Ref<Eigen::ArrayXd>& xs, const Eigen::Ref<Eigen::ArrayXd>& ys, double t) {
		return CC(EXP(POW2((xs - x0) / (2 * omega_x)) * -1.0))
			* (CC(COS(xs * px0)) + CC(SIN(xs * px0)) * 1i) * Cx
			* CC(EXP(POW2((ys - y0) / (2 * omega_y)) * -1.0))
			* (CC(COS(ys * py0)) + CC(SIN(ys * py0)) * 1i) * Cy;
	};
}

inline auto gauss_vfunc_3d(
	double omega_x, double x0, double px0,
	double omega_y, double y0, double py0,
	double omega_z, double z0, double pz0
)
{
	using namespace std::literals;
	std::complex<double> Cx = 1.0 / std::pow(2 * PI * omega_x * omega_x, 0.25);
	std::complex<double> Cy = 1.0 / std::pow(2 * PI * omega_y * omega_y, 0.25);
	std::complex<double> Cz = 1.0 / std::pow(2 * PI * omega_z * omega_z, 0.25);

	return [=](
		const Eigen::Ref<Eigen::ArrayXd>& xs,
		const Eigen::Ref<Eigen::ArrayXd>& ys,
		const Eigen::Ref<Eigen::ArrayXd>& zs, double t
	) {
		return CC(EXP(POW2((xs - x0) / (2 * omega_x)) * -1.0))
			* (CC(COS(xs * px0)) + CC(SIN(xs * px0)) * 1i) * Cx
			* CC(EXP(POW2((ys - y0) / (2 * omega_y)) * -1.0))
			* (CC(COS(ys * py0)) + CC(SIN(ys * py0)) * 1i) * Cy
			* CC(EXP(POW2((zs - z0) / (2 * omega_z)) * -1.0))
			* (CC(COS(zs * pz0)) + CC(SIN(zs * pz0)) * 1i) * Cz;
	};
}



template<int N, typename _Tf>
inline void update_wave_by_vectorized_func(Eigen::Ref<WaveData> wave, LinSpaceData<N>& xvars, const _Tf& func, double t = 0.0)
{ 
	TIME_TEST_START(update_wave)
	static_assert(!(N != 1 && N != 2 && N != 3), "the dimension N must be 1, 2 or 3.");
	//LinSpaceData<N>& ref = const_cast<LinSpaceData<N>&>(xvars);

#ifdef _USING_OPENMP_IN_VFUNC

	int block_size = wave.rows() / VFUNC_BLOCK_NUM;

	#pragma omp parallel for
	for(int i = 0; i < VFUNC_BLOCK_NUM; i++)
	{
		if (i == (VFUNC_BLOCK_NUM - 1)){
			auto seq = Eigen::seq(i * block_size, Eigen::last);
			if constexpr (N == 1)
				wave(seq) = func(xvars(seq, 0), t).matrix();
			else if constexpr (N == 2)
				wave(seq) = func(xvars(seq, 0), xvars(seq, 1), t).matrix();
			else if constexpr (N == 3)
				wave(seq) = func(xvars(seq, 0), xvars(seq, 1), xvars(seq, 2), t).matrix();
		}
		else {
			auto seq = Eigen::seqN(i * block_size, block_size);
			if constexpr (N == 1)
				wave(seq) = func(xvars(seq, 0), t).matrix();
			else if constexpr (N == 2)
				wave(seq) = func(xvars(seq, 0), xvars(seq, 1), t).matrix();
			else if constexpr (N == 3)
				wave(seq) = func(xvars(seq, 0), xvars(seq, 1), xvars(seq, 2), t).matrix();
		}
	}

#else
	if constexpr (N == 1)
		wave = func(xvars.col(0), t).matrix();
	else if constexpr (N == 2)
		wave = func(xvars.col(0), xvars.col(1), t).matrix();
	else if constexpr (N == 3)
		wave = func(xvars.col(0), xvars.col(1), xvars.col(2), t).matrix();
	else
		static_assert(!(N != 1 && N != 2 && N != 3), "the dimension N must be 1, 2 or 3.");

#endif //_USING_OPENMP_IN_VFUNC
	TIME_TEST_END(update_wave)
}


template<typename Derived1, typename Derived2, typename Derived3>
inline void fast_cwise_add(
	const Eigen::MatrixBase<Derived1>& 		input1,
	const Eigen::MatrixBase<Derived2>&		input2,
	Eigen::MatrixBase<Derived3>&			output,
	int 									block_num = CWISE_BLOCK_NUM
)
{
	assert(input1.rows() == input2.rows() && input1.rows() == output.rows());
	assert(input1.cols() == input2.cols() && input1.cols() == output.cols());

	TIME_TEST_START(f_cwise_add)

#ifdef _USING_OPENMP_IN_CWISE
	int block_size = input1.rows() / block_num;

	#pragma omp parallel for
	for(int i = 0; i < block_num; i++){
		if (i == (block_num - 1)){
			auto seq = Eigen::seq(i * block_size, Eigen::last);
			output(seq, Eigen::all) = input1(seq, Eigen::all) + input2(seq, Eigen::all);
		}
		else {
			auto seq = Eigen::seqN(i * block_size, block_size);
			output(seq, Eigen::all) = input1(seq, Eigen::all) + input2(seq, Eigen::all);
		}
	}
#else 
	output = input1 + input2;

#endif //_USING_OPENMP_IN_CWISE

	TIME_TEST_END(f_cwise_add)
}


template<typename Derived1, typename Derived2, typename Derived3>
inline void fast_cwise_add_tasks(
	const Eigen::MatrixBase<Derived1>& 		input1,
	const Eigen::MatrixBase<Derived2>&		input2,
	Eigen::MatrixBase<Derived3>&			output,
	int 									block_num = CWISE_BLOCK_NUM
)
{
	assert(input1.rows() == input2.rows() && input1.rows() == output.rows());
	assert(input1.cols() == input2.cols() && input1.cols() == output.cols());

#ifdef _USING_OPENMP_IN_CWISE
	int block_size = input1.rows() / block_num;

	for(int i = 0; i < block_num; i++){
		#pragma omp task
		{
			if (i == (block_num - 1)){
				auto seq = Eigen::seq(i * block_size, Eigen::last);
				output(seq, Eigen::all) = input1(seq, Eigen::all) + input2(seq, Eigen::all);
			}
			else {
				auto seq = Eigen::seqN(i * block_size, block_size);
				output(seq, Eigen::all) = input1(seq, Eigen::all) + input2(seq, Eigen::all);
			}
		}
	}
#else
	#pragma omp task
	output = input1 + input2;
#endif //_USING_OPENMP_IN_CWISE
}


template<typename Derived1, typename Derived2, typename Derived3>
inline void fast_cwise_multiply(
	const Eigen::MatrixBase<Derived1>& 		input1,
	const Eigen::MatrixBase<Derived2>&		input2,
	Eigen::MatrixBase<Derived3>&			output,
	int 									block_num = CWISE_BLOCK_NUM
)
{
	TIME_TEST_START(f_cwise_mul)
	assert(input1.rows() == input2.rows() && input1.rows() == output.rows());
	assert(input1.cols() == input2.cols() && input1.cols() == output.cols());

#ifdef _USING_OPENMP_IN_CWISE
	int block_size = input1.rows() / block_num;

	#pragma omp parallel for
	for(int i = 0; i < block_num; i++){
		if (i == (block_num - 1)){
			auto seq = Eigen::seq(i * block_size, Eigen::last);
			output(seq, Eigen::all) = input1(seq, Eigen::all).cwiseProduct(input2(seq, Eigen::all));
		}
		else {
			auto seq = Eigen::seqN(i * block_size, block_size);
			output(seq, Eigen::all) = input1(seq, Eigen::all).cwiseProduct(input2(seq, Eigen::all));
		}
	}
#else 
	output.array() = input1.array() * input2.array();

#endif //_USING_OPENMP_IN_CWISE
	TIME_TEST_END(f_cwise_mul)
}


template<typename Derived1, typename Derived2, typename Derived3>
inline void fast_cwise_multiply_tasks(
	const Eigen::MatrixBase<Derived1>& 		input1,
	const Eigen::MatrixBase<Derived2>&		input2,
	Eigen::MatrixBase<Derived3>&			output,
	int 									block_num = CWISE_BLOCK_NUM
)
{
	TIME_TEST_START(f_cwise_mul)
	assert(input1.rows() == input2.rows() && input1.rows() == output.rows());
	assert(input1.cols() == input2.cols() && input1.cols() == output.cols());

#ifdef _USING_OPENMP_IN_CWISE
	int block_size = input1.rows() / block_num;

	for(int i = 0; i < block_num; i++){
		#pragma omp task
		{
			if (i == (block_num - 1)){
				auto seq = Eigen::seq(i * block_size, Eigen::last);
				output(seq, Eigen::all) = input1(seq, Eigen::all).cwiseProduct(input2(seq, Eigen::all));
			}
			else {
				auto seq = Eigen::seqN(i * block_size, block_size);
				output(seq, Eigen::all) = input1(seq, Eigen::all).cwiseProduct(input2(seq, Eigen::all));
			}
		}
	}
#else
	#pragma omp task
	output.array() = input1.array() * input2.array();

#endif //_USING_OPENMP_IN_CWISE
	TIME_TEST_END(f_cwise_mul)
}



template<typename Derived1, typename Derived2, typename T>
inline void fast_cwise_scale(
	const Eigen::MatrixBase<Derived1>& 		input1,
	T										scaler,
	Eigen::MatrixBase<Derived2>&			output,
	int										block_num = CWISE_BLOCK_NUM
	)
{
	TIME_TEST_START(f_cwise_sc)
	assert(input1.rows() == output.rows());
	assert(input1.cols() == output.cols());

#ifdef _USING_OPENMP_IN_CWISE
	int block_size = input1.rows() / block_num;

	#pragma omp parallel for
	for(int i = 0; i < block_num; i++){
		if (i == (block_num - 1)){
			auto seq = Eigen::seq(i * block_size, Eigen::last);
			output(seq, Eigen::all) = input1(seq, Eigen::all) * scaler;
		}
		else {
			auto seq = Eigen::seqN(i * block_size, block_size);
			output(seq, Eigen::all) = input1(seq, Eigen::all) * scaler;
		}
	}
#else 
	output = input1 * scaler;

#endif //_USING_OPENMP_IN_CWISE
	TIME_TEST_END(f_cwise_sc)
}


template<typename Derived1, typename Derived2, typename T>
inline void fast_cwise_scale_tasks(
	const Eigen::MatrixBase<Derived1>& 		input1,
	T										scaler,
	Eigen::MatrixBase<Derived2>&			output,
	int										block_num = CWISE_BLOCK_NUM
	)
{
	TIME_TEST_START(f_cwise_sc)
	assert(input1.rows() == output.rows());
	assert(input1.cols() == output.cols());

#ifdef _USING_OPENMP_IN_CWISE
	int block_size = input1.rows() / block_num;

	for(int i = 0; i < block_num; i++){
		#pragma omp task
		{
			if (i == (block_num - 1)){
				auto seq = Eigen::seq(i * block_size, Eigen::last);
				output(seq, Eigen::all) = input1(seq, Eigen::all) * scaler;
			}
			else {
				auto seq = Eigen::seqN(i * block_size, block_size);
				output(seq, Eigen::all) = input1(seq, Eigen::all) * scaler;
			}
		}
	}
#else
	#pragma omp task
	output = input1 * scaler;

#endif //_USING_OPENMP_IN_CWISE
	TIME_TEST_END(f_cwise_sc)
}


inline void fast_inner_product(
	const Eigen::Ref<WaveData>&		input1,
	const Eigen::Ref<WaveData>&		input2,
	std::complex<double>&			output)
{
	assert(input1.rows() == input2.rows() && input1.cols() == input2.cols());

#ifdef _USING_OPENMP_IN_DOT
	TIME_TEST_START(dot_task)

	std::vector<std::complex<double>> buffer(DOT_BLOCK_NUM, 0);
	int block_size = input1.rows() / std::max(DOT_BLOCK_NUM - 1, 1);
	int residual_size = input1.rows() % std::max(DOT_BLOCK_NUM - 1, 1);
	//int block_cnt = DOT_BLOCK_NUM - 1 + (residual_size != 0);

	for(int i = 0; i < DOT_BLOCK_NUM; i++){
		#pragma omp task shared(buffer, block_size, residual_size) firstprivate(i)
		{
			TIME_TEST_START(dot_subtask)
			if (i == (DOT_BLOCK_NUM - 1) && residual_size != 0){
				buffer[i] = input1.bottomRows(residual_size)
					.dot(input2.bottomRows(residual_size));
			}
			else {	
				buffer[i] = input1.middleRows(i * block_size, block_size)
					.dot(input2.middleRows(i * block_size, block_size));
			}
			TIME_TEST_END(dot_subtask)
		}
	}
	
	#pragma omp taskwait

	output = 0;
	for(int i = 0; i < DOT_BLOCK_NUM; i++){		// reduction
		output += buffer[i];
	}

	TIME_TEST_END(dot_task)

#else
	// do inner_product directly
	output = input1.dot(input2);

#endif
}


inline void fast_gram_schmidt(std::vector<WaveData>& waves)
{
	TIME_TEST_START(f_gram_schmidt)
	std::vector<std::complex<double>> self_product_buffer(waves.size(), 0.0);

	for (int i = 0; i < waves.size(); i++) {
		fast_inner_product(waves[i], waves[i], self_product_buffer[i]);
	}

	std::complex<double> tmp = 0;
	std::complex<double> scaler = 0;
	for (int i = 0; i < waves.size(); i++) {
		for (int j = 0; j < i; j++) {
			fast_inner_product(waves[j], waves[i], tmp);
			scaler = tmp / self_product_buffer[j];
			//waves[i] = waves[i] - waves[j] * scaler;
			fast_cwise_add(waves[i], - waves[j] * scaler, waves[i]);
		}
	}
	
	TIME_TEST_END(f_gram_schmidt)
}


}


#endif //__WAVE_FUNCTION_H__