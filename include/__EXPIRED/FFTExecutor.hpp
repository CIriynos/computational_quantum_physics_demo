#ifndef __FFT_EXECUTOR_H__
#define __FFT_EXECUTOR_H__

#include "Wavefunction.hpp"
#include "Util.hpp"
#include "TimeTest.h"

namespace CQP {

constexpr int FFT_D = 0;
constexpr int IFFT_D = 1;

template<unsigned N, unsigned FFTDirection>
class FFTExecutor
{
public:
	static_assert((FFTDirection == FFT_D || FFTDirection == IFFT_D), "FFTDirection must be FFT_D(0) or IFFT_D(1).");
	static_assert((N > 0), "The dimension N must be an unsigned integer.");

	FFTExecutor(const FFTExecutor<N, FFTDirection>& obj)
	{
		FFTExecutor(obj.assigned_grid);
	}

	FFTExecutor<N, FFTDirection>& operator=(const FFTExecutor<N, FFTDirection>& obj)
	{
		FFTExecutor(obj.assigned_grid);
	}

	FFTExecutor(const NumericalGrid<N>& grid)
		: assigned_grid(grid), is_initialized(false), sign(-1), 
		//new_wave_data(Eigen::VectorXcd::Zero(grid.getTotalCount())),
		new_wave_data(grid.getTotalCount(), 0),
		tmp_return(new_grid, new_wave_data)
	{
		//std::cout << "fft executor created with grid : " << std::endl;
		//std::cout << grid << std::endl;

		int mode = 0;
		if (FFTDirection == FFT_D) {
			mode = FFTW_FORWARD;
			sign = -1;
		}
		else if(FFTDirection == IFFT_D){
			mode = FFTW_BACKWARD;
			sign = 1;
		}

		for (int i = 0; i < N; i++) {
			cnt[i] = assigned_grid.getCount(i);
			xmid[i] = assigned_grid.getOffset(i);
			kmid[i] = 0.0;
			length_k[i] = 2 * PI / assigned_grid.getLength(i) * assigned_grid.getCount(i);
			cnt_reverse[N - 1 - i] = assigned_grid.getCount(i);
		}

		in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * grid.getTotalCount());
		out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * grid.getTotalCount());

#ifdef _MEASURE_MODE_IN_FFT
		p = fftw_plan_dft(N, cnt_reverse, in, out, mode, FFTW_MEASURE);
#else
		p = fftw_plan_dft(N, cnt_reverse, in, out, mode, FFTW_ESTIMATE);
#endif
		new_grid = NumericalGrid<N>(cnt, length_k, kmid);
		
	}

	FFTExecutor<N, FFTDirection>& inputWave(const Wavefunction<N>& wave)
	{
		using namespace std::literals;
		assert(wave.getGrid() == assigned_grid);
		
		int fft_index[N] = {0};
#ifdef _USING_OPENMP_IN_FFT
		#pragma omp parallel for firstprivate(fft_index)
#endif
		for (int i = 0; i < assigned_grid.getTotalCount(); i++) {
			auto indice = assigned_grid.expand(i);
			for (int rank = 0; rank < N; rank++) {
				int split_index = (int)std::floor((double)cnt[rank] / 2.0);
				fft_index[rank] = indice[rank] - split_index;
				if (indice[rank] < split_index) {
					fft_index[rank] += cnt[rank];
				}
			}
			in[assigned_grid.shrink(fft_index)][0] = wave.getValueByIndex(i).real();
			in[assigned_grid.shrink(fft_index)][1] = wave.getValueByIndex(i).imag();
		}

		is_initialized = true;

		return *this;
	}

	Wavefunction<N> execute()
	{
		assert(is_initialized == true);

		int total_cnt = assigned_grid.getTotalCount();
		int split_index[N] = {0};

		//vars for parallel 
		int grid_index[N] = {0};
		GridIndice<N> indice;		
		int ptr = 0;
		
		//init split_index
		for(int rank = 0; rank < N; rank++){
			split_index[rank] = (int)std::floor((double)cnt[rank] / 2.0);
		}

		//new_wave_data.assign(new_wave_data.size(), 0);	//total: 0.003 ms
		
		TIME_TEST(fftw_execute(p), infact_fft);			//total: 0.002 ms
		
#ifdef _USING_OPENMP_IN_FFT
		#pragma omp parallel for firstprivate(grid_index, indice, ptr)
#endif
		for (int i = 0; i < total_cnt; i++) {
			indice = assigned_grid.expand(i);				//total: 0.014 ms

			for (int rank = 0; rank < N; rank++) {			//total: 0.011 ms
				grid_index[rank] = indice[rank] + split_index[rank];
				grid_index[rank] -= ((cnt[rank] - 1 - indice[rank]) < split_index[rank]) * cnt[rank];
			}

			//grid_index[0] = indice[0] + split_index[0] - ((cnt[0] - 1 - indice[0]) < split_index[0]) * cnt[0];
			//grid_index[1] = indice[1] + split_index[1] - ((cnt[1] - 1 - indice[1]) < split_index[1]) * cnt[1];
			//grid_index[2] = indice[2] + split_index[2] - ((cnt[2] - 1 - indice[2]) < split_index[2]) * cnt[2];

			ptr = assigned_grid.shrink(grid_index);			//total: almost 0ms
			new_wave_data[ptr] = std::complex<double>(out[i][0], out[i][1]) / sqrt(total_cnt);	//total: 0.009 ms
		}

		return tmp_return;
	}

	/*
	Wavefunction<N> getResult()
	{
		std::vector< std::complex<double> > new_wave_data(assigned_grid.getTotalCount(), 0);
		int grid_index[N] = {0};
		int total_cnt = assigned_grid.getTotalCount();

		for (int i = 0; i < total_cnt; i++) {
			GridIndice<N> indice = assigned_grid.expand(i);
			for (int rank = 0; rank < N; rank++) {
				int split_index = (int)std::floor((double)cnt[rank] / 2.0);
				grid_index[rank] = indice[rank] + split_index;
				if ((cnt[rank] - 1 - indice[rank]) < split_index) {
					grid_index[rank] -= cnt[rank];
				}
			}
			int ptr = assigned_grid.shrink(grid_index);
			new_wave_data[ptr] = std::complex<double>(out[i][0], out[i][1]) / sqrt(total_cnt);
		}

		return Wavefunction<N>(new_grid, new_wave_data);
	}
	*/

	~FFTExecutor()
	{
		fftw_destroy_plan(p);
		fftw_free(in);
		fftw_free(out);
	}

private:
	fftw_complex* in;
	fftw_complex* out;
	fftw_plan p;

	NumericalGrid<N> assigned_grid;
	NumericalGrid<N> new_grid;
	
	//Eigen::VectorXcd new_wave_data;
	std::vector< std::complex<double> > new_wave_data;
	Wavefunction<N> tmp_return;

	int cnt[N]; //x-fast index
	int cnt_reverse[N];  //y-fast index

	double xmid[N], kmid[N], length_k[N];
	double sign;

	bool is_initialized;
};


#ifdef _USING_OPENMP_IN_FFT

template<unsigned N>
inline Wavefunction<N> fft(const Wavefunction<N>& wave, const std::vector<double>& k)
{
	static FFTExecutor<N, FFT_D> executor(wave.getGrid());
	executor.inputWave(wave);
	return executor.execute();
}

template<unsigned N>
inline Wavefunction<N> ifft(const Wavefunction<N>& wave, const std::vector<double>& k)
{
	static FFTExecutor<N, IFFT_D> executor(wave.getGrid());
	executor.inputWave(wave);
	return executor.execute();
}

#else

template<unsigned N>
inline Wavefunction<N> fft(const Wavefunction<N>& wave, const std::vector<double>& k)
{
	return fft_agent_func(k, wave.getGrid(), wave, FFTW_FORWARD);
}

template<unsigned N>
inline Wavefunction<N> ifft(const Wavefunction<N>& wave, const std::vector<double>& k) {
	return fft_agent_func(k, wave.getGrid(), wave, FFTW_BACKWARD);
}

#endif


template<unsigned N>
inline Wavefunction<N> fft_agent_func(std::vector<double> center_k_vector, const NumericalGrid<N>& grid, const Wavefunction<N>& wave, int mode)
{
	assert(mode == FFTW_FORWARD || mode == FFTW_BACKWARD);
	assert(center_k_vector.size() == N);
	using namespace std::literals;

	int cnt[N]; //x-fast index
	double xmid[N], kmid[N], length_k[N];
	int cnt_reverse[N];  //y-fast index
	for (int i = 0; i < N; i++) {
		cnt[i] = grid.getCount(i);
		xmid[i] = grid.getOffset(i);
		length_k[i] = 2 * PI / grid.getLength(i) * grid.getCount(i);
		kmid[i] = center_k_vector[i];
		cnt_reverse[N - 1 - i] = grid.getCount(i);
	}

	double sign = 1;
	if (mode == FFTW_FORWARD) {
		sign = -1;
	}
	else if (mode == FFTW_BACKWARD) {
		sign = 1;
	}

	NumericalGrid<N> new_grid = NumericalGrid<N>(cnt, length_k, kmid);
	std::vector< std::complex<double> > new_wave_data(grid.getTotalCount(), 0);
	fftw_complex* in, * out;
	fftw_plan p;

	in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * grid.getTotalCount());
	out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * grid.getTotalCount());
	p = fftw_plan_dft(N, cnt_reverse, in, out, mode, FFTW_ESTIMATE);
	
	int fft_index[N], grid_index[N];
	for (int i = 0; i < grid.getTotalCount(); i++) {
		auto indice = grid.expand(i);
		for (int rank = 0; rank < N; rank++) {
			int split_index = (int)std::floor((double)cnt[rank] / 2.0);
			fft_index[rank] = indice[rank] - split_index;
			if (indice[rank] < split_index) {
				fft_index[rank] += cnt[rank];
			}
		}
		in[grid.shrink(fft_index)][0] = wave.getValueByIndex(i).real();
		in[grid.shrink(fft_index)][1] = wave.getValueByIndex(i).imag();
	}

	fftw_execute(p);

	for (int i = 0; i < grid.getTotalCount(); i++) {
		auto indice = grid.expand(i);
		for (int rank = 0; rank < N; rank++) {
			int split_index = (int)std::floor((double)cnt[rank] / 2.0);
			grid_index[rank] = indice[rank] + split_index;
			if ((cnt[rank] - 1 - indice[rank]) < split_index) {
				grid_index[rank] -= cnt[rank];
			}
		}
		new_wave_data[grid.shrink(grid_index)] = std::complex<double>(out[i][0], out[i][1]) / sqrt(grid.getTotalCount());
		//std::cout << "id: " << grid.shrink(grid_index) << std::endl;
	}

	//for (int i = 0; i < grid.getTotalCount(); i++) {
	//	//std::cout << std::complex<double>(out[i][0], out[i][1]) << " ";
	//	std::cout << new_wave_data[i] << " ";
	//}
	//std::cout << std::endl;
	//getchar();

	fftw_destroy_plan(p);
	fftw_free(in);
	fftw_free(out);

	return Wavefunction<N>(new_grid, new_wave_data);
}

}

#endif