#ifndef __CQP_FFT_HPP__
#define __CQP_FFT_HPP__

#include "wave_function.hpp"
#include "cqp_util.hpp"
#include <fftw3.h>

namespace CQP {

template<unsigned N>
struct FFTCore
{
    int                 cnt;
    fftw_complex *      pre_defined_in;
    fftw_complex *      pre_defined_out;
    fftw_plan           plan;
    int                 cnt_reverse[N];

    FFTCore(const NumericalGrid<N>& grid)
    {
        cnt = grid.getTotalCount();
        pre_defined_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * cnt);
        pre_defined_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * cnt);
        plan = NULL;

        for (int i = 0; i < N; i++) {
			cnt_reverse[N - 1 - i] = grid.getCount(i);
		}
    }

    ~FFTCore(){
        fftw_free(pre_defined_in);
        fftw_free(pre_defined_out);
        if(plan != NULL){
            fftw_destroy_plan(plan);
        }
    }
};

typedef FFTCore<1> FFTCore1D;
typedef FFTCore<2> FFTCore2D;
typedef FFTCore<3> FFTCore3D;

template<unsigned N>
inline auto create_fft_core(const NumericalGrid<N>& grid) 
{
    return FFTCore<N>(grid);
}


template<unsigned N>
inline void fft(
    Eigen::Ref<WaveData>    wave_in,
    Eigen::Ref<WaveData>    wave_out,
    FFTCore<N>&             fft_core,
    int                     direction
)
{
    TIME_TEST_START(fft_process)

    assert(direction == FFTW_FORWARD || direction == FFTW_BACKWARD);
    assert(wave_in.rows() == fft_core.cnt && wave_in.rows() == wave_out.rows());

    fftw_complex * ptr_in = reinterpret_cast<fftw_complex*>(wave_in.data());
    fftw_complex * ptr_out = reinterpret_cast<fftw_complex*>(wave_out.data());

    //if it is the first time to call fft, then create the plan.
    if(fft_core.plan == NULL) {
        unsigned flags = FFTW_ESTIMATE;
        
        // deal with the alignment issue.
        if(fftw_alignment_of(reinterpret_cast<double*>(ptr_in))
        != fftw_alignment_of(reinterpret_cast<double*>(fft_core.pre_defined_in)) ||
        fftw_alignment_of(reinterpret_cast<double*>(ptr_out))
        != fftw_alignment_of(reinterpret_cast<double*>(fft_core.pre_defined_out)))
        {
            std::cout << "[WARNING] (fft) Alignment does not match, which may decrease the performance." << std::endl;
            flags |= FFTW_UNALIGNED;
        }

        // create the plan
        fft_core.plan = fftw_plan_dft(N, fft_core.cnt_reverse, fft_core.pre_defined_in, fft_core.pre_defined_out, direction, flags);
    }

    TIME_TEST(fftw_execute_dft(fft_core.plan, ptr_in, ptr_out), infact_fft);
    // wave_out /= std::sqrt((double)fft_core.cnt);

    TIME_TEST_END(fft_process)
}


inline void fft_1d(
    Eigen::Ref<WaveData>    wave_in,
    Eigen::Ref<WaveData>    wave_out,
    FFTCore1D&              fft_core,
    int                     direction
)
{
    assert(direction == FFTW_FORWARD || direction == FFTW_BACKWARD);
    assert(wave_in.rows() == fft_core.cnt && wave_in.rows() == wave_out.rows());

    fftw_complex * ptr_in = reinterpret_cast<fftw_complex*>(wave_in.data());
    fftw_complex * ptr_out = reinterpret_cast<fftw_complex*>(wave_out.data());
    
    //if it is the first time to call fft, then create the plan.
    if(fft_core.plan == NULL) {
        unsigned flags = FFTW_ESTIMATE;
        // deal with the alignment issue.
        if(fftw_alignment_of(reinterpret_cast<double*>(ptr_in))
        != fftw_alignment_of(reinterpret_cast<double*>(fft_core.pre_defined_in)) ||
        fftw_alignment_of(reinterpret_cast<double*>(ptr_out))
        != fftw_alignment_of(reinterpret_cast<double*>(fft_core.pre_defined_out)))
        {
            std::cout << "[WARNING] (fft) Alignment does not match." << std::endl;
            flags |= FFTW_UNALIGNED;
        }
        fft_core.plan = fftw_plan_dft_1d(fft_core.cnt, fft_core.pre_defined_in, fft_core.pre_defined_out, direction, flags);
    }

    TIME_TEST(fftw_execute_dft(fft_core.plan, ptr_in, ptr_out), infact_fft);
    wave_out /= std::sqrt((double)fft_core.cnt);
}


inline void fft_1d_with_copy(
    Eigen::Ref<WaveData>    wave_in,
    Eigen::Ref<WaveData>    wave_out,
    FFTCore1D&              fft_core,
    int                     direction
)
{
    assert(direction == FFTW_FORWARD || direction == FFTW_BACKWARD);
    assert(wave_in.rows() == fft_core.cnt && wave_in.rows() == wave_out.rows());

    fftw_complex * ptr_in = reinterpret_cast<fftw_complex*>(wave_in.data());
    fftw_complex * ptr_out = reinterpret_cast<fftw_complex*>(wave_out.data());

    if(fft_core.plan == NULL){
        fft_core.plan = fftw_plan_dft_1d(fft_core.cnt, fft_core.pre_defined_in, fft_core.pre_defined_out, direction, FFTW_ESTIMATE);   
    }

    memcpy(fft_core.pre_defined_in, ptr_in, fft_core.cnt * sizeof(fftw_complex));

    TIME_TEST(fftw_execute(fft_core.plan), infact_fft);

    memcpy(ptr_out, fft_core.pre_defined_out, fft_core.cnt * sizeof(fftw_complex));
    wave_out /= std::sqrt((double)fft_core.cnt);
}


template<unsigned N>
inline NumericalGrid<N> grid_after_fft(const NumericalGrid<N>& grid)
{
    int cnt[N] = {0}; // x-fast index
	double xmid[N] = {0.0}, kmid[N] = {0.0}, length_k[N] = {0.0};

    for (int i = 0; i < N; i++) {
		cnt[i] = grid.getCount(i);
		xmid[i] = grid.getOffset(i);
		kmid[i] = 0.0;
		length_k[i] = 2 * PI / grid.getLength(i) * grid.getCount(i);
	}

    return NumericalGrid<N>(cnt, length_k, kmid);
}


}

#endif