#ifndef __TDSE_FFT_HPP__
#define __TDSE_FFT_HPP__

#include "cqp_util.hpp"
#include "wave_function.hpp"
#include "cqp_fft.hpp"

namespace CQP {

template<unsigned N>
struct TDSEBufferFFT
{
    int                 cnt;
    int                 cond;

    // necessary vars
    WaveData            phase_factor_1_tmp;
    WaveData            phase_factor_1;
    WaveData            phase_factor_2;

    LinSpaceData<N>     linspace_x;
    LinSpaceData<N>     linspace_k;
    FFTCore<N>          core_forward;
    FFTCore<N>          core_backward;

    // runtime vars in main loop
    Eigen::ArrayXd      tmp_po_data;
    WaveData            po_data;
    WaveData            k_state;

    TDSEBufferFFT(const NumericalGrid<N>& grid, int cond)
        : cnt(grid.getTotalCount()), cond(cond),

        phase_factor_1_tmp(create_empty_wave(grid)),
        phase_factor_1(create_empty_wave(grid)),
        phase_factor_2(create_empty_wave(grid)),

        linspace_x(create_linspace(grid)),
        linspace_k(create_linspace_in_kspace(grid_after_fft(grid))),
        core_forward(grid), core_backward(grid),

        tmp_po_data(cnt, 1),
        po_data(create_empty_wave(grid)),
        k_state(create_empty_wave(grid))
    {
        assert(cond == PERIODIC_BOUNDARY_COND
            || cond == IMAG_TIME_PROPAGATION_COND);
    }
};


template<unsigned N>
inline TDSEBufferFFT<N> init_split_operator(const NumericalGrid<N>& grid, int cond)
{
    return TDSEBufferFFT<N>(grid, cond);
}


// when calling this function, be sure that the fft_core has been updated.

template<unsigned N>
inline void split_operator_method_mainloop_no_time(
    TDSEBufferFFT<N>&       bf,
    Eigen::Ref<WaveData>    crt_state,
    int                     total_steps
)
{
    TIME_TEST_START(split_mainloop_nt)
    using namespace std::complex_literals;
    assert(total_steps >= 0);
    assert(crt_state.rows() == bf.cnt);

    std::complex<double> scaler = 1.0 / std::sqrt((double)bf.cnt);
    
    fft(crt_state, bf.k_state, bf.core_forward, FFTW_FORWARD);
    // main loop of split-operator method.
    for(int i = 0; i < total_steps; i++){
        fast_cwise_multiply(bf.k_state, bf.phase_factor_1, bf.k_state);
        fft(bf.k_state, crt_state, bf.core_backward, FFTW_BACKWARD);
        fast_cwise_multiply(crt_state, bf.phase_factor_2 * scaler, crt_state);
        fft(crt_state, bf.k_state, bf.core_forward, FFTW_FORWARD);
        fast_cwise_multiply(bf.k_state, bf.phase_factor_1 * scaler, bf.k_state);
    }
    fft(bf.k_state, crt_state, bf.core_backward, FFTW_BACKWARD);
    fast_cwise_scale(crt_state, scaler * scaler, crt_state);

    TIME_TEST_END(split_mainloop_nt)
}


template<unsigned N>
void update_fft_core_factor_1(
    TDSEBufferFFT<N>&       bf,
    double                  delta_t
)
{
    TIME_TEST_START(update_factor1)
    using namespace std::complex_literals;
    assert(delta_t > 0.0);
    
    auto dt = (bf.cond == IMAG_TIME_PROPAGATION_COND) ? (-1i * delta_t) : delta_t;
    
    // update phase factor 1
    if constexpr (N == 1) {
        auto expr1 = CREATE_1D_VFUNC_C( CIEXP( POW2(XS) * dt * (-0.25) ), dt );
        update_wave_by_vectorized_func(bf.phase_factor_1, bf.linspace_k, expr1, 0.0);
    } 
    else if constexpr (N == 2) {
        auto expr1 = CREATE_2D_VFUNC_C( CIEXP(POW2(XS) * dt * (-0.25)) * CIEXP(POW2(YS) * dt * (-0.25)), dt);
        update_wave_by_vectorized_func(bf.phase_factor_1, bf.linspace_k, expr1, 0.0);
    } 
    else if constexpr (N == 3) {
        auto expr1 = CREATE_3D_VFUNC_C( CIEXP(POW2(XS) * dt * (-0.25)) * CIEXP(POW2(YS) * dt * (-0.25)) * CIEXP(POW2(ZS) * dt * (-0.25)), dt);
        update_wave_by_vectorized_func(bf.phase_factor_1, bf.linspace_k, expr1, 0.0);
    }

    TIME_TEST_END(update_factor1)
}

template<unsigned N, typename _Tf>
void update_fft_core_factor_2(
    TDSEBufferFFT<N>&       bf,
    _Tf                     potiential_func,
    double                  delta_t,
    double                  crt_t
)
{
    TIME_TEST_START(update_factor2)
    using namespace std::complex_literals;
    assert(delta_t > 0.0);
    
    auto dt = (bf.cond == IMAG_TIME_PROPAGATION_COND) ? (-1i * delta_t) : delta_t;

    // update po_data by current time (crt_t)
    update_wave_by_vectorized_func(bf.po_data, bf.linspace_x, potiential_func, crt_t);

    // update phase factor 2
    auto expr2 = CREATE_1D_VFUNC_C( CIEXP( XS * dt * (-1.0) ), dt );
    bf.tmp_po_data = bf.po_data.real().array();
    update_wave_by_vectorized_func(bf.phase_factor_2, bf.tmp_po_data, expr2, 0.0);

    TIME_TEST_END(update_factor2)
}


template<unsigned N, typename _Tf>
void update_fft_core_no_time(
    TDSEBufferFFT<N>&       bf,
    _Tf                     potiential_func,
    double                  delta_t
)
{
    update_fft_core_factor_1(bf, delta_t);
    update_fft_core_factor_2(bf, potiential_func, delta_t, 0.0);
}


template<unsigned N, typename _Tf>
inline void split_operator_method_no_time(
    TDSEBufferFFT<N>&       bf,
    Eigen::Ref<WaveData>    crt_state,
    _Tf                     potiential_func,
    double                  delta_t,
    int                     total_steps
)
{
    using namespace std::complex_literals;
    assert(total_steps >= 0 && delta_t > 0);
    assert(crt_state.rows() == bf.cnt);

    update_fft_core_no_time(bf, potiential_func, delta_t);
    split_operator_method_mainloop_no_time(bf, crt_state, total_steps);
}


template<unsigned N, typename _Tf>
inline void split_operator_method(
    TDSEBufferFFT<N>&       bf,
    Eigen::Ref<WaveData>    crt_state,
    _Tf                     potiential_func,
    double                  start_t,
    double                  delta_t,
    int                     total_steps
)
{
    using namespace std::complex_literals;
    assert(total_steps >= 0 && delta_t > 0);
    assert(crt_state.rows() == bf.cnt);

    // first, update phase_factor_1 because it's not time-related.
    update_fft_core_factor_1(bf, delta_t);

    // main loop of split-operator method. time related.
    fft(crt_state, bf.k_state, bf.core_forward, FFTW_FORWARD);
    for(int i = 0; i < total_steps; i++)
    {
        // update phase_factor_2 for each step. time-related.
        update_fft_core_factor_2(bf, potiential_func, delta_t, start_t + i * delta_t);

        bf.k_state.array() *= bf.phase_factor_1.array();
        fft(bf.k_state, crt_state, bf.core_backward, FFTW_BACKWARD);
        crt_state.array() *= bf.phase_factor_2.array();
        fft(crt_state, bf.k_state, bf.core_forward, FFTW_FORWARD);
        bf.k_state.array() *= bf.phase_factor_1.array();
    }
    fft(bf.k_state, crt_state, bf.core_backward, FFTW_BACKWARD);
}


/*
template<typename _Tf>
inline void split_operator_method_1d(
    TDSEBufferFFT<1>&        bf,
    Eigen::Ref<WaveData>    crt_state,
    _Tf                     potiential_func,
    double                  start_t,
    double                  delta_t,
    int                     total_steps,
    bool                    is_time_dependent = true
)
{
    using namespace std::complex_literals;
    assert(total_steps >= 0 && delta_t > 0);
    assert(crt_state.rows() == bf.cnt);

    if(is_time_dependent == true){
        auto dt = (bf.cond == IMAG_TIME_PROPAGATION_COND) ? (-1i * delta_t) : delta_t;
        auto expr1 = CREATE_1D_VFUNC_C( CIEXP( POW2(XS) * dt * (-0.25) ), dt );
        auto expr2 = CREATE_1D_VFUNC_C( CIEXP( XS * dt * (-1.0) ), dt );

        update_wave_by_vectorized_func_1d(bf.phase_factor_1, bf.linspace_k, expr1, 0.0);
        //change_wave_to_kspace(bf.phase_factor_1_tmp, bf.phase_factor_1);

        fft_1d(crt_state, bf.k_state, bf.core_forward, FFTW_FORWARD);

        for(int i = 0; i < total_steps; i++){
            double crt_t = start_t + delta_t * i;
            update_wave_by_vectorized_func_1d(bf.po_data, bf.linspace_x, potiential_func, crt_t);
            bf.tmp_po_data = bf.po_data.real().array();
            update_wave_by_vectorized_func_1d(bf.phase_factor_2, bf.tmp_po_data, expr2, crt_t);

            bf.k_state.array() *= bf.phase_factor_1.array();
            fft_1d(bf.k_state, crt_state, bf.core_backward, FFTW_BACKWARD);
            crt_state.array() *= bf.phase_factor_2.array();
            fft_1d(crt_state, bf.k_state, bf.core_forward, FFTW_FORWARD);
            bf.k_state.array() *= bf.phase_factor_1.array();
        }

        fft_1d(bf.k_state, crt_state, bf.core_backward, FFTW_BACKWARD);
    }
    else{
        auto dt = (bf.cond == IMAG_TIME_PROPAGATION_COND) ? (-1i * delta_t) : delta_t;
        update_wave_by_vectorized_func_1d(bf.po_data, bf.linspace_x, potiential_func, 0.0);

        auto expr1 = CREATE_1D_VFUNC_C( CIEXP( POW2(XS) * dt * (-0.25) ), dt );
        auto expr2 = CREATE_1D_VFUNC_C( CIEXP( XS * dt * (-1.0) ), dt );

        update_wave_by_vectorized_func_1d(bf.phase_factor_1, bf.linspace_k, expr1, 0.0);
        bf.tmp_po_data = bf.po_data.real().array();
        update_wave_by_vectorized_func_1d(bf.phase_factor_2, bf.tmp_po_data, expr2, 0.0);

        fft_1d(crt_state, bf.k_state, bf.core_forward, FFTW_FORWARD);

        for(int i = 0; i < total_steps; i++){
            bf.k_state.array() *= bf.phase_factor_1.array();
            fft_1d(bf.k_state, crt_state, bf.core_backward, FFTW_BACKWARD);
            crt_state.array() *= bf.phase_factor_2.array();
            fft_1d(crt_state, bf.k_state, bf.core_forward, FFTW_FORWARD);
            bf.k_state.array() *= bf.phase_factor_1.array();
        }
        fft_1d(bf.k_state, crt_state, bf.core_backward, FFTW_BACKWARD);
    }
}
*/


}


#endif //__TDSE_FFT_HPP__