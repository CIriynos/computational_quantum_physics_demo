#ifndef __CQP_ITP_HPP__
#define __CQP_ITP_HPP__

#include "wave_function.hpp"
#include "tdse_fd.hpp"
#include "tdse_fft.hpp"
#include "cqp_sh.hpp"
#include "tdse_sh_laser.hpp"

namespace CQP {

//////////////////////////////////
// ITP & energy

struct ExpectedValueSolvingBufferSH
{   
    int                 l_num;
    std::vector<int>    lmap;
    std::vector<SpMat>  H_part_l;   // size: l_num

    // vars used in forward-backward procedure.
    SpMat               M2;
    SpMat               M2_boost;
    SpMat               M2_T;
    SpMat               M2_boost_T;

    std::vector<TrimatElimBuffer>    A_add_list;     // size: l_num * l_num
    std::vector<TrimatElimBuffer>    B_add_list;

    // phi -> phi' (after forward-backward)
    std::vector<WaveData>       rfunc_tmps;    // size: l_num * l_num
    WaveDataSH                  res_state;     // used in get residual value

    // Note: DO NOT call this constructor directly. use init function.
    explicit ExpectedValueSolvingBufferSH(const TDSEBufferSH& bf)
        : l_num(bf.l_num), lmap(bf.l_map),
        H_part_l(bf.l_num, SpMat(bf.cnt_r, bf.cnt_r)), 

        M2(bf.core_r.M), M2_boost(bf.core_r_coulomb.M),
        M2_T(bf.core_r.M.transpose()), M2_boost_T(bf.core_r_coulomb.M.transpose()),

        A_add_list(l_num * l_num, TrimatElimBuffer(bf.cnt_r, 0)),
        B_add_list(l_num * l_num, TrimatElimBuffer(bf.cnt_r, 0)),

        rfunc_tmps(l_num * l_num, create_empty_wave(bf.cnt_r)),
        res_state(Eigen::MatrixXcd::Zero(bf.cnt_r, l_num * l_num))
    {}
    
    // Note: DO NOT call this constructor directly. use init function.
    explicit ExpectedValueSolvingBufferSH(const CQP::TDSEBufferPolarLaserSH& bf_pl)
        : l_num(bf_pl.l_num), lmap(l_num * l_num),
        H_part_l(bf_pl.l_num, SpMat(bf_pl.cnt_r, bf_pl.cnt_r)),

        M2(bf_pl.core.M2), M2_boost(bf_pl.core.M2_boosted),
        M2_T(bf_pl.core.M2.transpose()), M2_boost_T(bf_pl.core.M2_boosted.transpose()),

        A_add_list(l_num * l_num, TrimatElimBuffer(bf_pl.core.cnt_r, 0)),
        B_add_list(l_num * l_num, TrimatElimBuffer(bf_pl.core.cnt_r, 0)),

        rfunc_tmps(l_num * l_num, create_empty_wave(bf_pl.core.cnt_r)),
        res_state(Eigen::MatrixXcd::Zero(bf_pl.cnt_r, l_num * l_num))
    {
        update_lmap_for_sh(lmap, l_num);
    }
};


inline ExpectedValueSolvingBufferSH init_expected_value_solver(const TDSEBufferSH& bf)
{
    ExpectedValueSolvingBufferSH solver_bf(bf);

    for(int l = 0; l < solver_bf.l_num; l++){
        if (l == 0) {
            solver_bf.H_part_l[l] = bf.core_r_coulomb.D + bf.core_r_coulomb.M * bf.rts[l].po_data.asDiagonal();
        } else {
            solver_bf.H_part_l[l] = bf.core_r.D + bf.core_r.M * bf.rts[l].po_data.asDiagonal();
        }
    }
    return solver_bf;
}


inline ExpectedValueSolvingBufferSH init_expected_value_solver(const TDSEBufferPolarLaserSH& bf)
{
    ExpectedValueSolvingBufferSH solver_bf(bf);

    for(int l = 0; l < solver_bf.l_num; l++){
        if (l == 0) {
            solver_bf.H_part_l[l] = bf.core.D2_boosted + bf.core.M2_boosted * bf.po_data_l[l].asDiagonal();
        } else {
            solver_bf.H_part_l[l] = bf.core.D2 + bf.core.M2 * bf.po_data_l[l].asDiagonal();
        }
    }
    return solver_bf;
}


inline double get_energy_sh(
    ExpectedValueSolvingBufferSH&   bf,
    Eigen::Ref<WaveDataSH>          shwave
)
{
    double energy = 0;
    for(int i = 0; i < bf.l_num * bf.l_num; i++){
        int l = bf.lmap[i];
        if (l == 0) {
            tridiagonal_mat_elimination_optimized(bf.M2_boost_T, bf.rfunc_tmps[i], shwave.col(i), bf.A_add_list[i], bf.B_add_list[i]);
        } else {
            tridiagonal_mat_elimination_optimized(bf.M2_T, bf.rfunc_tmps[i], shwave.col(i), bf.A_add_list[i], bf.B_add_list[i]);
        }
        energy += bf.rfunc_tmps[i].dot(bf.H_part_l[l] * shwave.col(i)).real();
    }
    return energy;
}


inline std::complex<double> get_residual_norm_sh(
    ExpectedValueSolvingBufferSH&   bf,
    Eigen::Ref<WaveDataSH>          shwave
)
{
    double energy = get_energy_sh(bf, shwave);

    for(int i = 0; i < bf.l_num * bf.l_num; i++){
        int l = bf.lmap[i];
        bf.rfunc_tmps[i] = bf.H_part_l[l] * shwave.col(i);  // now, rfunc_tmps are used to store tmp result.
        if (l == 0) {
            tridiagonal_mat_elimination_optimized(bf.M2_boost, bf.res_state.col(i), bf.rfunc_tmps[i], bf.A_add_list[i], bf.B_add_list[i]);
        } else {
            tridiagonal_mat_elimination_optimized(bf.M2, bf.res_state.col(i), bf.rfunc_tmps[i], bf.A_add_list[i], bf.B_add_list[i]);
        }
        bf.res_state.col(i) = bf.res_state.col(i) - energy * shwave.col(i);
    }

    std::complex<double> norm = inner_product_sh(shwave, shwave);
    return norm;
}


template<typename _Tf>
inline std::vector<double> imag_time_propagation_sh(
    const NumericalGrid1D&      rgrid,
    std::vector<WaveDataSH>&    shwaves,    // user need to initialize them manually.
    int                         l_num,

    _Tf                         r_po_func,      // potiential_func
    bool                        coulomb_flag,
    double                      po_Z,
    
    double                      delta_t,
    double                      convergence_error = 1e-6
    )
{
    TIME_TEST_START(itp_sh)

    std::vector<double> energy_results(shwaves.size(), 0.0);
    TDSEBufferSH bf = init_fd_sh(rgrid, l_num, IMAG_TIME_PROPAGATION_COND, coulomb_flag, po_Z);
    update_runtime_sh(bf, r_po_func, delta_t, 0.0);

    // Caution: update tdse buffer first, then create ExpectedValueSolver buffer
    ExpectedValueSolvingBufferSH exbf = init_expected_value_solver(bf);
    
    // mainloop for imag_time_propagation_sh
    int loop_times = 0;
    double last_energy = 1000000.0;   // relatively impossible value
    double crt_energy = 0;
    double energy_diff = 1000000.0;

    do {
        for(int j = 0; j < shwaves.size(); j++){
            tdse_fd_sh_mainloop_no_time(bf, shwaves[j], 1);
            normalize_sh(shwaves[j]);
        }
        gram_schmidt_sh(shwaves);

        // check convergence per 10 loops
        // only shwaves[last], cuz the last state converge most slowly, practically.
        if(loop_times % 10 == 0){
            crt_energy = get_energy_sh(exbf, shwaves.back());
            energy_diff = std::abs(crt_energy - last_energy);
            last_energy = crt_energy;
        }
        
        loop_times ++;
    } while(energy_diff >= convergence_error);

    for(int j = 0; j < shwaves.size(); j++){
        energy_results[j] = get_energy_sh(exbf, shwaves[j]);
    }
    
    TIME_TEST_END(itp_sh)
    return energy_results;
}


// init strategy
inline void init_shwaves_for_itp(
    const NumericalGrid1D&      rgrid,
    std::vector<WaveDataSH>&    shwaves,
    int                         l_num
)
{
    LinSpaceData1D linspace = create_linspace(rgrid);
    std::vector<int> lmap(l_num * l_num, 0);
    std::vector<int> select_list;

    update_lmap_for_sh(lmap, l_num);
    int cnt = 0, flag = 0;
    for(int n = 1; n < 100; n++){   // 100 may be enough.
        for(int l = 0; l < n; l++){
            for(int m = -l; m <= l; m++){
                if(cnt >= shwaves.size()) goto end_loop;
                select_list.push_back(get_index_from_lm(l, m, l_num));
                cnt ++;
            }
        }
    }
    end_loop:;
    
    for(int i = 0; i < shwaves.size(); i++){
        shwaves[i] = create_empty_wave_sh(rgrid, l_num);
        auto func = CREATE_1D_VFUNC_C(XS * EXP(-XS / (i + 1.0)), i);
        update_wave_by_vectorized_func(shwaves[i].col(select_list[i]), linspace, func);
    }
}


}

#endif //__CQP_ITP_HPP__