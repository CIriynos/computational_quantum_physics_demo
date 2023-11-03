#ifndef __TDSE_SH_LASER_HPP__
#define __TDSE_SH_LASER_HPP__

#include "cqp_util.hpp"
#include "cqp_sh.hpp"
#include "tdse_fd.hpp"
#include <complex>
#include "cqp_time_test.h"

namespace CQP {

SpMat create_M1_for_sh(const NumericalGrid1D& grid);

SpMat create_D1_for_sh(const NumericalGrid1D& grid);

struct TDSECorePolarLaserSH
{
    // key parameters for sh
    int                     cnt_r;
    int                     l_num;
    NumericalGrid1D         rgrid;
    LinSpaceData<1>         rlinspace;
    
    // neccessary r-space matrix.
    SpMat                   M2;
    SpMat                   D2;
    SpMat                   M2_boosted;
    SpMat                   D2_boosted;
    SpMat                   M1;
    SpMat                   D1;
    SpMat                   Ir;
    SpMat                   Rr;

    // the potiential function, which is luckily time-indepentent.
    WaveData                po_data;
    
    // Note: DO NOT call this constructor directly. use init function.
    TDSECorePolarLaserSH(const NumericalGrid1D &rgrid, const WaveData& po_data, int l_num)  // l_num should be the max_l_num.
        :   cnt_r(rgrid.getTotalCount()), l_num(l_num),
            rgrid(rgrid), rlinspace(create_linspace(rgrid)),

            M2(cnt_r, cnt_r), D2(cnt_r, cnt_r),
            M2_boosted(cnt_r, cnt_r), D2_boosted(cnt_r, cnt_r),
            M1(cnt_r, cnt_r), D1(cnt_r, cnt_r),
            Ir(cnt_r, cnt_r), Rr(cnt_r, cnt_r),

            po_data(po_data)
        {}
};


struct TDSEBufferPolarLaserSH
{
    int                     cnt_r;
    int                     l_num;
    std::vector<int>        lmap;
    
    TDSECorePolarLaserSH    core;

    // neccessary l-space matrices, but m-related.
    // the order is: m = 0, 1, -1, 2, -2, ..., lmax, -lmax (m -> m') size = 2 * l_num - 1
    std::vector<Eigen::MatrixXcd>   Ls;
    std::vector<Eigen::MatrixXcd>   Ts;
    std::vector<Eigen::MatrixXcd>   Bs;
    std::vector<Eigen::MatrixXcd>   As;
    std::vector<Eigen::MatrixXcd>   BT_A;
    std::vector<Eigen::MatrixXcd>   AT_B;
    
    std::vector<Eigen::MatrixXcd>   Ls_in_diag;
    std::vector<Eigen::MatrixXcd>   Ts_in_diag;

    std::vector<WaveData>           po_data_l;          // size: l_num
    std::vector<SpMat>              Hat_rmats_neg;      // size: l_num
    std::vector<SpMat>              Hat_rmats_pos;
    
    // temp vars in mainloop
    std::vector<Eigen::MatrixXcd>   tmp_bundle;         // 2 * l_num - 1
    std::vector<Eigen::MatrixXcd>   tmp_bundle_2;

    std::vector<TrimatElimBuffer>   tri_elim_bfs_A;     // size: l_num * l_num
    std::vector<TrimatElimBuffer>   tri_elim_bfs_B;     // size: l_num * l_num
    std::vector<SpMat>              diag_rmats;         // size: l_num * l_num

    TDSEBufferPolarLaserSH(const NumericalGrid1D &rgrid, const WaveData& po_data, int l_num)
        : 
        cnt_r(rgrid.getTotalCount()), l_num(l_num), lmap(l_num * l_num),
        core(rgrid, po_data, l_num),

        //Ls(2 * l_num - 1), Ts(2 * l_num - 1),
        //Bs(2 * l_num - 1), B_invs(2 * l_num - 1),
        //As(2 * l_num - 1), As_inv(2 * l_num - 1),
        //tmp_bundle(2 * l_num - 1, Eigen::MatrixXcd::Zero(cnt_r, l_num)),
        po_data_l(l_num, create_empty_wave(rgrid)),
        Hat_rmats_neg(l_num, SpMat(cnt_r, cnt_r)),
        Hat_rmats_pos(l_num, SpMat(cnt_r, cnt_r)),

        tri_elim_bfs_A(l_num * l_num, TrimatElimBuffer(cnt_r, 0.0)),
        tri_elim_bfs_B(l_num * l_num, TrimatElimBuffer(cnt_r, 0.0)),
        diag_rmats(l_num * l_num, SpMat(cnt_r, cnt_r))

        {}
};


template<typename _Tf>
inline TDSEBufferPolarLaserSH init_tdse_sh_pl(const NumericalGrid1D &rgrid, _Tf po_func, int l_num, double Z)
{
    using namespace std::complex_literals;

    WaveData po_data = create_empty_wave(rgrid);
    WaveData r_rev_data = create_empty_wave(rgrid);
    LinSpaceData1D r_linspace = create_linspace(rgrid);

    update_wave_by_vectorized_func(po_data, r_linspace, po_func);
    update_wave_by_vectorized_func(r_rev_data, r_linspace, CREATE_1D_VFUNC(INV(XS)));

    TDSEBufferPolarLaserSH bf = TDSEBufferPolarLaserSH(rgrid, po_data, l_num);
    double delta_x = rgrid.getDelta(0);

    // init lmap
    update_lmap_for_sh(bf.lmap, l_num);
    
    // first, update core's r-space matrics
    bf.core.Ir.setIdentity();
    bf.core.Rr = r_rev_data.asDiagonal(); bf.core.Rr.makeCompressed();
    bf.core.D2 = create_matrix_hamiltonian(rgrid, REFLECTING_BOUNDARY_COND);
    bf.core.D2 = bf.core.D2 * (-2.0);
    bf.core.M2 = create_M2_for_sh(rgrid);

    std::complex<double> d = (-2.0 / (delta_x * delta_x)) * (1.0 - Z * delta_x / (12.0 - 10.0 * Z * delta_x)); 
    bf.core.D2_boosted = bf.core.D2;
    bf.core.D2_boosted.coeffRef(0, 0) = d;
    bf.core.M2_boosted = bf.core.M2;
    bf.core.M2_boosted.coeffRef(0, 0) = -2.0 * (1.0 + delta_x * delta_x * d / 12.0);
    bf.core.D2_boosted.makeCompressed();
    bf.core.M2_boosted.makeCompressed();

    bf.core.M1 = create_M1_for_sh(rgrid);
    bf.core.D1 = create_D1_for_sh(rgrid);

    // create l-space matrices by emplace_back() into std::vector
    for(size_t i = 0; i < 2 * l_num - 1; i++){
        int m = get_m_from_mm(i);
        int block_size = l_num - std::abs(m);
        bf.Ls.emplace_back(block_size, block_size);
        bf.Ts.emplace_back(block_size, block_size);
        bf.Bs.emplace_back(block_size, block_size);
        bf.As.emplace_back(block_size, block_size);
        bf.BT_A.emplace_back(block_size, block_size);
        bf.AT_B.emplace_back(block_size, block_size);
        bf.Ls_in_diag.emplace_back(block_size, block_size);
        bf.Ts_in_diag.emplace_back(block_size, block_size);

        // by the way, init tmp_bundle
        bf.tmp_bundle.emplace_back(bf.cnt_r, block_size);
        // tmp_bundle_2 for lspace multiply.
        bf.tmp_bundle_2.emplace_back(bf.cnt_r, block_size * block_size);
    }

    auto c_expr = [](double l, double m){
        return std::sqrt(((l + 1.0) * (l + 1.0) - m * m) / ((2.0 * l + 1.0) * (2.0 * l + 3.0)));
    };

    // update l-space matrices in core. (L, T, B, B_inv, A, A_inv)
    int l = 0, m = 0;
    for(size_t i = 0; i < 2 * l_num - 1; i++){
        m = get_m_from_mm(i);
        for(Eigen::Index j = 0; j < l_num - std::abs(m) - 1; j++){
            l = std::abs(m) + j;
            bf.Ls[i](j, j + 1) = c_expr(l, m);
            bf.Ls[i](j + 1, j) = bf.Ls[i](j, j + 1);
            bf.Ts[i](j, j + 1) = c_expr(l, m) * (l + 1);
            bf.Ts[i](j + 1, j) = -bf.Ts[i](j, j + 1);
        }
    }

    // get the eigenvalues(in diag matrix) and eigenvectors(in matrix)
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(l_num);
    for(size_t i = 0; i < 2 * l_num - 1; i++){
        //int block_size = bf.Ls[i].rows();
        solver.compute(bf.Ls[i]);
        bf.Bs[i] = solver.eigenvectors();
        bf.Ls_in_diag[i] = solver.eigenvalues().asDiagonal();
        
        solver.compute(bf.Ts[i] * 1i);
        bf.As[i] = solver.eigenvectors();
        bf.Ts_in_diag[i] = solver.eigenvalues().asDiagonal();

        bf.AT_B[i].noalias() = bf.As[i].adjoint() * bf.Bs[i];
        bf.BT_A[i].noalias() = bf.Bs[i].adjoint() * bf.As[i];
    }

    return bf;
}


inline void update_runtime_sh_polar_laser(
    TDSEBufferPolarLaserSH&     bf,
    double                      delta_t
)
{
    using namespace std::complex_literals;
    assert(delta_t > 0.0);

    std::vector<WaveData> vl_appendix(bf.l_num, create_empty_wave(bf.cnt_r));

    // update Hat_rmats_neg and Hat_rmats_pos
    // init vl_appendix and po_data_l(Vl)
    for(int l = 0; l < bf.l_num; l++){
        double ll = static_cast<double>(l);
        auto apdix_func = CREATE_1D_VFUNC_C((ll * (ll + 1) / 2.0) * INV(POW2(XS)), ll);
        update_wave_by_vectorized_func(vl_appendix[l], bf.core.rlinspace, apdix_func);
        bf.po_data_l[l] = bf.core.po_data + vl_appendix[l];
    }

    // init Hat_rmats_neg & Hat_rmats_pos with numerical boost.
    for(int l = 0; l < bf.l_num; l++){
        if (l == 0) {
            bf.Hat_rmats_neg[l] = bf.core.M2_boosted - (bf.core.D2_boosted +
                bf.core.M2_boosted * bf.po_data_l[l].asDiagonal()) * (0.5i * delta_t);
            bf.Hat_rmats_pos[l] = bf.core.M2_boosted + (bf.core.D2_boosted +
                bf.core.M2_boosted * bf.po_data_l[l].asDiagonal()) * (0.5i * delta_t);
        } else {
            bf.Hat_rmats_neg[l] = bf.core.M2 - (bf.core.D2 + bf.core.M2 * bf.po_data_l[l].asDiagonal()) * (0.5i * delta_t);
            bf.Hat_rmats_pos[l] = bf.core.M2 + (bf.core.D2 + bf.core.M2 * bf.po_data_l[l].asDiagonal()) * (0.5i * delta_t);
        }
        bf.Hat_rmats_neg[l].makeCompressed();
        bf.Hat_rmats_pos[l].makeCompressed();
    }
}


inline SpMat create_M1_for_sh(const NumericalGrid1D& grid)
{
	int cnt = grid.getTotalCount();
	double scaler = 1.0 / 6.0;
    double y = std::sqrt(3.0) - 2.0;

	SpMat matrix(cnt, cnt);
    matrix.reserve(Eigen::VectorXi::Constant(cnt, 3));

    for(int i = 0; i < cnt; i++) {
        if(i - 1 >= 0) matrix.insert(i, i - 1) = 1.0 * scaler;
        if(i == 0 || i == (cnt - 1)) matrix.insert(i, i) = (4.0 + y) * scaler;
        else matrix.insert(i, i) = 4.0 * scaler;
        if(i + 1 < cnt) matrix.insert(i, i + 1) = 1.0 * scaler;
    }
    matrix.makeCompressed();

	return matrix;
}


inline SpMat create_D1_for_sh(const NumericalGrid1D& grid)
{
	int cnt = grid.getTotalCount();
	double scaler = 1.0 / (2.0 * grid.getDelta(0));
    double y = std::sqrt(3.0) - 2.0;

	SpMat matrix(cnt, cnt);
    matrix.reserve(Eigen::VectorXi::Constant(cnt, 3));

    for(int i = 0; i < cnt; i++) {
        if(i - 1 >= 0) matrix.insert(i, i - 1) = -1.0 * scaler;
        if(i == 0) matrix.insert(i, i) = y * scaler;
        else if(i == cnt - 1) matrix.insert(i, i) = -y * scaler;
        else matrix.insert(i, i) = 0;
        if(i + 1 < cnt) matrix.insert(i, i + 1) = 1.0 * scaler;
    }
    matrix.makeCompressed();

	return matrix;
}

inline void copy_bundle_sh(
    Eigen::Ref<Eigen::MatrixXcd>    output_bundle,
    Eigen::Ref<Eigen::MatrixXcd>    input_bundle
)
{
    assert(input_bundle.cols() == output_bundle.cols()
        && input_bundle.rows() == output_bundle.rows());

    int cnt_l = input_bundle.cols();
    for(Eigen::Index i = 0; i < cnt_l; i++){
        #pragma omp task firstprivate(i, output_bundle, input_bundle)
        output_bundle.col(i) = input_bundle.col(i);
    }
}


inline void apply_pure_lspace_mat_sh(
    Eigen::Ref<Eigen::MatrixXcd>    lmat,
    Eigen::Ref<Eigen::MatrixXcd>    input_bundle,
    Eigen::Ref<Eigen::MatrixXcd>    output_bundle,
    Eigen::Ref<Eigen::MatrixXcd>    bundle_bf,
    bool                            adjoint_flag = false
    )
{
    assert(lmat.cols() == lmat.rows() 
        && lmat.rows() == input_bundle.cols()
        && input_bundle.cols() == output_bundle.cols()
        && input_bundle.rows() == output_bundle.rows());

    int cnt_l = lmat.cols();
    int cnt_r = input_bundle.rows();
    output_bundle.setZero();
    if (adjoint_flag == true) {
        for(Eigen::Index i = 0; i < cnt_l; i++){
            #pragma omp task //firstprivate(i, cnt_l, lmat, input_bundle, output_bundle, bundle_bf)
            // {
                // bundle_bf.middleCols(i * cnt_l, cnt_l) = input_bundle.array().rowwise() * lmat.col(i).adjoint().array();
                // output_bundle.col(i) = bundle_bf.middleCols(i * cnt_l, cnt_l).rowwise().sum();
            // }
            for(Eigen::Index j = 0; j < cnt_l; j++){
               output_bundle.col(i) += input_bundle.col(j) * std::conj(lmat(j, i));
            }
        }
    } else {
        for(Eigen::Index i = 0; i < cnt_l; i++){
            #pragma omp task //firstprivate(i, cnt_l, lmat, input_bundle, output_bundle, bundle_bf)
            // {
            //     bundle_bf.middleCols(i * cnt_l, cnt_l) = input_bundle.array().rowwise() * lmat.row(i).array();
            //     output_bundle.col(i) = bundle_bf.middleCols(i * cnt_l, cnt_l).rowwise().sum();                
            // }
            for(Eigen::Index j = 0; j < cnt_l; j++){
                output_bundle.col(i) += input_bundle.col(j) * lmat(i, j);
            }
        }
    }
}


// inline void apply_pure_lspace_mat_sh(
//     Eigen::Ref<Eigen::MatrixXcd>    lmat,
//     Eigen::Ref<Eigen::MatrixXcd>    input_bundle,
//     Eigen::Ref<Eigen::MatrixXcd>    output_bundle,
//     Eigen::Ref<Eigen::MatrixXcd>    bundle_bf,
//     bool                            adjoint_flag = false
//     )
// {
//     assert(lmat.cols() == lmat.rows() 
//         && lmat.rows() == input_bundle.cols()
//         && input_bundle.cols() == output_bundle.cols()
//         && input_bundle.rows() == output_bundle.rows());

//     int cnt_l = lmat.cols();
//     int cnt_r = input_bundle.rows();
//     output_bundle.setZero();
//     if (adjoint_flag == true) {
//         #pragma omp parallel for
//         for(Eigen::Index i = 0; i < cnt_l; i++){
//             // #pragma omp task //firstprivate(i, cnt_l, lmat, input_bundle, output_bundle, bundle_bf)
//             for(Eigen::Index j = 0; j < cnt_l; j++){
//                output_bundle.col(i) += input_bundle.col(j) * std::conj(lmat(j, i));
//             }
//         }
//     } else {
//         #pragma omp parallel for
//         for(Eigen::Index i = 0; i < cnt_l; i++){
//             // #pragma omp task //firstprivate(i, cnt_l, lmat, input_bundle, output_bundle, bundle_bf)
//             for(Eigen::Index j = 0; j < cnt_l; j++){
//                 output_bundle.col(i) += input_bundle.col(j) * lmat(i, j);
//             }
//         }
//     }
// }


inline void diag_mat_elimination(
    Eigen::Ref<SpMat>       A,
    Eigen::Ref<WaveData>    X,
    Eigen::Ref<WaveData>    B
)
{
    assert(A.nonZeros() == B.rows() && X.rows() == A.rows()
        && A.rows() == A.cols() && A.rows() == B.rows());
    
    int cnt = A.cols();
    const std::complex<double> *value_ptr = A.valuePtr();

    for(int i = 0; i < cnt; i++){
        X(i) = B(i) / *(value_ptr + i);
    }
}


constexpr int HMIX_FLAG = 0;
constexpr int HANG_FLAG = 1;

/*
inline void apply_Hmix_ang(
    TDSEBufferPolarLaserSH&         bf,
    Eigen::Ref<Eigen::MatrixXcd>    crt_bundle,
    const int                       mm,
    const double                    delta_t,
    const double                    At,
    const int                       flag    //HMIX_FLAG or HANG_FLAG
)
{
    using namespace std::complex_literals;

    //TIME_TEST_START(apply_Hmix_ang)

    int m = get_m_from_mm(mm);
    int bundle_size = bf.l_num - std::abs(m);
    int head_ptr = get_index_from_lm(std::abs(m), m, bf.l_num);

    // #pragma omp task \
    //    firstprivate(bf, crt_bundle, mm, delta_t, At, flag) depend(out: bf.tmp_bundle)

    // mul A_inv | B_inv
    if (flag == HMIX_FLAG) {
        apply_pure_lspace_mat_sh(bf.Bs[mm], crt_bundle, bf.tmp_bundle[mm], true);   // contains child tasks
    } else {
        apply_pure_lspace_mat_sh(bf.As[mm], crt_bundle, bf.tmp_bundle[mm], true);
    }
    #pragma omp taskwait
    crt_bundle = bf.tmp_bundle[mm];
    #pragma omp taskwait
    
    for(Eigen::Index j = 0; j < bundle_size; j++){
        #pragma omp task
        {
            //TIME_TEST_START(mix_ang_tasks)
            // apply Hang0_neg | Hmix0_neg
            if (flag == HANG_FLAG)
                bf.diag_rmats[head_ptr + j] = 1.0i * bf.core.Ir - 0.25 * delta_t * At * bf.Ts_in_diag[mm](j, j) * bf.core.Rr;
            else if (flag == HMIX_FLAG)
                bf.diag_rmats[head_ptr + j] = bf.core.M1 - 0.25 * delta_t * At * bf.Ls_in_diag[mm](j, j) * bf.core.D1;
            
            bf.tmp_bundle[mm].col(j).noalias() = bf.diag_rmats[head_ptr + j] * crt_bundle.col(j);
            crt_bundle.col(j) = bf.tmp_bundle[mm].col(j);

            // apply Hang0_pos_inv | Hmix0_pos_inv
            if (flag == HANG_FLAG) {
                bf.diag_rmats[head_ptr + j] = 1.0i * bf.core.Ir + 0.25 * delta_t * At * bf.Ts_in_diag[mm](j, j) * bf.core.Rr;
                diag_mat_elimination(bf.diag_rmats[head_ptr + j], bf.tmp_bundle[mm].col(j), crt_bundle.col(j));
            } 
            else if(flag == HMIX_FLAG) {
                bf.diag_rmats[head_ptr + j] = bf.core.M1 + 0.25 * delta_t * At * bf.Ls_in_diag[mm](j, j) * bf.core.D1;
                tridiagonal_mat_elimination_optimized(bf.diag_rmats[head_ptr + j], bf.tmp_bundle[mm].col(j),
                    crt_bundle.col(j), bf.tri_elim_bfs_A[head_ptr + j], bf.tri_elim_bfs_B[head_ptr + j]);
            }
            crt_bundle.col(j) = bf.tmp_bundle[mm].col(j);

            //TIME_TEST_END(mix_ang_tasks)
        }
    }
    #pragma omp taskwait

    // mul A | B
    if (flag == HMIX_FLAG) {
        apply_pure_lspace_mat_sh(bf.Bs[mm], crt_bundle, bf.tmp_bundle[mm], false);   // contains child tasks
    } else {
        apply_pure_lspace_mat_sh(bf.As[mm], crt_bundle, bf.tmp_bundle[mm], false);   // contains child tasks
    }
    #pragma omp taskwait
    crt_bundle = bf.tmp_bundle[mm];
    #pragma omp taskwait

    //TIME_TEST_END(apply_Hmix_ang)
}
*/


inline void tdse_sh_polar_laser_once(
    TDSEBufferPolarLaserSH&     bf,
    Eigen::Ref<WaveDataSH>      crt_shwave,
    double                      At,     // A(t)
    double                      delta_t,
    int                         steps_num = 1
    //const std::vector<int>&     mm_selected_map
)
{
    using namespace std::complex_literals;
    assert(delta_t > 0 && Ats.size() >= steps_num);
    //assert(mm_selected_map.size() == 2 * bf.l_num - 1);
    assert(crt_shwave.rows() == bf.cnt_r && crt_shwave.cols() == bf.l_num * bf.l_num);
    
    //TIME_TEST_START(tdse_sh_pl)

    // for each bundle with a certain m (or m')
    #pragma omp parallel
    #pragma omp single
    for(int mm = 0; mm < 2 * bf.l_num - 1; mm++) {
        if(mm == 0) {
            #pragma omp task firstprivate(mm) untied
            {
                TIME_TEST_START(mm_task)
                int m = get_m_from_mm(mm);
                int bundle_size = bf.l_num - std::abs(m);
                int head_ptr = get_index_from_lm(std::abs(m), m, bf.l_num);
                Eigen::Ref<Eigen::MatrixXcd> crt_bundle = crt_shwave.middleCols(head_ptr, bundle_size);

                // -------------------------------------------------------- Hang start
                // mul A_T
                TIME_TEST_START(left_part)

                TIME_TEST_START(pure_lspace)
                apply_pure_lspace_mat_sh(bf.As[mm], crt_bundle, bf.tmp_bundle[mm], bf.tmp_bundle_2[mm], true);
                #pragma omp taskwait
                TIME_TEST_END(pure_lspace)
                
                TIME_TEST_START(apply_Hang)
                // apply Hang0_neg | Hmix0_neg
                for(Eigen::Index j = 0; j < bundle_size; j++){
                    #pragma omp task
                    {
                        //bf.tmp_bundle_2[mm].middleCols(j * bundle_size, bundle_size) = crt_bundle.array().rowwise() * bf.As[mm].col(j).adjoint().array();
                        //bf.tmp_bundle[mm].col(j) = bf.tmp_bundle_2[mm].middleCols(j * bundle_size, bundle_size).rowwise().sum();

                        bf.diag_rmats[head_ptr + j] = 1.0i * bf.core.Ir - 0.25 * delta_t * At * bf.Ts_in_diag[mm](j, j) * bf.core.Rr;
                        crt_bundle.col(j).noalias() = bf.diag_rmats[head_ptr + j] * bf.tmp_bundle[mm].col(j);

                        bf.diag_rmats[head_ptr + j] = 1.0i * bf.core.Ir + 0.25 * delta_t * At * bf.Ts_in_diag[mm](j, j) * bf.core.Rr;
                        diag_mat_elimination(bf.diag_rmats[head_ptr + j], bf.tmp_bundle[mm].col(j), crt_bundle.col(j));

                        crt_bundle.col(j) = bf.tmp_bundle[mm].col(j);
                    }
                }
                #pragma omp taskwait
                TIME_TEST_END(apply_Hang)

                // // -------------------------------------------------------- Hang end
                apply_pure_lspace_mat_sh(bf.BT_A[mm], crt_bundle, bf.tmp_bundle[mm], bf.tmp_bundle_2[mm], false);
                #pragma omp taskwait
                // // -------------------------------------------------------- Hmix start
                
                TIME_TEST_START(apply_Hmix)
                // apply Hmix0_neg
                for(Eigen::Index j = 0; j < bundle_size; j++){
                    #pragma omp task
                    {
                        bf.diag_rmats[head_ptr + j] = bf.core.M1 - 0.25 * delta_t * At * bf.Ls_in_diag[mm](j, j) * bf.core.D1;
                        crt_bundle.col(j).noalias() = bf.diag_rmats[head_ptr + j] * bf.tmp_bundle[mm].col(j);

                        bf.diag_rmats[head_ptr + j] = bf.core.M1 + 0.25 * delta_t * At * bf.Ls_in_diag[mm](j, j) * bf.core.D1;
                        tridiagonal_mat_elimination_optimized(bf.diag_rmats[head_ptr + j], bf.tmp_bundle[mm].col(j),
                            crt_bundle.col(j), bf.tri_elim_bfs_A[head_ptr + j], bf.tri_elim_bfs_B[head_ptr + j]);

                        crt_bundle.col(j) = bf.tmp_bundle[mm].col(j);
                    }
                }
                #pragma omp taskwait
                TIME_TEST_END(apply_Hmix)

                // mul B
                apply_pure_lspace_mat_sh(bf.Bs[mm], crt_bundle, bf.tmp_bundle[mm], bf.tmp_bundle_2[mm], false);   // contains child tasks
                #pragma omp taskwait
                // crt_bundle = bf.tmp_bundle[mm];
                copy_bundle_sh(crt_bundle, bf.tmp_bundle[mm]);
                #pragma omp taskwait
                // -------------------------------------------------------- Hmix end
                TIME_TEST_END(left_part)


                // -------------------------------------------------------- Hat
                // apply H_at, using crank-nicolson implement.
                TIME_TEST_START(apply_Hat)
                for(int j = 0; j < bundle_size; j++){
                    #pragma omp task firstprivate(j)
                    {
                        TIME_TEST_START(task_Hat)
                        int id = head_ptr + j;
                        int l = bf.lmap[id];
                        bf.tmp_bundle[mm].col(j).noalias() = bf.Hat_rmats_neg[l] * crt_shwave.col(id);
                        tridiagonal_mat_elimination_optimized(bf.Hat_rmats_pos[l], crt_shwave.col(id),
                            bf.tmp_bundle[mm].col(j), bf.tri_elim_bfs_A[id], bf.tri_elim_bfs_B[id]);
                        TIME_TEST_END(task_Hat)
                    }
                }
                #pragma omp taskwait
                TIME_TEST_END(apply_Hat)
                // -------------------------------------------------------- Hat
                
                
                // -------------------------------------------------------- Hmix start
                // mul B_T
                apply_pure_lspace_mat_sh(bf.Bs[mm], crt_bundle, bf.tmp_bundle[mm], bf.tmp_bundle_2[mm], true);
                #pragma omp taskwait
                
                // apply Hmix0_neg
                for(Eigen::Index j = 0; j < bundle_size; j++){
                    #pragma omp task
                    {   
                        bf.diag_rmats[head_ptr + j] = bf.core.M1 - 0.25 * delta_t * At * bf.Ls_in_diag[mm](j, j) * bf.core.D1;
                        crt_bundle.col(j).noalias() = bf.diag_rmats[head_ptr + j] * bf.tmp_bundle[mm].col(j);

                        bf.diag_rmats[head_ptr + j] = bf.core.M1 + 0.25 * delta_t * At * bf.Ls_in_diag[mm](j, j) * bf.core.D1;
                        tridiagonal_mat_elimination_optimized(bf.diag_rmats[head_ptr + j], bf.tmp_bundle[mm].col(j),
                            crt_bundle.col(j), bf.tri_elim_bfs_A[head_ptr + j], bf.tri_elim_bfs_B[head_ptr + j]);

                        crt_bundle.col(j) = bf.tmp_bundle[mm].col(j);
                    }
                }
                #pragma omp taskwait

                // -------------------------------------------------------- Hmix end
                apply_pure_lspace_mat_sh(bf.AT_B[mm], crt_bundle, bf.tmp_bundle[mm], bf.tmp_bundle_2[mm], false);
                #pragma omp taskwait
                // -------------------------------------------------------- Hang start
                
                // apply Hang0_neg | Hmix0_neg
                for(Eigen::Index j = 0; j < bundle_size; j++){
                    #pragma omp task
                    {
                        bf.diag_rmats[head_ptr + j] = 1.0i * bf.core.Ir - 0.25 * delta_t * At * bf.Ts_in_diag[mm](j, j) * bf.core.Rr;
                        crt_bundle.col(j).noalias() = bf.diag_rmats[head_ptr + j] * bf.tmp_bundle[mm].col(j);

                        bf.diag_rmats[head_ptr + j] = 1.0i * bf.core.Ir + 0.25 * delta_t * At * bf.Ts_in_diag[mm](j, j) * bf.core.Rr;
                        diag_mat_elimination(bf.diag_rmats[head_ptr + j], bf.tmp_bundle[mm].col(j), crt_bundle.col(j));

                        crt_bundle.col(j) = bf.tmp_bundle[mm].col(j);
                    }
                }
                #pragma omp taskwait

                // mul A
                apply_pure_lspace_mat_sh(bf.As[mm], crt_bundle, bf.tmp_bundle[mm], bf.tmp_bundle_2[mm], false);   // contains child tasks
                #pragma omp taskwait
                copy_bundle_sh(crt_bundle, bf.tmp_bundle[mm]);
                #pragma omp taskwait
                // -------------------------------------------------------- Hang end

                TIME_TEST_END(mm_task)
            }
        }
    }
   //TIME_TEST_END(tdse_sh_pl)
}


/*
inline void tdse_sh_polar_laser_once(
    TDSEBufferPolarLaserSH&     bf,
    Eigen::Ref<WaveDataSH>      crt_shwave,
    Eigen::Ref<WaveData>        Ats,     // A(t)
    double                      delta_t,
    int                         steps_num = 1
    //const std::vector<int>&     mm_selected_map
)
{
    using namespace std::complex_literals;
    assert(delta_t > 0 && Ats.size() >= steps_num);
    //assert(mm_selected_map.size() == 2 * bf.l_num - 1);
    assert(crt_shwave.rows() == bf.cnt_r && crt_shwave.cols() == bf.l_num * bf.l_num);
    
    //TIME_TEST_START(tdse_sh_pl)

    // for each bundle with a certain m (or m')
    // #pragma omp parallel
    for(int mm = 0; mm < 1; mm++)
    {
        TIME_TEST_START(mm_task)
        int m = get_m_from_mm(mm);
        int bundle_size = bf.l_num - std::abs(m);
        int head_ptr = get_index_from_lm(std::abs(m), m, bf.l_num);
        Eigen::Ref<Eigen::MatrixXcd> crt_bundle = crt_shwave.middleCols(head_ptr, bundle_size);

        for(int step = 0; step < steps_num; step++) 
        {
            if(step % 100 == 0) std::cout << "step: " << step << std::endl;
            double At = Ats(step).real();
            // -------------------------------------------------------- Hang start
            // mul A_T
            TIME_TEST_START(left_part)

            TIME_TEST_START(pure_lspace)
            apply_pure_lspace_mat_sh(bf.As[mm], crt_bundle, bf.tmp_bundle[mm], bf.tmp_bundle_2[mm], true);
            TIME_TEST_END(pure_lspace)
            
            TIME_TEST_START(apply_Hang)
            // apply Hang0_neg | Hmix0_neg
            #pragma omp parallel for
            for(Eigen::Index j = 0; j < bundle_size; j++){
                // #pragma omp task
                {
                    bf.diag_rmats[head_ptr + j] = 1.0i * bf.core.Ir - 0.25 * delta_t * At * bf.Ts_in_diag[mm](j, j) * bf.core.Rr;
                    crt_bundle.col(j).noalias() = bf.diag_rmats[head_ptr + j] * bf.tmp_bundle[mm].col(j);

                    bf.diag_rmats[head_ptr + j] = 1.0i * bf.core.Ir + 0.25 * delta_t * At * bf.Ts_in_diag[mm](j, j) * bf.core.Rr;
                    diag_mat_elimination(bf.diag_rmats[head_ptr + j], bf.tmp_bundle[mm].col(j), crt_bundle.col(j));

                    crt_bundle.col(j) = bf.tmp_bundle[mm].col(j);
                }
            }
            // #pragma omp taskwait
            TIME_TEST_END(apply_Hang)

            // // -------------------------------------------------------- Hang end
            apply_pure_lspace_mat_sh(bf.BT_A[mm], crt_bundle, bf.tmp_bundle[mm], bf.tmp_bundle_2[mm], false);
            // #pragma omp taskwait
            // // -------------------------------------------------------- Hmix start
            
            TIME_TEST_START(apply_Hmix)
            // apply Hmix0_neg
            #pragma omp parallel for
            for(Eigen::Index j = 0; j < bundle_size; j++){
                // #pragma omp task
                {
                    bf.diag_rmats[head_ptr + j] = bf.core.M1 - 0.25 * delta_t * At * bf.Ls_in_diag[mm](j, j) * bf.core.D1;
                    crt_bundle.col(j).noalias() = bf.diag_rmats[head_ptr + j] * bf.tmp_bundle[mm].col(j);

                    bf.diag_rmats[head_ptr + j] = bf.core.M1 + 0.25 * delta_t * At * bf.Ls_in_diag[mm](j, j) * bf.core.D1;
                    tridiagonal_mat_elimination_optimized(bf.diag_rmats[head_ptr + j], bf.tmp_bundle[mm].col(j),
                        crt_bundle.col(j), bf.tri_elim_bfs_A[head_ptr + j], bf.tri_elim_bfs_B[head_ptr + j]);

                    crt_bundle.col(j) = bf.tmp_bundle[mm].col(j);
                }
            }
            // #pragma omp taskwait
            TIME_TEST_END(apply_Hmix)

            // mul B
            apply_pure_lspace_mat_sh(bf.Bs[mm], crt_bundle, bf.tmp_bundle[mm], bf.tmp_bundle_2[mm], false);   // contains child tasks
            // #pragma omp taskwait
            // copy
            #pragma omp parallel for
            for(Eigen::Index i = 0; i < bundle_size; i++){
                crt_bundle.col(i) = bf.tmp_bundle[mm].col(i);
            }
            // #pragma omp taskwait
            // -------------------------------------------------------- Hmix end
            TIME_TEST_END(left_part)

            // -------------------------------------------------------- Hat
            // apply H_at, using crank-nicolson implement.
            TIME_TEST_START(apply_Hat)
            #pragma omp parallel for
            for(int j = 0; j < bundle_size; j++){
                //#pragma omp task firstprivate(j)
                {
                    TIME_TEST_START(task_Hat)
                    int id = head_ptr + j;
                    int l = bf.lmap[id];
                    bf.tmp_bundle[mm].col(j).noalias() = bf.Hat_rmats_neg[l] * crt_shwave.col(id);
                    tridiagonal_mat_elimination_optimized(bf.Hat_rmats_pos[l], crt_shwave.col(id),
                        bf.tmp_bundle[mm].col(j), bf.tri_elim_bfs_A[id], bf.tri_elim_bfs_B[id]);
                    TIME_TEST_END(task_Hat)
                }
            }
            //#pragma omp taskwait
            TIME_TEST_END(apply_Hat)
            // -------------------------------------------------------- Hat
            
            // -------------------------------------------------------- Hmix start
            // mul B_T
            apply_pure_lspace_mat_sh(bf.Bs[mm], crt_bundle, bf.tmp_bundle[mm], bf.tmp_bundle_2[mm], true);
            //#pragma omp taskwait
            
            // apply Hmix0_neg
            #pragma omp parallel for
            for(Eigen::Index j = 0; j < bundle_size; j++){
                //#pragma omp task
                {   
                    bf.diag_rmats[head_ptr + j] = bf.core.M1 - 0.25 * delta_t * At * bf.Ls_in_diag[mm](j, j) * bf.core.D1;
                    crt_bundle.col(j).noalias() = bf.diag_rmats[head_ptr + j] * bf.tmp_bundle[mm].col(j);

                    bf.diag_rmats[head_ptr + j] = bf.core.M1 + 0.25 * delta_t * At * bf.Ls_in_diag[mm](j, j) * bf.core.D1;
                    tridiagonal_mat_elimination_optimized(bf.diag_rmats[head_ptr + j], bf.tmp_bundle[mm].col(j),
                        crt_bundle.col(j), bf.tri_elim_bfs_A[head_ptr + j], bf.tri_elim_bfs_B[head_ptr + j]);

                    crt_bundle.col(j) = bf.tmp_bundle[mm].col(j);
                }
            }
            //#pragma omp taskwait

            // -------------------------------------------------------- Hmix end
            apply_pure_lspace_mat_sh(bf.AT_B[mm], crt_bundle, bf.tmp_bundle[mm], bf.tmp_bundle_2[mm], false);
            //#pragma omp taskwait
            // -------------------------------------------------------- Hang start
            
            // apply Hang0_neg | Hmix0_neg
            #pragma omp parallel for
            for(Eigen::Index j = 0; j < bundle_size; j++){
                //#pragma omp task
                {
                    bf.diag_rmats[head_ptr + j] = 1.0i * bf.core.Ir - 0.25 * delta_t * At * bf.Ts_in_diag[mm](j, j) * bf.core.Rr;
                    crt_bundle.col(j).noalias() = bf.diag_rmats[head_ptr + j] * bf.tmp_bundle[mm].col(j);

                    bf.diag_rmats[head_ptr + j] = 1.0i * bf.core.Ir + 0.25 * delta_t * At * bf.Ts_in_diag[mm](j, j) * bf.core.Rr;
                    diag_mat_elimination(bf.diag_rmats[head_ptr + j], bf.tmp_bundle[mm].col(j), crt_bundle.col(j));

                    crt_bundle.col(j) = bf.tmp_bundle[mm].col(j);
                }
            }
            //#pragma omp taskwait

            // mul A
            apply_pure_lspace_mat_sh(bf.As[mm], crt_bundle, bf.tmp_bundle[mm], bf.tmp_bundle_2[mm], false);   // contains child tasks
            //#pragma omp taskwait
            // copy
            #pragma omp parallel for
            for(Eigen::Index i = 0; i < bundle_size; i++){
                crt_bundle.col(i) = bf.tmp_bundle[mm].col(i);
            }
            //#pragma omp taskwait
            // -------------------------------------------------------- Hang end

        }
        TIME_TEST_END(mm_task)
    }
   //TIME_TEST_END(tdse_sh_pl)
}
*/


struct ExpectedValueBufferSHPL
{
    std::vector<WaveDataSH> intermediate_results;

    std::vector<SpMat>  Hat_part_l;   // size: l_num

    std::vector<TrimatElimBuffer>    A_add_list_1;     // size: l_num * l_num
    std::vector<TrimatElimBuffer>    B_add_list_1;
    std::vector<TrimatElimBuffer>    A_add_list_2;     
    std::vector<TrimatElimBuffer>    B_add_list_2;

    WaveDataSH&      Hat_result;
    WaveDataSH&      Hat_result_tmp;

    WaveDataSH&      Hang_result;
    WaveDataSH&      Hang_result_tmp_1;
    WaveDataSH&      Hang_result_tmp_2;

    WaveDataSH&      Hmix_result;
    WaveDataSH&      Hmix_result_tmp_1;
    WaveDataSH&      Hmix_result_tmp_2;

    WaveDataSH&      scaler_result;
    WaveDataSH&      tmp_state;

    ExpectedValueBufferSHPL(int cnt_r, int l_num)
        : 
        intermediate_results(10, create_empty_wave_sh(cnt_r, l_num)),
        Hat_part_l(l_num, SpMat(cnt_r, cnt_r)),

        A_add_list_1(l_num * l_num, TrimatElimBuffer(cnt_r, 0)), B_add_list_1(l_num * l_num, TrimatElimBuffer(cnt_r, 0)),
        A_add_list_2(l_num * l_num, TrimatElimBuffer(cnt_r, 0)), B_add_list_2(l_num * l_num, TrimatElimBuffer(cnt_r, 0)),

        // just for Readability.
        Hat_result(intermediate_results[0]), Hat_result_tmp(intermediate_results[1]),
        Hang_result(intermediate_results[2]), Hang_result_tmp_1(intermediate_results[3]),
        Hang_result_tmp_2(intermediate_results[4]), Hmix_result(intermediate_results[5]),
        Hmix_result_tmp_1(intermediate_results[6]), Hmix_result_tmp_2(intermediate_results[7]),
        scaler_result(intermediate_results[8]), tmp_state(intermediate_results[9])

        {}
    
    void clear() {
        for(auto &wave : intermediate_results){
            wave.setZero();
        }
    }
};


inline ExpectedValueBufferSHPL init_expected_value_solver_sh_pl(const TDSEBufferPolarLaserSH& bf)
{
    ExpectedValueBufferSHPL exbf(bf.cnt_r, bf.l_num);

    for(int l = 0; l < bf.l_num; l++){
        if (l == 0) {
            exbf.Hat_part_l[l] = bf.core.D2_boosted + bf.core.M2_boosted * bf.po_data_l[l].asDiagonal();
        } else {
            exbf.Hat_part_l[l] = bf.core.D2 + bf.core.M2 * bf.po_data_l[l].asDiagonal();
        }
    }
    return exbf;
}


inline double get_energy_sh_pl(
    TDSEBufferPolarLaserSH&     bf,
    ExpectedValueBufferSHPL&    exbf,
    Eigen::Ref<WaveDataSH>      shwave,
    double                      At,
    double                      delta_t
)
{
    using namespace std::complex_literals;
    assert(shwave.rows() == bf.cnt_r
        && shwave.cols() == bf.l_num * bf.l_num && delta_t > 0);

    TIME_TEST_START(get_energy)

    // clear ex buffer first (important)
    exbf.clear();

#pragma omp parallel
#pragma omp single
{
    // apply Hat
    for(int i = 0; i < bf.l_num * bf.l_num; i++){
        #pragma omp task
        {
            int l = bf.lmap[i];
            exbf.Hat_result_tmp.col(i).noalias() = exbf.Hat_part_l[l] * shwave.col(i);
            if (l == 0) {
                tridiagonal_mat_elimination_optimized(bf.core.M2_boosted, exbf.Hat_result.col(i),
                    exbf.Hat_result_tmp.col(i), exbf.A_add_list_1[i], exbf.B_add_list_1[i]);
            } else {
                tridiagonal_mat_elimination_optimized(bf.core.M2, exbf.Hat_result.col(i),
                    exbf.Hat_result_tmp.col(i), exbf.A_add_list_1[i], exbf.B_add_list_1[i]);
            }
        }
    }

    for(int mm = 0; mm < 2 * bf.l_num - 1; mm++)
    {
        int m = get_m_from_mm(mm);
        int bundle_size = bf.l_num - std::abs(m);
        int head_ptr = get_index_from_lm(std::abs(m), m, bf.l_num);

        // apply Hmix
        for(int j = 0; j < bundle_size; j++){
            #pragma omp task
            {
                int id = head_ptr + j;
                if (j - 1 >= 0) {
                    exbf.Hmix_result_tmp_1.col(id).noalias() = bf.core.D1 * shwave.col(id - 1);
                    exbf.Hmix_result_tmp_1.col(id) *= (-1i * At * bf.Ls[mm](j, j - 1));
                    tridiagonal_mat_elimination_optimized(bf.core.M1, exbf.Hmix_result_tmp_2.col(id),
                        exbf.Hmix_result_tmp_1.col(id), exbf.A_add_list_2[id], exbf.B_add_list_2[id]);
                    exbf.Hmix_result.col(id) += exbf.Hmix_result_tmp_2.col(id);
                }
                // the same. reuse tmp_1, tmp_2
                if (j + 1 < bundle_size) {
                    exbf.Hmix_result_tmp_1.col(id).noalias() = bf.core.D1 * shwave.col(id + 1);
                    exbf.Hmix_result_tmp_1.col(id) *= (-1i * At * bf.Ls[mm](j, j + 1));
                    tridiagonal_mat_elimination_optimized(bf.core.M1, exbf.Hmix_result_tmp_2.col(id),
                        exbf.Hmix_result_tmp_1.col(id), exbf.A_add_list_2[id], exbf.B_add_list_2[id]);
                    exbf.Hmix_result.col(id) += exbf.Hmix_result_tmp_2.col(id);
                }
            }
        }

        // apply Hang (reuse A_add and B_add)
        for(int j = 0; j < bundle_size; j++){
            #pragma omp task
            {
                int id = head_ptr + j;
                if (j - 1 >= 0) {
                    exbf.Hang_result_tmp_1.col(id).noalias() = bf.core.Rr * shwave.col(id - 1) * (-1i * At * bf.Ts[mm](j, j - 1));
                    exbf.Hang_result.col(id) += exbf.Hang_result_tmp_1.col(id);
                }
                if (j + 1 < bundle_size) {
                    exbf.Hang_result_tmp_1.col(id).noalias() = bf.core.Rr * shwave.col(id + 1) * (-1i * At * bf.Ts[mm](j, j + 1));
                    exbf.Hang_result.col(id) += exbf.Hang_result_tmp_1.col(id);
                }
            }
        }
    }

    // scaler part
    #pragma omp task
    exbf.scaler_result = (0.5 * At * At) * shwave;
}

    // add them together, them get the energy
    #pragma omp parallel for
    for(int i = 0; i < bf.l_num * bf.l_num; i++)
        exbf.tmp_state.col(i) = exbf.Hat_result.col(i) + exbf.Hang_result.col(i) + exbf.Hmix_result.col(i) + 2 * exbf.scaler_result.col(i);

    double energy = fast_inner_product_sh(shwave, exbf.tmp_state).real();

    TIME_TEST_END(get_energy)
    return energy;
}



}


#endif //__TDSE_SH_LASER_HPP__