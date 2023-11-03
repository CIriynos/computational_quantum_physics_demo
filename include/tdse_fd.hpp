#ifndef __TDSE_FD_HPP__
#define __TDSE_FD_HPP__

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "wave_function.hpp"
#include "numerical_grid.hpp"

namespace CQP {

constexpr int REFLECTING_BOUNDARY_COND = 0;
constexpr int PERIODIC_BOUNDARY_COND = 1;
constexpr int IMAG_TIME_PROPAGATION_COND = 2;

typedef Eigen::SparseMatrix<std::complex<double>, Eigen::RowMajor> SpMat;

template<unsigned N>
inline SpMat create_matrix_hamiltonian(const NumericalGrid<N>& grid, int cond);

inline void elimination_process_fd1d (
    Eigen::Ref<SpMat>                   A,
    Eigen::Ref<Eigen::VectorXcd>        X,
    Eigen::Ref<Eigen::VectorXcd>        B
);


struct TDSECoreFD1D
{
    int cnt;
    int cond;

    // 1. Never change in the whole question, only related to given grid.
    SpMat                   M;
    SpMat                   D;
    SpMat                   I;
    NumericalGrid1D         grid;
    LinSpaceData<1>         linspace;

    TDSECoreFD1D(const NumericalGrid<1>& grid, int cond)
        : cnt(grid.getTotalCount()), cond(cond),
        M(cnt, cnt), D(cnt, cnt), I(cnt, cnt),
        grid(grid), linspace(create_linspace(grid)) 
    {}
};


struct TDSERuntimeBufferFD1D
{
    int cnt;
    int cond;

    // 2. change when time-dependent, never changed when no time. (related to potiential, or l number, etc.)
    SpMat                   A_pos;
    SpMat                   A_neg;
    WaveData                po_data;
    SpMat                   infact_h;

    TDSERuntimeBufferFD1D(const NumericalGrid<1>& grid, int cond)
        : cnt(grid.getTotalCount()), cond(cond),
        A_pos(cnt, cnt), A_neg(cnt, cnt),
        po_data(create_empty_wave(cnt)),
        infact_h(cnt, cnt)
        {}
};


struct TDSEMainloopBufferFD1D 
{
    int cnt;
    int cond;

    WaveData            half_state;
    SpMat               A_tmp_1;
    //Eigen::VectorXcd    B_tmp_1;
    //SpMat               A_tmp_2;
    //Eigen::VectorXcd    B_tmp_2;

    // used in both conds.
    std::vector<std::complex<double>> A_add;
    std::vector<std::complex<double>> B_add;

    // used in periodical cond.
    Eigen::VectorXcd    xdata1;
    Eigen::VectorXcd    xdata2;
    Eigen::VectorXcd    udata;
    Eigen::VectorXcd    vdata;

    TDSEMainloopBufferFD1D(const NumericalGrid<1>& grid, int cond)
        : cnt(grid.getTotalCount()), cond(cond),
        half_state(create_empty_wave(grid)),
        A_tmp_1(cnt, cnt), //B_tmp_1(cnt),
        //A_tmp_2(cnt, cnt), B_tmp_2(cnt),
        A_add(cnt, 0), B_add(cnt, 0),
        // firstly assign empty vector.
        xdata1(0), xdata2(0), udata(0), vdata(0) 
        {
            if(cond == PERIODIC_BOUNDARY_COND){
                xdata1 = Eigen::VectorXcd::Zero(cnt);
                xdata2 = Eigen::VectorXcd::Zero(cnt);
                udata = Eigen::VectorXcd::Zero(cnt);
                vdata = Eigen::VectorXcd::Zero(cnt);
            }
        }
};


struct TDSEBufferFD1D
{
    int cnt;
    int cond;

    // 1. Never change in the whole question, only related to given grid.
    TDSECoreFD1D            core;

    // 2. change when time-dependent, never changed when no time. (related to potiential, or l number, etc.)
    TDSERuntimeBufferFD1D   rt;
    // SpMat                infact_hamiltonian;
    
    // 3. run-time vars
    // incluing half_state, and temporary vars in elimination.
    TDSEMainloopBufferFD1D  mlbf;

    // DO NOT CREATE TDSE_BUFFER DIRECTLY BY THIS METHOD!
    TDSEBufferFD1D(const NumericalGrid<1>& grid, int cond)
        : 
        cnt(grid.getTotalCount()), cond(cond),
        core(grid, cond),
        rt(grid, cond),
        // infact_hamiltonian(cnt, cnt),
        mlbf(grid, cond)
    {}
};


inline TDSEBufferFD1D init_crank_nicolson_1d(const NumericalGrid<1>& grid, int cond)
{
    assert(cond == REFLECTING_BOUNDARY_COND
            || cond == PERIODIC_BOUNDARY_COND
            || cond == IMAG_TIME_PROPAGATION_COND);

    auto bf = TDSEBufferFD1D(grid, cond);
    double delta_x = grid.getDelta(0);
    
    bf.core.I.setIdentity();
    bf.core.D = create_matrix_hamiltonian(grid, cond);
    bf.core.D = bf.core.D * (-2.0);
    bf.core.M = bf.core.I + bf.core.D * (delta_x * delta_x / 12);
    std::cout << "[info] tdse_fd1d buffer created." << std::endl;

    return bf;
}


template<unsigned N>
inline SpMat create_matrix_hamiltonian(const NumericalGrid<N>& grid, int cond)
{
    TIME_TEST_START(create_H)
    assert(cond == REFLECTING_BOUNDARY_COND
		|| cond == PERIODIC_BOUNDARY_COND
		|| cond == IMAG_TIME_PROPAGATION_COND);

    int cnt = grid.getTotalCount();
    double scalers[N] = { 0.0 };
    for(int rank = 0; rank < N; rank++){
        double delta = grid.getDelta(rank);
        scalers[rank] = -1.0 / (2.0 * delta * delta);
    }

    SpMat mat(cnt, cnt);
    mat.reserve(Eigen::VectorXi::Constant(cnt, (1 + 2 * N)));

    GridIndice<N> indice;
    GridIndice<N> next_indice;

    double center_scaler = 0.0;
    for(int rank = 0; rank < N; rank++) {
        center_scaler += scalers[rank];
    }

    for(int i = 0; i < cnt; i++) {
        indice = grid.expand(i);
        for(int j = 0; j < N; j++){
            for(int k = -1; k <= 1; k += 2){
                next_indice = indice;
                next_indice[j] += k;
                if(next_indice[j] >= 0 && next_indice[j] < grid.getCount(j)){
                    int next_index = grid.shrink(next_indice);
                    mat.insert(i, next_index) = scalers[j];
                }
                else if(cond == PERIODIC_BOUNDARY_COND){
                    next_indice[j] = (next_indice[j] + grid.getCount(j)) % grid.getCount(j);
                    int next_index = grid.shrink(next_indice);
                    mat.insert(i, next_index) = scalers[j];
                }
            }
        }
        mat.insert(i, i) = -2.0 * center_scaler;
    }
    
    mat.makeCompressed();
    
    TIME_TEST_END(create_H);
    return mat;
}


template<unsigned N, typename _Tf>
inline SpMat create_infact_hamiltonian(const NumericalGrid<N>& grid, _Tf potiential_func, int cond)
{
    int cnt = grid.getTotalCount();
    LinSpaceData<N> linspace = create_linspace(grid);
    WaveData po_data = create_empty_wave(grid);
    update_wave_by_vectorized_func(po_data, linspace, potiential_func);
    SpMat H = create_matrix_hamiltonian(grid, cond);
    
    SpMat I(cnt, cnt);
    I.setIdentity();
    return H + I * po_data.asDiagonal();
}


inline void tridiagonal_mat_elimination (
    SpMat&                          A,
    Eigen::Ref<Eigen::VectorXcd>    X,
    Eigen::Ref<Eigen::VectorXcd>    B,
    bool                            corner_flag = false     // True if right-up & left-down corner has value (even equals 0).
    )
{
    TIME_TEST_START(trimat_elim)

	assert(A.cols() == X.rows() && A.cols() == B.rows());
    assert(A.nonZeros() == (3 * A.rows() - 2) || A.nonZeros() == 3 * A.rows());
    assert(A.isCompressed() == true);

    int cnt = A.cols();

	// for (int m = 1; m < cnt; m++) {
	// 	for (int l = m; l >= m - 1; l--) {
	// 		A.coeffRef(m, l) -= (A.coeffRef(m, m - 1) / A.coeffRef(m - 1, m - 1)) * A.coeffRef(m - 1, l);
	// 		B(m) -= A.coeffRef(m, m - 1) / A.coeffRef(m - 1, m - 1) * B(m - 1);
	// 	}
	// }
	// X(cnt - 1) = B(cnt - 1) / A.coeffRef(cnt - 1, cnt - 1);
	// for (int m = cnt - 2; m >= 0; m--) {
	// 	X(m) = (B(m) - A.coeffRef(m, m + 1) * X(m + 1)) / A.coeffRef(m, m);
	// }
    
    // the following code is equivalent to the above implement,
    // but run faster due to direct memory access, according to Eigen's doc.
    std::complex<double> *value_ptr = A.valuePtr();
    for(int i = 1; i < cnt; i++){
        int crt_id_apdix = corner_flag * ((i > 0 && i < cnt - 1) ? (1) : ((i == cnt - 1) ? 2 : 0));
        int last_id_apdix = corner_flag * (((i - 1) > 0 && (i - 1) < cnt - 1) ? (1) : (((i - 1) == cnt - 1) ? 2 : 0));
        
        std::complex<double> c = (*(value_ptr + i * 3 - 1 + crt_id_apdix) / *(value_ptr + (i - 1) * 3 + last_id_apdix));
        *(value_ptr + i * 3 + crt_id_apdix) -= c * (*(value_ptr + (i - 1) * 3 + 1 + last_id_apdix)); 
        B(i) -= c * B(i - 1);
        *(value_ptr + i * 3 - 1 + crt_id_apdix) = 0;
    }
    
    X(cnt - 1) = B(cnt - 1) / *(value_ptr + (cnt - 1) * 3 + corner_flag * 2);
    
    for(int i = cnt - 2; i >= 0; i--) {
        int crt_id_apdix = corner_flag * ((i > 0 && i < cnt - 1) ? (1) : ((i == cnt - 1) ? 2 : 0));
        X(i) = (B(i) - *(value_ptr + i * 3 + 1 + crt_id_apdix) * X(i + 1)) / *(value_ptr + i * 3 + crt_id_apdix);
    }

    TIME_TEST_END(trimat_elim)
}

typedef std::vector<std::complex<double>> TrimatElimBuffer;

inline void tridiagonal_mat_elimination_optimized (
    const SpMat&                            A,
    Eigen::Ref<Eigen::VectorXcd>            X,
    const Eigen::Ref<Eigen::VectorXcd>&     B,
    TrimatElimBuffer&                       A_add,
    TrimatElimBuffer&                       B_add,
    bool                                    corner_flag = false     // True if right-up & left-down corner has value (even equals 0).
)
{
    TIME_TEST_START(trimat_elim_o)

	assert(A.cols() == X.rows() && A.cols() == B.rows());
    assert(A.nonZeros() == (3 * A.rows() - 2) || A.nonZeros() == 3 * A.rows());
    assert(A.isCompressed() == true);

    int cnt = A.cols();
    const std::complex<double> *value_ptr = A.valuePtr();
    A_add[0] = 0;
    B_add[0] = 0;

    for(int i = 1; i < cnt; i++){
        int crt_id_apdix = corner_flag * ((i > 0 && i < cnt - 1) ? (1) : ((i == cnt - 1) ? 2 : 0));
        int last_id_apdix = corner_flag * (((i - 1) > 0 && (i - 1) < cnt - 1) ? (1) : (((i - 1) == cnt - 1) ? 2 : 0));
        
        std::complex<double> c = ((*(value_ptr + i * 3 - 1 + crt_id_apdix)) / (*(value_ptr + (i - 1) * 3 + last_id_apdix) + A_add[i - 1]));
        //*(value_ptr + i * 3 + crt_id_apdix) -= c * (*(value_ptr + (i - 1) * 3 + 1 + last_id_apdix)); 
        A_add[i] = -c * (*(value_ptr + (i - 1) * 3 + 1 + last_id_apdix));

        //B(i) -= c * B(i - 1);
        B_add[i] = -c * (B(i - 1) + B_add[i - 1]);
        
        //*(value_ptr + i * 3 - 1 + crt_id_apdix) = 0;
    }
    
    X(cnt - 1) = (B(cnt - 1) + B_add[cnt - 1]) / (*(value_ptr + (cnt - 1) * 3 + corner_flag * 2) + A_add[cnt - 1]);
    
    for(int i = cnt - 2; i >= 0; i--) {
        int crt_id_apdix = corner_flag * ((i > 0 && i < cnt - 1) ? (1) : ((i == cnt - 1) ? 2 : 0));
        X(i) = (B(i) + B_add[i] - *(value_ptr + i * 3 + 1 + crt_id_apdix) * X(i + 1)) / (*(value_ptr + i * 3 + crt_id_apdix) + A_add[i]);
    }

    TIME_TEST_END(trimat_elim_o)
}


inline void solve_linear_equation_fd1d_rcond(
    const Eigen::Ref<SpMat>&                A,
    Eigen::Ref<Eigen::VectorXcd>            X,
    const Eigen::Ref<Eigen::VectorXcd>&     B,
// temporary vars, to avoid redundant memory allocated.
    TDSEMainloopBufferFD1D&                 buffer
    )
{
    TIME_TEST_START(backforward_rc)
    tridiagonal_mat_elimination_optimized(A, X, B, buffer.A_add, buffer.B_add);
    TIME_TEST_END(backforward_rc)
}


inline void solve_linear_equation_fd1d_pcond(
    const Eigen::Ref<SpMat>&                A,
    Eigen::Ref<Eigen::VectorXcd>            X,
    const Eigen::Ref<Eigen::VectorXcd>&     B,
// temporary vars, to avoid redundant memory allocated.
    TDSEMainloopBufferFD1D&                 buffer
    )
{
    TIME_TEST_START(backforward_pc)
    assert(A.cols() == A.rows() && A.cols() == X.rows() && A.cols() == B.rows());
    int cnt = A.cols();

    buffer.A_tmp_1 = A;
	std::complex<double> corner_num = buffer.A_tmp_1.coeffRef(0, cnt - 1);

	buffer.udata = Eigen::VectorXcd::Zero(cnt);
	buffer.vdata = Eigen::VectorXcd::Zero(cnt);

	buffer.udata(0) = 1;
	buffer.udata(cnt - 1) = 1;
	buffer.vdata(0) = corner_num;
	buffer.vdata(cnt - 1) = corner_num;

	buffer.A_tmp_1.coeffRef(0, 0) -= corner_num;
	buffer.A_tmp_1.coeffRef(0, cnt - 1) -= corner_num;
	buffer.A_tmp_1.coeffRef(cnt - 1, 0) -= corner_num;
	buffer.A_tmp_1.coeffRef(cnt - 1, cnt - 1) -= corner_num;

	//buffer.A_tmp_2 = buffer.A_tmp_1;

    //buffer.B_tmp_1 = B;
	buffer.xdata1 = Eigen::VectorXcd::Zero(cnt);
    //buffer.B_tmp_2 = buffer.udata;
    buffer.xdata2 = Eigen::VectorXcd::Zero(cnt);

	tridiagonal_mat_elimination_optimized(buffer.A_tmp_1, buffer.xdata1, B, buffer.A_add, buffer.B_add, true);
	tridiagonal_mat_elimination_optimized(buffer.A_tmp_1, buffer.xdata2, buffer.udata, buffer.A_add, buffer.B_add, true);

	std::complex<double> tmp1 = buffer.vdata.transpose() * buffer.xdata2;
	std::complex<double> tmp2 = buffer.vdata.transpose() * buffer.xdata1;
	X = buffer.xdata1 - buffer.xdata2 * (tmp2 / (1.0 + tmp1));

    TIME_TEST_END(backforward_pc)
}


template<typename _Tf>
inline void update_A_matrix(
    TDSECoreFD1D&           core,
    TDSERuntimeBufferFD1D&  rt,
    _Tf                     potiential_func,
    double                  delta_t,
    double                  crt_t
)
{
    TIME_TEST_START(update_Am)
    using namespace std::complex_literals;

    std::complex<double> delta_t_infact = (core.cond == IMAG_TIME_PROPAGATION_COND) ? (-1i * delta_t) : delta_t;
    update_wave_by_vectorized_func(rt.po_data, core.linspace, potiential_func, crt_t);

    //V = po_data.asDiagonal();
    rt.A_pos = core.M - (core.D * (-0.5) + core.M * rt.po_data.asDiagonal()) * (0.5i * delta_t_infact);
    rt.A_neg = core.M + (core.D * (-0.5) + core.M * rt.po_data.asDiagonal()) * (0.5i * delta_t_infact);

    //if(core.cond == IMAG_TIME_PROPAGATION_COND)
    rt.infact_h = core.D * (-0.5) + core.I * rt.po_data.asDiagonal();

    TIME_TEST_END(update_Am)
}


// main function for Crank-Nicolson in FD, time-related. [user interface]
template<typename _Tf>
inline void crank_nicolson_method_1d(
    TDSEBufferFD1D&         bf,
    Eigen::Ref<WaveData>    crt_state,
    _Tf                     potiential_func,
    double                  start_t,
    double                  delta_t,
    int                     total_steps
)
{
    TIME_TEST_START(crank_nico)
    assert(total_steps >= 0 && delta_t > 0);
    assert(crt_state.rows() == bf.cnt);

    for(int i = 0; i < total_steps; i++){
        update_A_matrix(bf.core, bf.rt, potiential_func, delta_t, start_t + i * delta_t);

        bf.mlbf.half_state.noalias() = bf.rt.A_pos * crt_state;
        if(bf.cond == PERIODIC_BOUNDARY_COND)
            solve_linear_equation_fd1d_pcond(bf.rt.A_neg, crt_state, bf.mlbf.half_state, bf.mlbf);
        else
            solve_linear_equation_fd1d_rcond(bf.rt.A_neg, crt_state, bf.mlbf.half_state, bf.mlbf);
    }

    TIME_TEST_END(crank_nico)
}


// crank nicolson "mainloop" part for no-time situation.

inline void crank_nicolson_method_1d_mainloop_no_time(
    TDSERuntimeBufferFD1D&      rt,
    TDSEMainloopBufferFD1D&     mlbf,
    Eigen::Ref<WaveData>        crt_state,
    int                         total_steps
)
{
    TIME_TEST_START(cn_mainloop_nt)
    assert(total_steps >= 0);

    for(int i = 0; i < total_steps; i++){
        mlbf.half_state.noalias() = rt.A_pos * crt_state;
    
        if(rt.cond == PERIODIC_BOUNDARY_COND)
            solve_linear_equation_fd1d_pcond(rt.A_neg, crt_state, mlbf.half_state, mlbf);
        else
            //solve_linear_equation_fd1d_rcond(rt.A_neg, crt_state, mlbf.half_state, mlbf);  <-- slower ×2 (why???)
            tridiagonal_mat_elimination_optimized(rt.A_neg, crt_state, mlbf.half_state, mlbf.A_add, mlbf.B_add);
    }
    TIME_TEST_END(cn_mainloop_nt)
}


template<typename _Tf>
inline void crank_nicolson_method_1d_no_time(
    TDSEBufferFD1D&         bf,
    Eigen::Ref<WaveData>    crt_state,
    _Tf                     potiential_func,
    double                  delta_t,
    int                     total_steps
)
{
    assert(total_steps >= 0 && delta_t > 0);
    assert(crt_state.rows() == bf.cnt);
    
    update_A_matrix(bf.core, bf.rt, potiential_func, delta_t, 0.0);
    crank_nicolson_method_1d_mainloop_no_time(bf.rt, bf.mlbf, crt_state, total_steps);
}

}

#endif