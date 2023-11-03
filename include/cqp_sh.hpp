#ifndef __CQP_SH_HPP__
#define __CQP_SH_HPP__

#include "wave_function.hpp"
#include "tdse_fd.hpp"
#include "numerical_grid.hpp"
#include "cqp_util.hpp"

#include <vector>

namespace CQP {

////////////
// sh part

typedef Eigen::MatrixXcd WaveDataSH;


/*
inline int get_index_sh(int l, int m, int l_num) 
    { return l * l + m + l; }

inline int get_l_sh(int index, int l_num)
    { return (int)std::floor(sqrt(index)); }

inline int get_m_sh(int index)
    { int l = get_l_sh(index); return -l + (index - l * l); }
*/

inline auto create_empty_wave_sh(const NumericalGrid1D& grid, int l_num)  
{
    return Eigen::MatrixXcd::Zero(grid.getTotalCount(), l_num * l_num);
}

inline auto create_empty_wave_sh(int cnt_r, int l_num)  
{
    return Eigen::MatrixXcd::Zero(cnt_r, l_num * l_num);
}

inline auto create_random_wave_sh(const NumericalGrid1D& grid, int l_num)
{
    return Eigen::MatrixXcd::Random(grid.getTotalCount(), l_num * l_num);
}

inline auto create_random_wave_sh(int cnt_r, int l_num)
{
    return Eigen::MatrixXcd::Random(cnt_r, l_num * l_num);
}

inline NumericalGrid1D create_r_grid(int cnt, double max_r)
{
    return NumericalGrid1D(cnt, max_r, max_r / 2 + (max_r / cnt));
}


inline std::complex<double> inner_product_sh(
	Eigen::Ref<WaveDataSH>		input1,
	Eigen::Ref<WaveDataSH>		input2)
{
    assert(input1.rows() == input2.rows() && input1.cols() == input2.cols());

	std::complex<double> ans = 0;

    TIME_TEST_START(in_product_sh)
    for(int i = 0; i < input1.cols(); i++){
        ans += input1.col(i).dot(input2.col(i));
    }
	TIME_TEST_END(in_product_sh)

    return ans;
}


inline std::complex<double> fast_inner_product_sh(
	Eigen::Ref<WaveDataSH>		input1,
	Eigen::Ref<WaveDataSH>		input2
)
{
    assert(input1.rows() == input2.rows() && input1.cols() == input2.cols());
    assert(input1.cols() <= MAX_BUFFER_SIZE_OF_INNER_PRODUCT_SH);
    
    int cnt = input1.cols();
	std::complex<double> ans = 0;
    static std::vector<std::complex<double>> buffer(MAX_BUFFER_SIZE_OF_INNER_PRODUCT_SH, 0);

    TIME_TEST_START(f_dot_sh)

    #pragma omp parallel for
    for(int i = 0; i < cnt; i++){
        buffer[i] = input1.col(i).dot(input2.col(i));
    }

    for(int i = 0; i < cnt; i++){
        ans += buffer[i];
    }

    TIME_TEST_END(f_dot_sh)
    return ans;
}


inline std::complex<double> fast_inner_product_sh_tasks(
	Eigen::Ref<WaveDataSH>		input1,
	Eigen::Ref<WaveDataSH>		input2
)
{
    assert(input1.rows() == input2.rows() && input1.cols() == input2.cols());
    assert(input1.cols() <= MAX_BUFFER_SIZE_OF_INNER_PRODUCT_SH);
    
    int cnt = input1.cols();
	std::complex<double> ans = 0;
    static std::vector<std::complex<double>> buffer(MAX_BUFFER_SIZE_OF_INNER_PRODUCT_SH, 0);

    for(int i = 0; i < cnt; i++){
        #pragma omp task
        buffer[i] = input1.col(i).dot(input2.col(i));
    }

    #pragma omp taskwait

    for(int i = 0; i < cnt; i++){
        ans += buffer[i];
    }
    
    return ans;
}


inline void gram_schmidt_sh(std::vector<WaveDataSH>& waves)
{
	TIME_TEST_START(g_schmidt_sh)
	std::vector<std::complex<double>> self_product_buffer(waves.size(), 0.0);

	for (int i = 0; i < waves.size(); i++) {
		self_product_buffer[i] = inner_product_sh(waves[i], waves[i]);
	}

	for (int i = 0; i < waves.size(); i++) {
		for (int j = 0; j < i; j++) {
			std::complex<double> scaler = inner_product_sh(waves[j], waves[i]) / self_product_buffer[j];
			//waves[i] = waves[i] - waves[j] * scaler;
			fast_cwise_add(waves[i], - waves[j] * scaler, waves[i]);
		}
	}
	TIME_TEST_END(g_schmidt_sh)
}


inline void normalize_sh(Eigen::Ref<WaveDataSH> wave)
{
    std::complex<double> scaler = 1.0 / std::sqrt(inner_product_sh(wave, wave));
    //std::cout << scaler << " ";
    fast_cwise_scale(wave, scaler, wave);
}


////////////////
// tdse_sh


struct TDSEBufferSH
{
    int cnt_r;
    int l_num;
    bool coulomb_boost_flag;
    std::vector<int> l_map;
    
    // the TDSECore defined for all rfuncs.
    TDSECoreFD1D            core_r;
    TDSECoreFD1D            core_r_coulomb;     // used for coulomb-like potiential boosting.

    // runtime buffers for a group of l parts. (size = l_num)
    std::vector<TDSERuntimeBufferFD1D>      rts;

    // mainloops buffers for each rfuncs. (size = l_num * l_num)
    std::vector<TDSEMainloopBufferFD1D>     mlbfs;

    // V appendix for l parts (size = l_num)
    std::vector<WaveData>                   vl_appendix;

    // Never call this constructor directly !
    TDSEBufferSH(const NumericalGrid1D& rgrid, int l_num, int cond, bool coulomb_flag)
        : 
        cnt_r(rgrid.getTotalCount()), l_num(l_num),
        coulomb_boost_flag(coulomb_flag), l_map(l_num * l_num, 0),

        core_r(rgrid, cond),
        core_r_coulomb(rgrid, cond),

        rts(l_num, TDSERuntimeBufferFD1D(rgrid, cond)),
        mlbfs(l_num * l_num, TDSEMainloopBufferFD1D(rgrid, cond)),
        vl_appendix(l_num, create_empty_wave(cnt_r))
    {}
};


inline SpMat create_M2_for_sh(const NumericalGrid1D& grid)
{
	int cnt = grid.getTotalCount();
	double scaler = -1.0 / 6.0;

	SpMat matrix(cnt, cnt);
    matrix.reserve(Eigen::VectorXi::Constant(cnt, 3));

    for(int i = 0; i < cnt; i++) {
        if(i - 1 >= 0) matrix.insert(i, i - 1) = 1.0 * scaler;
        matrix.insert(i, i) = 10.0 * scaler;
        if(i + 1 < cnt) matrix.insert(i, i + 1) = 1.0 * scaler;
    }
    matrix.makeCompressed();

	return matrix;
}

inline void update_lmap_for_sh(std::vector<int>& lmap, int l_num)
{
    assert(lmap.size() == l_num * l_num);
    int l = 0, m = 0;
    bool flag = 0;
    for(size_t i = 0; i < l_num * l_num; i++){
        flag = (l + 1) == l_num;
        lmap[i] = l;
        m = flag * ((m <= 0) ? (-m + 1) : (-m)) + (!flag) * m;
        l = (flag) ? (std::abs(m)) : (l + 1);
    }
}

inline int get_index_from_lm(int l, int m, int l_num)
{
    int lmax = l_num - 1;
    int mm = (m <= 0) ? (-2 * m) : (2 * m - 1);
    return l + (lmax * (mm / 2) - (mm / 2 - 1) * (mm / 2) / 2) * 2 + (lmax - mm / 2) * (mm % 2);
}

inline int get_m_from_mm(unsigned mm)
{
    return (mm % 2 == 0) ? (mm - (mm / 2) * 3) : (mm - (mm - 1) / 2); 
}

inline TDSEBufferSH init_fd_sh(
    const NumericalGrid1D&  rgrid,
    int                     l_number, 
    int                     cond,
    bool                    coulomb_flag = false,
    double                  Z = 1.0)
{
    assert(l_number > 0);
    assert(cond == REFLECTING_BOUNDARY_COND
        || cond == IMAG_TIME_PROPAGATION_COND);

    auto bf = TDSEBufferSH(rgrid, l_number, cond, coulomb_flag);
    double delta_x = rgrid.getDelta(0);

    // init core_r
    bf.core_r.I.setIdentity();
    bf.core_r.D = create_matrix_hamiltonian(rgrid, cond);
    bf.core_r.D = bf.core_r.D * (-2.0);
    bf.core_r.M = create_M2_for_sh(rgrid); // M is M2 now.

    if(coulomb_flag == true){
        std::complex<double> d = (-2.0 / (delta_x * delta_x)) * (1.0 - Z * delta_x / (12.0 - 10.0 * Z * delta_x)); 
        bf.core_r_coulomb.I.setIdentity();
        bf.core_r_coulomb.D = bf.core_r.D;
        bf.core_r_coulomb.D.coeffRef(0, 0) = d;
        bf.core_r_coulomb.M = bf.core_r.M;
        bf.core_r_coulomb.M.coeffRef(0, 0) = -2.0 * (1.0 + delta_x * delta_x * d / 12.0);
    } 

    // init vl_appendix
    for(int l = 0; l < bf.l_num; l++){
        double ll = static_cast<double>(l);
        auto apdix_func = CREATE_1D_VFUNC_C((ll * (ll + 1) / 2.0) * INV(POW2(XS)), ll);
        update_wave_by_vectorized_func(bf.vl_appendix[l], bf.core_r.linspace, apdix_func);
    }

    // init lmap
    update_lmap_for_sh(bf.l_map, bf.l_num);
    
    return bf;
}


template<typename _Tf>
inline void update_runtime_sh(
    TDSEBufferSH&       bf,
    _Tf                 r_potiential_func,
    double              delta_t,
    double              crt_t
)
{
    TIME_TEST_START(update_rtsh)
    using namespace std::complex_literals;
    std::complex<double> delta_t_infact = (bf.core_r.cond == IMAG_TIME_PROPAGATION_COND) ? (-1i * delta_t) : delta_t;

    for(int i = 0; i < bf.l_num; i++){
        update_wave_by_vectorized_func(bf.rts[i].po_data, bf.core_r.linspace, r_potiential_func, crt_t);
        fast_cwise_add(bf.rts[i].po_data, bf.vl_appendix[i], bf.rts[i].po_data);

        if(bf.coulomb_boost_flag == true && i == 0){
            bf.rts[i].A_pos = bf.core_r_coulomb.M - (bf.core_r_coulomb.D + bf.core_r_coulomb.M * bf.rts[i].po_data.asDiagonal()) * (0.5i * delta_t_infact);
            bf.rts[i].A_neg = bf.core_r_coulomb.M + (bf.core_r_coulomb.D + bf.core_r_coulomb.M * bf.rts[i].po_data.asDiagonal()) * (0.5i * delta_t_infact);
        }
        else{
            bf.rts[i].A_pos = bf.core_r.M - (bf.core_r.D + bf.core_r.M * bf.rts[i].po_data.asDiagonal()) * (0.5i * delta_t_infact);
            bf.rts[i].A_neg = bf.core_r.M + (bf.core_r.D + bf.core_r.M * bf.rts[i].po_data.asDiagonal()) * (0.5i * delta_t_infact);
        }

        if(bf.core_r.cond == IMAG_TIME_PROPAGATION_COND)
            bf.rts[i].infact_h = bf.core_r.D * (-0.5) + bf.core_r.I * bf.rts[i].po_data.asDiagonal();
    }
    TIME_TEST_END(update_rtsh)
}


inline void tdse_fd_sh_mainloop_no_time(
    TDSEBufferSH&           bf,
    Eigen::Ref<WaveDataSH>  crt_state_sh,
    int                     total_steps
)
{
    TIME_TEST_START(fd_mainloop_sh)
    assert(total_steps >= 0);

#ifdef _USING_OPENMP_IN_SH
    #pragma omp parallel for num_threads(THREAD_NUM_SH)
#endif
    for(int i = 0; i < bf.l_num * bf.l_num; i++){
        int l = bf.l_map[i];
        crank_nicolson_method_1d_mainloop_no_time(bf.rts[l], bf.mlbfs[i], crt_state_sh.col(i), total_steps);
    }
    TIME_TEST_END(fd_mainloop_sh)
}


template<typename _Tf>
inline void tdse_fd_sh_no_time(
    TDSEBufferSH&           bf,
    Eigen::Ref<WaveDataSH>  crt_state_sh,
    _Tf                     r_potiential_func,
    double                  delta_t,
    int                     total_steps
)
{
    update_runtime_sh(bf, r_potiential_func, delta_t, 0.0);
    tdse_fd_sh_mainloop_no_time(bf, crt_state_sh, total_steps);
}





}

#endif //__CQP_SH_HPP__