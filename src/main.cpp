#include "cqp_time_test.h"
#include "cqp_util.hpp"
#include "wave_function.hpp"
#include "tdse_fd.hpp"
#include "tdse_fft.hpp"
#include "cqp_sh.hpp"
#include "cqp_itp.hpp"
#include "tdse_sh_laser.hpp"
#include "export_data_to_file.hpp"
#include <Eigen/Eigenvalues>
using namespace std;
using namespace CQP;

//b /usr/local/include/Eigen/src/Core/util/Memory.h:167

void test_tdse()
{
    NumericalGrid1D grid(1000, 20, 0);
    WaveData wave = create_empty_wave(grid);
    LinSpaceData1D linspace = create_linspace(grid);

    update_wave_by_vectorized_func(wave, linspace, gauss_vfunc_1d(10, 0, 5), 0);
    WaveData wave2 = wave;
    export_wave_to_file(grid, wave, "iwave", XYZ);

    Eigen::ArrayXd a1 = Eigen::ArrayXd::Random(5);
    cout << a1 << endl;
    cout << (a1.sign() + 1) << endl;

    auto po_func = CREATE_1D_VFUNC(XS * 0);
    //auto fft_bf = init_split_operator(grid, PERIODIC_BOUNDARY_COND);
    auto fd_bf = init_crank_nicolson_1d(grid, REFLECTING_BOUNDARY_COND);
    update_A_matrix(fd_bf.core, fd_bf.rt, po_func, 0.01, 0.0);

    cout << "init energy = " << wave2.dot(fd_bf.rt.infact_h * wave2).real() << endl;
    //split_operator_method_no_time(fft_bf, wave, po_func, 0.01, 350);
    crank_nicolson_method_1d_mainloop_no_time(fd_bf.rt, fd_bf.mlbf, wave2, 300);

    double energy = wave2.dot(fd_bf.rt.infact_h * wave2).real();
    cout << "final energy = " << energy << endl;

    //export_wave_to_file(grid, wave, "fft", XYZ);
    export_wave_to_file(grid, wave2, "fd", XYZ);

    //display_time_test_report();
}

void test_fft()
{
    constexpr unsigned N = 2;

    NumericalGrid<N> grid(200, 20, 0, 200, 20, 0);
    WaveData wave = create_empty_wave(grid);
    WaveData res = create_empty_wave(grid);
    LinSpaceData<N> linspace = create_linspace(grid);

    update_wave_by_vectorized_func(wave, linspace, gauss_vfunc_2d(1, 0, 2, 5, 0, 0), 0);
    export_wave_to_file(grid, wave, "wave1", XYZ, REAL_MODE);

    auto core1 = create_fft_core(grid);
    auto core2 = create_fft_core(grid_after_fft(grid));
    fft(wave, res, core1, FFTW_FORWARD);
    export_wave_to_file(grid, res, "wave2", XYZ, REAL_MODE);
    fft(res, wave, core2, FFTW_BACKWARD);
    export_wave_to_file(grid, wave, "wave3", XYZ, REAL_MODE);
}


void test_tdse_fft_nd()
{
    constexpr unsigned N = 3;
    NumericalGrid<N> grid(512, 40, 0, 512, 40, 0, 512, 40, 0);
    LinSpaceData<N> linspace = create_linspace(grid);

    // init waves.
    constexpr size_t M = 1;
    vector<WaveData> waves(M, create_empty_wave(grid));

    for(size_t i = 0; i < M; i++){
        waves[i] = create_random_wave(grid);
        waves[i].normalize();
    }

    // imag time propagation process
    auto pofunc = CREATE_3D_VFUNC(RSQRT(XS * XS + YS * YS + ZS * ZS + 1e-2) * -1.0);
    auto buffer = init_split_operator(grid, IMAG_TIME_PROPAGATION_COND);
    update_fft_core_no_time(buffer, pofunc, 0.5);
    
    cout << "main loop" << endl;

    TIME_TEST_START(imag_time_pg)
    std::complex<double> norm_value = 0.0;
    // main process
    for(int i = 0; i < 25; i++){
        for(size_t j = 0; j < M; j++){
            split_operator_method_mainloop_no_time(buffer, waves[j], 1);
            //normalize
            fast_inner_product(waves[j], waves[j], norm_value);
            fast_cwise_scale(waves[j], 1.0 / std::sqrt(norm_value.real()), waves[j]);
        }
        fast_gram_schmidt(waves);
        cout << "[info] step " << i << " is done." << endl;
    }
    TIME_TEST_END(imag_time_pg)

    // output data
    SpMat H = create_infact_hamiltonian(grid, pofunc, IMAG_TIME_PROPAGATION_COND);
    char wave_name[20];
    for(int j = 0; j < M; j++) {
        complex<double> energy = (waves[j].dot(H * waves[j]));
        cout << "Energy " << j << " = " << energy.real() << endl;
        sprintf(wave_name, "wave%d", j);
        //export_wave_to_file(grid, waves[j], wave_name, XYZ);
    }

    //display_time_test_report();
}

void test_double_and_float()
{
    constexpr unsigned N = 3;
    NumericalGrid<N> grid(256, 20, 0, 256, 20, 0, 256, 20, 0);
    LinSpaceData<N> linspace = create_linspace(grid);

    WaveData wave1 = create_empty_wave(grid);
    WaveData wave2 = create_empty_wave(grid);
    WaveData wave3 = create_empty_wave(grid);

    TIME_TEST_START(update)
    update_wave_by_vectorized_func(wave1, linspace, gauss_vfunc_3d(1, 0, 1, 1, 0, 1, 1, 0, 1));
    TIME_TEST_END(update)

    /*
        return CC(EXP(POW2((xs - x0) / (2 * omega_x)) * -1.0))
			* (CC(COS(xs * px0)) + CC(SIN(xs * px0)) * 1i) * Cx
			* CC(EXP(POW2((ys - y0) / (2 * omega_y)) * -1.0))
			* (CC(COS(ys * py0)) + CC(SIN(ys * py0)) * 1i) * Cy
			* CC(EXP(POW2((zs - z0) / (2 * omega_z)) * -1.0))
			* (CC(COS(zs * pz0)) + CC(SIN(zs * pz0)) * 1i) * Cz;
    */
    LinSpaceData_f<N> linspace_f = create_linspace_f(grid);

    TIME_TEST_START(cwise_mul)
    wave2 = CC(EXP(POW2((linspace_f.col(0) - 0.0f) / (2 * 1.0f)) * -1.0)) \
		* (CC(COS(linspace_f.col(0) * 1.0f)) + CC(SIN(linspace_f.col(0) * 1.0f)) * 1i) \
        * CC(EXP(POW2((linspace_f.col(1) - 0.0f) / (2 * 1.0f)) * -1.0)) \
		* (CC(COS(linspace_f.col(1) * 1.0f)) + CC(SIN(linspace_f.col(1) * 1.0f)) * 1i) \
        * CC(EXP(POW2((linspace_f.col(2) - 0.0f) / (2 * 1.0f)) * -1.0)) \
		* (CC(COS(linspace_f.col(2) * 1.0f)) + CC(SIN(linspace_f.col(2) * 1.0f)) * 1i);
    TIME_TEST_END(cwise_mul)

    cout << (wave1.normalized() - wave2.normalized()).sum() << endl;

    Eigen::VectorXd tmp1 = Eigen::VectorXd::Random(pow(256, 3));
    Eigen::VectorXf tmp2 = Eigen::VectorXf::Random(pow(256, 3));
    
    Eigen::VectorXd res1((int)pow(256, 3));
    Eigen::VectorXf res2((int)pow(256, 3));

    TIME_TEST(res1 = tmp1.cwiseProduct(tmp1), mul_d);
    TIME_TEST(res2 = tmp2.cwiseProduct(tmp2), mul_f);
    
    //display_time_test_report();
}


void test_sh()
{
    TIME_TEST_START(main)
    int l_num = 5;
    int M = 1;
    NumericalGrid1D rgrid = create_r_grid(2048, 100);
    LinSpaceData1D linspace = create_linspace(rgrid);

    // init strategy
    std::vector<WaveDataSH> shwaves(M);

    std::vector<int> lmap(l_num * l_num, 0);
    update_lmap_for_sh(lmap, l_num);
    int select_list[] = {
        get_index_from_lm(0, 0, l_num),
        get_index_from_lm(0, 0, l_num),
        get_index_from_lm(0, 0, l_num),
        get_index_from_lm(0, 0, l_num),
        get_index_from_lm(0, 0, l_num)
    };
    for(int i = 0; i < M; i++){
        shwaves[i] = create_empty_wave_sh(rgrid, l_num);
        auto func = CREATE_1D_VFUNC_C(XS * EXP(-XS / (i + 1.0)), i);
        update_wave_by_vectorized_func(shwaves[i].col(select_list[i]), linspace, func);
    }

    auto r_po_func = CREATE_1D_VFUNC(RSQRT(POW2(XS) + 1e-2) * -1.0);
    TDSEBufferSH bf = init_fd_sh(rgrid, l_num, IMAG_TIME_PROPAGATION_COND, true, 1.0);

    update_runtime_sh(bf, r_po_func, 0.1, 0.0);
    export_wave_to_file(rgrid, bf.rts[0].po_data, "pofunc", XYZ);
    
    for(int i = 0; i < 100; i++){
        for(int j = 0; j < M; j++){
            tdse_fd_sh_mainloop_no_time(bf, shwaves[j], 1);
            normalize_sh(shwaves[j]);
        }
        gram_schmidt_sh(shwaves);
    }

    SpMat M2_T = bf.core_r.M.transpose();
    SpMat M2_c_T = bf.core_r_coulomb.M.transpose();
    SpMat H(bf.cnt_r, bf.cnt_r);
    std::vector<std::complex<double>> A_add(bf.cnt_r, 0);
    std::vector<std::complex<double>> B_add(bf.cnt_r, 0);
    WaveData rfunc_tmp = create_empty_wave(bf.cnt_r);

    for(int j = 0; j < M; j++){
        double energy = 0;
        for(int i = 0; i < bf.l_num * bf.l_num; i++){
            int l = bf.l_map[i];
            //energy += shwaves[j].col(i).dot(bf.rts[l].infact_h * shwaves[j].col(i)).real();
            if(l == 0){
                tridiagonal_mat_elimination_optimized(M2_c_T, rfunc_tmp, shwaves[j].col(i), A_add, B_add);
                H = bf.core_r_coulomb.D + bf.core_r_coulomb.M * bf.rts[l].po_data.asDiagonal();
                energy += rfunc_tmp.transpose().dot(H * shwaves[j].col(i)).real();
            }
            else{
                tridiagonal_mat_elimination_optimized(M2_T, rfunc_tmp, shwaves[j].col(i), A_add, B_add);
                H = bf.core_r.D + bf.core_r.M * bf.rts[l].po_data.asDiagonal();
                energy += rfunc_tmp.transpose().dot(H * shwaves[j].col(i)).real();
            }
            //cout << "test energy: " << energy << endl;
        }
        cout << "Energy " << j << " : " << energy << endl;
    }

    export_wave_to_file(rgrid, shwaves[0].col(0), "rfunc0_ori", XYZ);

    TIME_TEST_END(main)
    //display_time_test_report();
}


struct TriMat
{
    Eigen::VectorXcd diag_data;
    Eigen::VectorXcd subdiag_data;
    Eigen::VectorXcd 
}

void test_practice()
{
    int cnt_r = 25000;
    NumericalGrid1D grid(cnt_r, 1000, 0);
    Eigen::SparseMatrix<std::complex<double>, Eigen::ColMajor> mat1(cnt_r, cnt_r);
    Eigen::VectorXcd vec1 = Eigen::VectorXcd::Random(cnt_r);
    Eigen::VectorXcd vec2(cnt_r);
    Eigen::VectorXcd tridiag_mat_1 = Eigen::VectorXcd::Random(cnt_r);
    Eigen::VectorXcd tridiag_mat_2 = Eigen::VectorXcd::Random(cnt_r);
    mat1 = create_matrix_hamiltonian(grid, REFLECTING_BOUNDARY_COND);
    
    TIME_TEST_START(mat_mul)
    for(int i = 0; i < 100; i++){
        vec2 += tridiag_mat_1.cwiseProduct(vec1);
        vec2 += tridiag_mat_2.cwiseProduct(vec1);
    }
    TIME_TEST_END(mat_mul)

    display_time_test_report({"mat_mul"});
}


void test_sh_laser_on()
{
    int l_num = 50;
    NumericalGrid1D rgrid = create_r_grid(25000, 1000);
    LinSpaceData1D linspace = create_linspace(rgrid);
    std::vector<WaveDataSH> shwave(1, create_empty_wave_sh(rgrid, l_num));
    auto r_po_func = CREATE_1D_VFUNC(-INV(XS));

    // using itp, get the base state of coulomb potiential
    init_shwaves_for_itp(rgrid, shwave, l_num);
    imag_time_propagation_sh(rgrid, shwave, l_num, r_po_func, true, 1.0, 0.1, 1e-8);

    // prepare for tdse sh with polar laser
    auto bf = init_tdse_sh_pl(rgrid, r_po_func, l_num, 1.0);
    update_runtime_sh_polar_laser(bf, 0.01);

    auto exbf = init_expected_value_solver_sh_pl(bf);

    double E0 = 0.05, omega = 0.1, nc = 9.0;
    double t_offset = PI / 2;
    auto At_func = CREATE_1D_VFUNC_C((E0 / omega) * POW2(COS(omega * (XS) / 2.0 / nc - t_offset)) * SIN(omega * XS), E0, omega, nc, t_offset);
    // auto At_func = CREATE_1D_VFUNC(XS * 0);
    // prepare time grid, as well as At
    double dt = 0.01, time_span = 0.1;

    NumericalGrid1D time_grid(time_span / dt, time_span, time_span / 2);
    LinSpaceData1D t_linspace = create_linspace(time_grid);
    WaveData At = create_empty_wave(time_grid);
    WaveDataSH ori_func = create_empty_wave_sh(rgrid, l_num);
    update_wave_by_vectorized_func(At, t_linspace, At_func);
    export_wave_to_file(time_grid, At, "At", XYZ, REAL_MODE);

    // mainloop of tdse with polar laser
    double at = 0.0;
    double integral_result = 0.0;
    double energy = 0.0;
    //tdse_sh_polar_laser_once(bf, shwave[0], At, dt, 100);
    for(int i = 0; i < time_grid.getTotalCount(); i++){
        //cout << i << endl;
        at = At(i).real();
        tdse_sh_polar_laser_once(bf, shwave[0], at, dt);
        integral_result += 0.5 * at * at * dt;
        if(i % 200 == 0){
            ori_func = shwave[0] * std::exp(-1i * integral_result);
            energy = get_energy_sh_pl(bf, exbf, shwave[0], at, dt);
            cout << "step " << i << " energy = " << energy << endl;        
        }
    }
    
    char name_str[10];
    for(int j = 0; j < l_num; j++) {
        sprintf(name_str, "rfunc_%d", j);
        export_wave_to_file(rgrid, shwave[0].col(j), name_str, XYZ);
    }

    display_time_test_report({"mm_task", "get_energy", "left_part", "apply_Hat", "pure_lspace", "apply_Hang", "apply_Hmix", "trimat_elim_o"});
}


int main()
{
    ios::sync_with_stdio(0);
    init_for_openmp();
    test_practice();
    return 0;
}