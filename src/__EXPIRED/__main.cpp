#include <iostream>
#include "Wavefunction.hpp"
#include "NumericalGrid.hpp"
#include "ExportDataToFile.hpp"
#include "TDSESolverFD.hpp"
#include "TDSESolverFFT.hpp"
#include "ImagTimePropagation.hpp"
#include "TDSESolverFDInSH.hpp"
#include <cmath>
#include "TestConfig.h"
#include <omp.h>
#include "FFTExecutor.hpp"
#include "Util.hpp"
#include "TimeTest.h"
using namespace std;
using namespace CQP;


Wavefunction<3> initWaveStrategy(const NumericalGrid<3>& grid, int order)
{
	Wavefunction<3> iwave = createRandomWave(grid, order);
	auto pkg = makeGaussPkgND<3>(std::vector<double>(3, 1), std::vector<double>(3, 0), std::vector<double>(3, 0));
	Wavefunction<3> envelope = createWaveByExpression(grid, pkg);
	return iwave * envelope;
}


void test_imag_time_propagation()
{
	constexpr int order = 3;
	constexpr int N = 3;
	NumericalGrid<N> grid(256, 40, 0, 256, 40, 0, 256, 40, 0);
	cout << "grid created." << endl;

	MathFuncType<N> func = [](double x, double y, double z) { return - 1.0 / (sqrt(x * x + y * y + z * z + 0.1)); };
	auto po_func = createWaveByExpression(grid, func);
	cout << "wave created." << endl;

	InitWaveGenerator<N> generator = initWaveStrategy;
	ImagTimePropagationSolver<N, order, TDSESolverFFT<N>, ITP_SCHMIDT > solver1(po_func, generator, 1, 1e-6);
	cout << "imaginary time solver created." << endl;

	solver1.execute();
	cout << "Execution ended." << endl;

	//vector< WaveMsg<N> > buffer;
	for (int i = 0; i < order; i++) {
		auto eigen_state1 = solver1.getResult(i);
		auto energy1 = solver1.getEnergy(i);
		cout << "Energy " << i << " = " << energy1 << endl;
		//buffer.push_back(WaveMsg<N>(eigen_state1, "it", 1));
	}
	//WaveAnimation3DToMat(buffer, NORM_MODE, CARTESIAN_COORDINATE);
}


WavefunctionSH initWaveStrategySH(const NumericalGrid1D& grid, int maxl, int order)
{
	WavefunctionSH ans(grid, maxl);
	
	int select_list[] = {
		0, 
		0, 1, 2, 3,
		0, 1, 2, 3, 4, 5, 6, 7, 8,
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
	};
	
	/*
	int select_list[] = {
		0, 0, 0, 0, 0
	};
	*/

	ans[select_list[order]] = createWaveByExpression(grid, [order](double r){ return r * std::exp(-r / (order + 1));});
	ans[select_list[order]].normalize();
	return ans;
}

void test_imag_time_propagation_sh()
{
	constexpr int order = 3;
	constexpr int N = 3;
	constexpr int maxl = 8;

	SphericalGrid grid = createSphericalGrid(50, 500, 50, 50);
	MathFuncType<N> func = [](double r, double theta, double phi){ return -1.0 / r; };
	auto po_func = createSCWaveByExpr(grid, func);
	
	ImagTimePropagationSolverSH<order> solver(po_func, initWaveStrategySH, 0.1, maxl, 1e-7);
	cout << "imaginary time solver created." << endl;

	solver.execute();
	cout << "Execution ended." << endl;

	/*
	char name_str[10];
	for (int i = 0; i < order; i++) {
		auto energy = solver.getEnergy(i);
		auto state = convergeShwaveToSCWave(grid, solver.getResult(i));
		cout << "Energy " << i << " = " << energy << endl;
		
		sprintf(name_str, "state_%d", i);
		auto wave = convertToFullThetaWaveInPolar(state.getSlice<2>({std::make_pair(PHI_DEGREE, 0)}));
		exportWaveToFile(wave, name_str, POLAR);
	}
	*/
}

void test_omp()
{
	//build a grid with 2^25 points
	NumericalGrid3D grid(200, 20, 0, 200, 20, 0, 200, 20, 0);

	//create a wave function, using MathFunc expr
	Wavefunction3D wave1 = createWaveByExpression(grid, makeGaussPkg3D(1, 1, 1, 0, 0, 0, 0, 0, 0));
	Wavefunction3D wave2(grid);

	FFTExecutor<3, FFT_D> e(grid);
	for(int i = 0; i < 50; i++){
		TIME_TEST(e.inputWave(wave1), inputwave);
		TIME_TEST(e.execute(), execute);
	}
	
	display_report();
}


void test_eigen()
{	
	
}


int main(int argc, char *argv[])
{
	init_for_openmp();
	test_eigen();
	
	return 0;
}