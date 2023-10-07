#include "ExportDataToMatFile.hpp"
#include "Wavefunction.hpp"
#include <mat.h>
#include <math.h>
#include <complex>
#include <memory.h>
#include <iostream>
#include <map>
using namespace std::literals;
using namespace CQP;

int SingleWave1DToMat(const Wavefunction1D& wave, int mode)
{
	MATFile* pMatFile = NULL;
	mxArray* pMxArray1 = NULL, * pMxArray2 = NULL;
	mxArray* pMxCount = NULL;
	double* pData1 = NULL, * pData2 = NULL;
	int error_code = 0;  //0: normal

	pMatFile = matOpen("test_mat_file.mat", "w");
	if (pMatFile == NULL) {
		error_code = -1;
		return error_code;
	}

	int len = wave.getGrid().getTotalCount();
	pMxArray1 = mxCreateDoubleMatrix(len, 1, mxREAL);
	pMxArray2 = mxCreateDoubleMatrix(len, 1, mxREAL);
	if (!pMxArray1 || !pMxArray2) {
		error_code = -2;
		matClose(pMatFile);
		return error_code;
	}

	pData1 = (double*)mxCalloc(len, sizeof(double));
	pData2 = (double*)mxCalloc(len, sizeof(double));
	if (!pData1 || !pData2) {
		error_code = -3;
		matClose(pMatFile);
		return error_code;
	}

	double norm = 0;
	for (int i = 0; i < len; i++) {
		pData1[i] = wave.getGrid().index(i).x();
		norm = sqrt(pow(wave.getValueByIndex(i).real(), 2) + pow(wave.getValueByIndex(i).imag(), 2));
		switch (mode) {
		case NORM_MODE:
			pData2[i] = norm;
			break;
		case REAL_MODE:
			pData2[i] = wave.getValueByIndex(i).real();
			break;
		case IMAG_MODE:
			pData2[i] = wave.getValueByIndex(i).imag();
			break;
		}
	}

	mxSetData_800(pMxArray1, pData1);
	mxSetData(pMxArray2, pData2);
	matPutVariable(pMatFile, "x_1", pMxArray1);
	matPutVariable(pMatFile, "y_1", pMxArray2);
	
	mxFree(pData1);
	mxFree(pData2);

	matClose(pMatFile);
	return 0;
}

int WaveAnimation1DToMat(const std::vector< WaveMsg<1> >& objects, int mode)
{
	MATFile* pMatFile = NULL;
	mxArray* pMxArray1 = NULL, * pMxArray2 = NULL;
	mxArray* pMxCount = NULL;
	double* pData1 = NULL, * pData2 = NULL;
	char name_str_1[10], name_str_2[10];
	int error_code = 0;  //0: normal
	std::map<std::string, int> cnt_map;

	pMatFile = matOpen("test_mat_file.mat", "w");
	if (pMatFile == NULL) {
		error_code = -1;
		return error_code;
	}

	int total_cnt = 0;
	for (auto iter = objects.begin(); iter != objects.end(); iter++) {
		auto obj = std::get<0>(*iter);
		auto name = std::get<1>(*iter);
		int len = obj.getGrid().getTotalCount();
		total_cnt++;

		pMxArray1 = mxCreateDoubleMatrix(len, 1, mxREAL);
		pMxArray2 = mxCreateDoubleMatrix(len, 1, mxREAL);
		if (!pMxArray1 || !pMxArray2) {
			error_code = -2; break;
		}

		pData1 = (double*)mxCalloc(len, sizeof(double));
		pData2 = (double*)mxCalloc(len, sizeof(double));
		if (!pData1 || !pData2) {
			error_code = -3; break;
		}

		double norm = 0;
		for (int i = 0; i < len; i++) {
			pData1[i] = obj.getGrid().index(i).x();
			switch (mode) {
			case NORM_MODE:
				pData2[i] = sqrt(pow(obj.getValueByIndex(i).real(), 2) + pow(obj.getValueByIndex(i).imag(), 2));
				break;
			case REAL_MODE:
				pData2[i] = obj.getValueByIndex(i).real();
				break;
			case IMAG_MODE:
				pData2[i] = obj.getValueByIndex(i).imag();
				break;
			}
		}

		if (cnt_map.count(name) == 0) {
			cnt_map.insert(std::pair<std::string, int>(name, 1));
		}
		else if (cnt_map.count(name) == 1) {
			cnt_map[name] += 1;
		}

		sprintf_s(name_str_1, "%s_x_%d", name.c_str(), cnt_map[name]);
		sprintf_s(name_str_2, "%s_y_%d", name.c_str(), cnt_map[name]);

		mxSetData(pMxArray1, pData1);
		mxSetData(pMxArray2, pData2);
		matPutVariable(pMatFile, name_str_1, pMxArray1);
		matPutVariable(pMatFile, name_str_2, pMxArray2);

		mxFree(pData1);
		mxFree(pData2);
	}

	for (auto iter = cnt_map.begin(); iter != cnt_map.end(); iter++) {
		pMxCount = mxCreateDoubleScalar((double)iter->second);
		sprintf_s(name_str_1, "%s_cnt", iter->first.c_str());
		matPutVariable(pMatFile, name_str_1, pMxCount);
	}

	pMxCount = mxCreateDoubleScalar((double)total_cnt);
	matPutVariable(pMatFile, "total", pMxCount);

	matClose(pMatFile);
	return 0;
}

int WaveAnimation2DToMat(const std::vector< WaveMsg<2> >& objects, int mode, int coordinate)
{
	MATFile* pMatFile = NULL;
	mxArray* pMxArray1 = NULL, * pMxArray2 = NULL, * pMxArray3 = NULL;
	mxArray* pMxCount = NULL;
	double* pData1 = NULL, * pData2 = NULL, *pData3 = NULL;
	char name_str_1[10], name_str_2[10], name_str_3[10];
	int error_code = 0;  //0: normal
	std::map<std::string, int> cnt_map;

	pMatFile = matOpen("test_mat_file.mat", "w");
	if (pMatFile == NULL) {
		error_code = -1;
		return error_code;
	}

	int k = 0;
	for (auto iter = objects.begin(); iter != objects.end(); iter++) {
		auto obj = std::get<0>(*iter);
		auto name = std::get<1>(*iter);
		int len1 = obj.getGrid().getCount(0);
		int len2 = obj.getGrid().getCount(1);
		k++;

		pMxArray1 = mxCreateDoubleMatrix(len1, len2, mxREAL);
		pMxArray2 = mxCreateDoubleMatrix(len1, len2, mxREAL);
		pMxArray3 = mxCreateDoubleMatrix(len1, len2, mxREAL);
		if (!pMxArray1 || !pMxArray2 || !pMxArray3) {
			error_code = -2; break;
		}

		pData1 = (double*)mxCalloc(len1 * len2, sizeof(double));
		pData2 = (double*)mxCalloc(len1 * len2, sizeof(double));
		pData3 = (double*)mxCalloc(len1 * len2, sizeof(double));

		if (!pData1 || !pData2 || !pData3) {
			error_code = -3; break;
		}

		for (int i = 0; i < len1 * len2; i++) {
			if (mode == NORM_MODE) {
				pData1[i] = sqrt(pow(obj.getValueByIndex(i).real(), 2) + pow(obj.getValueByIndex(i).imag(), 2));
			}
			else if (mode == REAL_MODE) {
				pData1[i] = obj.getValueByIndex(i).real();
			}
			else if (mode == IMAG_MODE) {
				pData1[i] = obj.getValueByIndex(i).imag();
			}

			if (coordinate == CARTESIAN_COORDINATE) {
				pData2[i] = obj.getGrid().index(i).x();
				pData3[i] = obj.getGrid().index(i).y();
			}
			else if (coordinate == POLAR_COORDINATE) {
				// x = r cos(phi)
				// y = r sin(phi)
				auto r = obj.getGrid().index(i).x();
				auto phi = obj.getGrid().index(i).y();
				pData2[i] = r * std::cos(phi);
				pData3[i] = r * std::sin(phi);
			}
		}

		if (cnt_map.count(name) == 0) {
			cnt_map.insert(std::pair<std::string, int>(name, 1));
		}
		else if (cnt_map.count(name) == 1) {
			cnt_map[name] += 1;
		}

		sprintf_s(name_str_1, "%s_z_%d", name.c_str(), cnt_map[name]);
		sprintf_s(name_str_2, "%s_x_%d", name.c_str(), cnt_map[name]);
		sprintf_s(name_str_3, "%s_y_%d", name.c_str(), cnt_map[name]);

		mxSetData(pMxArray1, pData1);
		mxSetData(pMxArray2, pData2);
		mxSetData(pMxArray3, pData3);
		matPutVariable(pMatFile, name_str_1, pMxArray1);
		matPutVariable(pMatFile, name_str_2, pMxArray2);
		matPutVariable(pMatFile, name_str_3, pMxArray3);
		mxFree(pData1);
		mxFree(pData2);
		mxFree(pData3);
	}

	for (auto iter = cnt_map.begin(); iter != cnt_map.end(); iter++) {
		pMxCount = mxCreateDoubleScalar((double)iter->second);
		sprintf_s(name_str_1, "%s_cnt", iter->first.c_str());
		matPutVariable(pMatFile, name_str_1, pMxCount);
	}

	pMxCount = mxCreateDoubleScalar((double)k);
	matPutVariable(pMatFile, "total", pMxCount);

	matClose(pMatFile);
	return 0;
}

int WaveAnimation3DToMat(const std::vector< WaveMsg<3> >& objects, int mode, int coordinate)
{
	MATFile* pMatFile = NULL;
	mxArray* pMxArray1 = NULL;
	mxArray* pMxCount = NULL;
	double* pData1 = NULL, * pre_alloc_pData1 = NULL;
	char name_str_1[10];
	int error_code = 0;  //0: normal
	std::map<std::string, int> cnt_map;
	const double MINIMAL_DISPLAY_THRESHOLD = 1e-4;

	pMatFile = matOpen("test_mat_file.mat", "w");
	if (pMatFile == NULL) {
		error_code = -1;
		return error_code;
	}

	int total_cnt = 0;
	for (auto iter = objects.begin(); iter != objects.end(); iter++) 
	{
		auto obj = std::get<0>(*iter);
		auto name = std::get<1>(*iter);
		auto interval = std::get<2>(*iter);

		int total_N = obj.getGrid().getTotalCount();
		int Nx = obj.getGrid().getCount(0);
		int Ny = obj.getGrid().getCount(1);
		int Nz = obj.getGrid().getCount(2);
		int rest_allocated_size = (int)(Nx / interval) * (int)(Ny / interval) * (int)(Nz / interval);

		total_cnt ++;

		pre_alloc_pData1 = (double*)mxCalloc(4 * rest_allocated_size, sizeof(double));

		if (!pre_alloc_pData1) {
			error_code = -3; break;
		}

		int phead = 0;

		for (int i = 0; i < Nx; i += interval) {
			for (int j = 0; j < Ny; j += interval) {
				for (int k = 0; k < Nz; k += interval) {
					int index = obj.getGrid().shrink(std::vector<int>{i, j, k});
					if (std::norm(obj.getValueByIndex(index)) < MINIMAL_DISPLAY_THRESHOLD) {
						continue;
					}
					if (mode == NORM_MODE) {
						pre_alloc_pData1[phead * 4 + 3] = std::norm(obj.getValueByIndex(index));
					}
					else if (mode == REAL_MODE) {
						pre_alloc_pData1[phead * 4 + 3] = obj.getValueByIndex(index).real();
					}
					else if (mode == IMAG_MODE) {
						pre_alloc_pData1[phead * 4 + 3] = obj.getValueByIndex(index).imag();
					}
					if (coordinate == CARTESIAN_COORDINATE) {
						pre_alloc_pData1[phead * 4] = obj.getGrid().index(index).x();
						pre_alloc_pData1[phead * 4 + 1] = obj.getGrid().index(index).y();
						pre_alloc_pData1[phead * 4 + 2] = obj.getGrid().index(index).z();
					}
					else if (coordinate == SPHERE_COORDINATE) {
						// x = rsin(theta)cos(phi)
						// y = rsin(theta)sin(phi)
						// z = rcos(theta)
						auto r = obj.getGrid().index(index).x();
						auto theta = obj.getGrid().index(index).y();
						auto phi = obj.getGrid().index(index).z();
						pre_alloc_pData1[phead * 4] = r * std::sin(theta) * std::cos(phi);
						pre_alloc_pData1[phead * 4 + 1] = r * std::sin(theta) * std::sin(phi);
						pre_alloc_pData1[phead * 4 + 2] = r * std::cos(theta);
					}
					phead += 1;
				}
			}
		}
		
		std::cout << phead << " " << rest_allocated_size << std::endl;

		pData1 = (double*)mxCalloc(4 * phead, sizeof(double));
		if (!pData1) {
			error_code = -3; break;
		}

		for (int i = 0; i < (4 * phead); i++) {
			pData1[i] = pre_alloc_pData1[i];
		}

		pMxArray1 = mxCreateDoubleMatrix(4, phead, mxREAL);
		if (!pMxArray1) {
			error_code = -2; break;
		}

		if (cnt_map.count(name) == 0) {
			cnt_map.insert(std::pair<std::string, int>(name, 1));
		}
		else if (cnt_map.count(name) == 1) {
			cnt_map[name] += 1;
		}

		sprintf_s(name_str_1, "%s_3d_%d", name.c_str(), cnt_map[name]);

		mxSetData(pMxArray1, pData1);
		matPutVariable(pMatFile, name_str_1, pMxArray1);

		//mxDestroyArray(pMxArray1);
		mxFree(pData1);
		mxFree(pre_alloc_pData1);
	}

	for (auto iter = cnt_map.begin(); iter != cnt_map.end(); iter++) {
		pMxCount = mxCreateDoubleScalar((double)iter->second);
		sprintf_s(name_str_1, "%s_cnt", iter->first.c_str());
		matPutVariable(pMatFile, name_str_1, pMxCount);
	}

	pMxCount = mxCreateDoubleScalar((double)total_cnt);
	matPutVariable(pMatFile, "total", pMxCount);

	mxDestroyArray(pMxCount);
	matClose(pMatFile);
	return 0;
}