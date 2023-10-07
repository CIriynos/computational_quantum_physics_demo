#ifndef __EXPORT_DATA_TO_MAT_FILE_H__
#define __EXPORT_DATA_TO_MAT_FILE_H__

#include <vector>
#include "Wavefunction.hpp"

constexpr int NORM_MODE = 1;
constexpr int REAL_MODE = 2;
constexpr int IMAG_MODE = 3;

constexpr int CARTESIAN_COORDINATE = 1;
constexpr int SPHERE_COORDINATE = 2;
constexpr int POLAR_COORDINATE = 3; //2d

template<unsigned N>
using WaveMsg = std::tuple<CQP::Wavefunction<N>, std::string, int>;

int SingleWave1DToMat(const CQP::Wavefunction1D&, int);

int WaveAnimation1DToMat(const std::vector< WaveMsg<1> >&, int);

int WaveAnimation2DToMat(const std::vector< WaveMsg<2> >&, int mode, int coordinate = CARTESIAN_COORDINATE);

int WaveAnimation3DToMat(const std::vector< WaveMsg<3> >& objects, int mode, int coordinate = CARTESIAN_COORDINATE);

#endif // !__EXPORT_DATA_TO_MAT_FILE_H__
