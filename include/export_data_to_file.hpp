#ifndef __EXPORT_DATA_TO_FILE_H__
#define __EXPORT_DATA_TO_FILE_H__

#include <Eigen/Dense>
#include "numerical_grid.hpp"
#include "wave_function.hpp"
#include <string>
#include <iostream>
#include <fstream>

constexpr int XYZ = 0;
constexpr int XY = 0;
constexpr int POLAR = 1;
constexpr int SPHERICAL = 2;

constexpr int ABS_MODE = 0;
constexpr int REAL_MODE = 1;
constexpr int IMAG_MODE = 2;

namespace CQP {

template<unsigned N>
inline void export_wave_to_file (
    const NumericalGrid<N>&         grid,
    const WaveData&                 wave,
    std::string                     name,
    int                             grid_mode,
    int                             num_mode = ABS_MODE,
    
    int interval = 1, double minimal_threshold = 1e-8
    )
{
    std::fstream s;
    std::string pathToOutput = "../output/";
    s.open(pathToOutput + name + ".wavedata", std::ios::out);

    // Firstly, statistics the count of points to be exported.
    int total_count = 0;
    for (int i = 0; i < grid.getTotalCount(); i += 1) {
        double value = std::norm(wave(i));
        if(value >= minimal_threshold || grid_mode != SPHERICAL) total_count += 1;
    }
    
    //write metadata -> (name, dimension, total_points_number, grid_mode, [count(0), count(1), ...])
    //grid_mode = XYZ | POLAR | SPHERICAL
    s << name << " ";
    s << N << " ";
    s << total_count << " ";
    s << grid_mode << " ";
    for (int i = 0; i < N; i++) {
        s << grid.getCount(i) << " ";
    }
    s << std::endl << std::endl;

    for(int i = 0; i < grid.getTotalCount(); i += 1){
        CQP::GridPoint<N> point = grid.index(i);

        double value = 0.0;
        if(num_mode == ABS_MODE){
            value = std::sqrt(std::norm(wave(i)));
        }
        else if(num_mode == REAL_MODE){
            value = wave(i).real();
        }
        else if(num_mode == IMAG_MODE){
            value = wave(i).imag();
        }

        if(value < minimal_threshold && grid_mode == SPHERICAL) continue;

        if (grid_mode == XY || grid_mode == XYZ) {
            for(int j = 0; j < N; j++){
                s << point.get(j) << " ";
            }
            s << value << " " << std::endl;
        }
        else if (grid_mode == SPHERICAL && N == 3) {
            double r = point.get(0);
            double theta = point.get(1);
            double phi = point.get(2);
            s << r * sin(theta) * cos(phi) << " ";
            s << r * sin(theta) * sin(phi) << " ";
            s << r * cos(theta) << " ";
            s << value << " " << std::endl;
        }
        else if (grid_mode == POLAR && N == 2) { 
            double r = point.get(0);
            double phi = point.get(1);
            s << r << " ";
            s << phi << " ";
            s << value << " " << std::endl;
        }
    }

    s.close();
    char logstr[100];
    sprintf(logstr, "Export data over : %s [With count=%d]", name.c_str(), total_count);
    std::cout << logstr << std::endl;
}

}
#endif