#ifndef __EXPORT_DATA_TO_FILE_H__
#define __EXPORT_DATA_TO_FILE_H__

#include "Wavefunction.hpp"
#include <string>
#include <iostream>
#include <fstream>

constexpr int XYZ = 0;
constexpr int XY = 0;
constexpr int POLAR = 1;
constexpr int SPHERICAL = 2;

template<unsigned N>
inline void exportWaveToFile(const CQP::Wavefunction<N>& wave, std::string name, int grid_mode, int interval = 1, double minimal_threshold = 1e-8)
{
    std::fstream s;
    std::string pathToOutput = "../output/";
    s.open(pathToOutput + name + ".wavedata", std::ios::out);

    // Firstly, statistics the count of points to be exported.
    int total_count = 0;
    for (int i = 0; i < wave.getGrid().getTotalCount(); i += 1) {
        double value = std::norm(wave.getValueByIndex(i));
        if(value >= minimal_threshold || grid_mode != SPHERICAL) total_count += 1;
    }
    
    //write metadata -> (name, dimension, total_points_number, grid_mode, [count(0), count(1), ...])
    //grid_mode = XYZ | POLAR | SPHERICAL
    s << name << " ";
    s << N << " ";
    s << total_count << " ";
    s << grid_mode << " ";
    for (int i = 0; i < N; i++) {
        s << wave.getGrid().getCount(i) << " ";
    }
    s << std::endl << std::endl;

    for(int i = 0; i < wave.getGrid().getTotalCount(); i += 1){
        CQP::GridPoint<N> point = wave.getGrid().index(i);
        double value = std::norm(wave.getValueByIndex(i));
        if(value < minimal_threshold && grid_mode == SPHERICAL) continue;

        if (grid_mode == XY || grid_mode == XYZ) {
            for(int j = 0; j < N; j++){
                s << point.get(j) << " ";
            }
            s << std::norm(wave.getValueByIndex(i)) << " " << std::endl;
        }
        else if (grid_mode == SPHERICAL) {
            double r = point.get(0);
            double theta = point.get(1);
            double phi = point.get(2);
            s << r * sin(theta) * cos(phi) << " ";
            s << r * sin(theta) * sin(phi) << " ";
            s << r * cos(theta) << " ";
            s << std::norm(wave.getValueByIndex(i)) << " " << std::endl;
        }
        else if (grid_mode == POLAR) { 
            double r = point.get(0);
            double phi = point.get(1);
            s << r << " ";
            s << phi << " ";
            s << std::norm(wave.getValueByIndex(i)) << " " << std::endl;
        }
    }

    s.close();
    char logstr[100];
    sprintf(logstr, "Export data over : %s [With count=%d]", name.c_str(), total_count);
    std::cout << logstr << std::endl;
}


#endif