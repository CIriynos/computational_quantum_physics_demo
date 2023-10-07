cmake_minimum_required(VERSION 3.15)

project(CQP_DEMO VERSION 1.0.2)
configure_file(TestConfig.h.in TestConfig.h)
string(TIMESTAMP COMPILE_TIME %Y%m%d-%H%M%S)

find_package(OpenMP REQUIRED)

add_compile_options(-Werror -Wunknown-pragmas -fopenmp)
add_link_options(-lfftw3_omp -lfftw3 -lm -fopenmp)

set(SRC_LIST 
    ./src/main.cpp 
)

set(INC_DIR /usr/local/include)
set(LINK_DIR /usr/local/lib)

add_executable(${PROJECT_NAME} ${SRC_LIST})

include_directories(${INC_DIR})
link_directories(${LINK_DIR})

target_link_libraries(
    ${PROJECT_NAME} PUBLIC 
    fftw3
    fftw3_omp
    OpenMP::OpenMP_CXX
)

target_include_directories(
    ${PROJECT_NAME} PUBLIC 
    ${PROJECT_BINARY_DIR} 
    ${PROJECT_SOURCE_DIR}/include
)
