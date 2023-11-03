#ifndef __CQP_TIME_TEST_H__
#define __CQP_TIME_TEST_H__

#include <omp.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include "cqp_util.hpp"

#ifdef _USING_OMP
extern std::vector<std::unordered_map<std::string, double>> _GLOBAL_TIME_RECODE;
extern std::vector<std::unordered_map<std::string, int>> _GLOBAL_CALL_TIMES;

#else
extern std::unordered_map<std::string, double>> _GLOBAL_TIME_RECODE;
extern std::unordered_map<std::string, int>> _GLOBAL_CALL_TIMES;

#endif //_USING_OMP

#ifdef _USING_OMP

#define TIME_TEST(expr, expr_name) \
{ \
	double _c_time_begin_##expr_name = omp_get_wtime() * 1000; \
	(expr); \
	double _c_time_end_##expr_name = omp_get_wtime() * 1000; \
	int _thread_num_##expr_name = omp_get_thread_num();	\
	_GLOBAL_TIME_RECODE[_thread_num_##expr_name][#expr_name] += _c_time_end_##expr_name - _c_time_begin_##expr_name; \
    _GLOBAL_CALL_TIMES[_thread_num_##expr_name][#expr_name] += 1; \
}

#define TIME_TEST_START(expr_name)	\
	double _c_time_begin_##expr_name = omp_get_wtime() * 1000;

#define TIME_TEST_END(expr_name)	\
	double _c_time_end_##expr_name = omp_get_wtime() * 1000; \
	int _thread_num_##expr_name = omp_get_thread_num();	\
	_GLOBAL_TIME_RECODE[_thread_num_##expr_name][#expr_name] += _c_time_end_##expr_name - _c_time_begin_##expr_name; \
    _GLOBAL_CALL_TIMES[_thread_num_##expr_name][#expr_name] += 1; \

#else

#define TIME_TEST(expr, expr_name) \
{	\
	auto t_start##expr_name = std::chrono::high_resolution_clock::now();	\
	(expr);	\
	auto t_end##expr_name = std::chrono::high_resolution_clock::now();	\
	_GLOBAL_TIME_RECODE[#expr_name] += std::chrono::duration<double, std::milli>(t_end##expr_name - t_start##expr_name).count(); \
    _GLOBAL_CALL_TIMES[#expr_name] += 1; \
}

#define TIME_TEST_START(expr_name)	\
	auto t_start##expr_name = std::chrono::high_resolution_clock::now();

#define TIME_TEST_END(expr_name)	\
	auto t_end##expr_name = std::chrono::high_resolution_clock::now();	\
	_GLOBAL_TIME_RECODE[#expr_name] += std::chrono::duration<double, std::milli>(t_end##expr_name - t_start##expr_name).count(); \
    _GLOBAL_CALL_TIMES[#expr_name] += 1;

#endif //_USING_OMP

//void init_time_test();

//void display_time_test_report();

void display_time_test_report(std::initializer_list<std::string> names_list);

#endif //__CQP_TIME_TEST_H__