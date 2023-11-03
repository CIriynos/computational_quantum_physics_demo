#ifndef __TIME_TEST_H__
#define __TIME_TEST_H__

#include <omp.h>
#include <iostream>
#include <map>

extern std::map<std::string, double> _GLOBAL_TIME_RECODE;
extern std::map<std::string, int> _GLOBAL_CALL_TIMES;

#define TIME_TEST(expr, expr_name) \
{ \
	double _c_time_begin_##expr_name = omp_get_wtime(); \
	(expr); \
	double _c_time_end_##expr_name = omp_get_wtime(); \
	_GLOBAL_TIME_RECODE[#expr_name] += _c_time_end_##expr_name - _c_time_begin_##expr_name; \
    _GLOBAL_CALL_TIMES[#expr_name] += 1; \
}

void display_report();

#endif //__TIME_TEST_H__