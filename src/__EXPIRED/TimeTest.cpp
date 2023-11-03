#include "TimeTest.h"
#include <string>

std::map<std::string, double> _GLOBAL_TIME_RECODE;
std::map<std::string, int> _GLOBAL_CALL_TIMES;

void display_report()
{
    int i = 0;
    char output_string[100];
    std::cout << " ----- Time Test Report -----" << std::endl;
    for(auto iter = _GLOBAL_TIME_RECODE.begin(); iter != _GLOBAL_TIME_RECODE.end(); iter ++){
        std::string name = iter->first;
        double average_time = iter->second / _GLOBAL_CALL_TIMES[name];
        double total_time = iter->second;
        sprintf(output_string, "[%d] Time expense in {%s} : \t%6.3f(average, ms) \t%6.3f(total)", i, name.c_str(), average_time, total_time);
        std::cout << output_string << std::endl;
        i++;
    }
    
    i = 0;
    std::cout << " ------ Calling Report ------" << std::endl;
    for(auto iter = _GLOBAL_TIME_RECODE.begin(); iter != _GLOBAL_TIME_RECODE.end(); iter ++){
        std::string name = iter->first;
        int call_times = _GLOBAL_CALL_TIMES[name];
        sprintf(output_string, "[%d] Calling Times for {%s}:\t %d", i, name.c_str(), call_times);
        std::cout << output_string << std::endl;
        i++;
    }
}