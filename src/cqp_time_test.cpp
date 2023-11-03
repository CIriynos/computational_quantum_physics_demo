#include "cqp_time_test.h"
#include <string>
#include <algorithm>

#ifdef _USING_OMP
std::vector<std::unordered_map<std::string, double>> _GLOBAL_TIME_RECODE(THREAD_NUM_CQP);
std::vector<std::unordered_map<std::string, int>> _GLOBAL_CALL_TIMES(THREAD_NUM_CQP);

#else
std::map<std::string, double>> _GLOBAL_TIME_RECODE;
std::map<std::string, int>> _GLOBAL_CALL_TIMES;

#endif

void display_time_test_report(std::initializer_list<std::string> name_list)
{
    typedef std::pair<std::string, double> pair_t;
    std::unordered_map<std::string, double> total_time_record;
    std::unordered_map<std::string, int> total_call_times;
    std::unordered_map<std::string, int> name_list_map;
    std::vector<pair_t> time_record_buffer;

    for(auto &u: name_list){
        name_list_map[u] = 1;
    }

#ifdef _USING_OMP
    std::unordered_map<std::string, int> thread_occupied_num;

    for(auto &map_in_thread: _GLOBAL_TIME_RECODE){
        for(auto &p: map_in_thread){
            total_time_record[p.first] += p.second;
        }
    }
    
    for(auto &map_in_thread: _GLOBAL_CALL_TIMES){
        for(auto &p: map_in_thread){
            total_call_times[p.first] += p.second;
            thread_occupied_num[p.first] += 1;
        }
    }

    // for(auto &p: total_time_record){
    //     total_time_record[p.first] /= thread_occupied_num[p.first];
    // }

#else
    total_time_record = _GLOBAL_TIME_RECODE;
    total_call_times = _GLOBAL_CALL_TIMES;

#endif  //_USING_OMP

    for(auto iter = total_time_record.begin(); iter != total_time_record.end(); iter ++){
        time_record_buffer.push_back(pair_t(iter->first, iter->second));
    }
    std::sort(time_record_buffer.begin(), time_record_buffer.end(),
        [](const pair_t& a, const pair_t& b){
            return (a.second) > (b.second);
        });

    int i = 0;
    char output_string[100];
    std::cout << " ----- Time Test Report -----" << std::endl;
    for(auto iter = time_record_buffer.begin(); iter != time_record_buffer.end(); iter ++){
        std::string display_name = "{" + iter->first + "}";
        if(name_list.size() != 0 && name_list_map.find(iter->first) == name_list_map.end()) continue;
        double average_time = (iter->second / total_call_times[iter->first]);
        double total_time = iter->second;
        sprintf(output_string, "[%d] Time expense in %-12s \t\t: %-8.3f (average, ms) \t %-8.3f (total, ms)", i, display_name.c_str(), average_time, total_time);
        std::cout << output_string << std::endl;
        i++;
    }
    
    i = 0;
    std::cout << " ------ Calling Report ------" << std::endl;
    for(auto iter = total_time_record.begin(); iter != total_time_record.end(); iter ++){
        std::string display_name = "{" + iter->first + "}";
        if(name_list.size() != 0 && name_list_map.find(iter->first) == name_list_map.end()) continue;
        int call_times = total_call_times[iter->first];
        sprintf(output_string, "[%d] Calling Times for %-12s \t\t: %-d", i, display_name.c_str(), call_times);
        std::cout << output_string << std::endl;
        i++;
    }

#ifdef _USING_OMP
    i = 0;
    std::cout << " ------ Thread Occupation Report ------" << std::endl;
    for(auto iter = thread_occupied_num.begin(); iter != thread_occupied_num.end(); iter ++){
        std::string display_name = "{" + iter->first + "}";
        if(name_list.size() != 0 && name_list_map.find(iter->first) == name_list_map.end()) continue;
        int thread_num = iter->second;
        sprintf(output_string, "[%d] Threads num for %-12s \t\t: %-d", i, display_name.c_str(), thread_num);
        std::cout << output_string << std::endl;
        i++;
    }
#endif //_USING_OMP
}