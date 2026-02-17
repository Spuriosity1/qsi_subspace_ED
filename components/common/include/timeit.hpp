#pragma once
#include <chrono>
#include <iostream>

// we must use two marco to combine the variable name with __LINE__, eg. if we just write 'name##__LINE__'
// the '##' will take place before __LINE__
// Because the preprocessor will only expand the macros recursively if neither the stringizing operator #
// nor the token-pasting operator ## are applied to it
// see more: https://stackoverflow.com/questions/1597007/creating-c-macro-with-and-line-token-concatenation-with-positioning-macr
#define COMBINE_HELPER(X,Y) X##Y  // helper macro
#define COMBINE(X,Y) COMBINE_HELPER(X,Y)

#define TIMEIT_PRINT_HELPER(msg, line_num, ...)                                                                        \
    auto COMBINE(timeit_start,line_num) =std::chrono::high_resolution_clock::now();                                    \
    __VA_ARGS__                                                                                                        \
    auto COMBINE(timeit_end, line_num) = std::chrono::high_resolution_clock::now();                                    \
    auto COMBINE(timeit_duration, line_num) =                                                                          \
         std::chrono::duration<double, std::milli>(COMBINE(timeit_end,line_num) - COMBINE(timeit_start,line_num));     \
    std::cout << msg << " execution time: " <<  COMBINE(timeit_duration, line_num).count() << " ms"<< std::endl;

#ifndef DISABLE_TIMEIT
// use __COUNTER__ instead of __LINE__ to make the TIMEIT marco able to be nested
#define TIMEIT(msg,  ...) \
     TIMEIT_PRINT_HELPER(msg, __COUNTER__,  __VA_ARGS__)
#else
#define TIMEIT(msg, ...) __VA_ARGS__
#endif

#ifdef SUBSPACE_ED_BENCHMARK_OPERATIONS
#define BENCH_TIMER_TIMEIT(timer,  ...) \
     timer.tik();\
     __VA_ARGS__\
     timer.tok();

#define BENCH_TIMEIT(msg,  ...) \
     TIMEIT_PRINT_HELPER(msg, __COUNTER__,  __VA_ARGS__)
#else
#define BENCH_TIMER_TIMEIT(timer, ...) __VA_ARGS__
#define BENCH_TIMEIT(timer, ...) __VA_ARGS__
#endif



#ifdef SUBSPACE_ED_BENCHMARK_OPERATIONS
#define BENCH_PRINT_VECTOR(msg,  v) \
    std::cout<<msg<<": "; \
    for (const auto& vi : v){std::cout<<vi<<", ";} \
    std::cout<<"\n";
#else 
#define BENCH_PRINT_VECTOR(msg, v)
#endif

#include <sstream>

class Timer {
    using Units=std::milli;
    std::chrono::time_point<std::chrono::high_resolution_clock> tik_time;
    std::vector<double> times;
    std::string name;
    public:
    Timer(const std::string& name_): name(name_){}
    Timer(const std::string& name_, int rank){
        std::ostringstream oss;
        oss << "[rank "<<rank<<"] "<<name_;
        name = oss.str();
    }

        void tik(){
            tik_time = std::chrono::high_resolution_clock::now();
        }
        auto tok(){
            auto diff = 
                    std::chrono::duration<double, Units>(
                    std::chrono::high_resolution_clock::now() - tik_time)
                    ;
            times.push_back(diff.count());
            std::cerr<<name<<diff.count()<<"ms"<<std::endl;
            return diff;
        }

        void print_summary(std::ostream& o) const{
            double acc = 0;
            double acc2 = 0;
            double min = std::numeric_limits<double>::max();
            double max = std::numeric_limits<double>::min();
            for ( auto t: times){
                acc += t;
                acc2 += t*t;
                min = std::min(min, t);
                max = std::max(max, t);
            }
            double avg = acc / times.size();
            if (times.size() > 1){
            o << name <<"\n"
                <<"--------------------------------\n"
                <<"\t"<<times.size()<<" samples\n"
                <<"\tavg time " << avg <<"ms"
                <<" +- " << sqrt(acc2 / times.size() - avg*avg) <<"ms\n"
                <<"\tmin time " << min <<"ms\n"
                <<"\tmax time " << max <<"ms\n"
                <<"--------------------------------\n";
            } else {
                o << name <<"\t "<<avg<<"ms\n";
            }
        }
};
