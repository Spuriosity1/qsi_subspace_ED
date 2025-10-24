#pragma once
#include <chrono>

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
#define BENCH_TIMEIT(msg,  ...) \
     TIMEIT_PRINT_HELPER(msg, __COUNTER__,  __VA_ARGS__)
#else 
#define BENCH_TIMEIT(msg, ...) __VA_ARGS__
#endif

