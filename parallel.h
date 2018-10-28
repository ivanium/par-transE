#ifndef PARALLEL_H
#define PARALLEL_H

#include <cstdint>

#define BIT32 1

#if BIT32
typedef int32_t intT;
typedef float   floatT;
#else
typedef int64_t intT;
typedef double  floatT;
#endif

#endif // !PARALLEL_H 