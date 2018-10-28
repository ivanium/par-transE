#ifndef UTIL_H
#define UTIL_H

#include <cmath>
#include "parallel.h"

typedef unsigned long long ULL;

const floatT pi = 3.141592653589793238462643383;

inline ULL simple_hash(ULL *_seed) {
  ULL seed = *_seed;
  seed = (seed+0x7ed55d16) + (seed<<12);
  seed = (seed^0xc761c23c) ^ (seed>>19);
  seed = (seed+0x165667b1) + (seed<<5);
  seed = (seed+0xd3a2646c) ^ (seed<<9);
  seed = (seed+0xfd7046c5) + (seed<<3);
  seed = (seed^0xb55a4f09) ^ (seed>>16);

  *_seed = seed;
  return seed;
}

inline int rand_max(ULL *_seed, int max) {
  int res = ((unsigned int)simple_hash(_seed) & (((unsigned int)1<<31) - 1)) % max;
  return res;
}

inline floatT bounded_rand(ULL *_seed, floatT min, floatT max) {
  return min + (max - min) * simple_hash(_seed) / (UINT64_MAX + 1.0);
}

inline floatT normal(floatT x, floatT miu, floatT sigma) {
  return 1.0 / (sqrt(2*pi) * sigma * exp((x-miu)*(x-miu)/(2*sigma*sigma)));
}

inline floatT randn(ULL *_seed, floatT miu, floatT sigma, floatT min, floatT max) {
  floatT x, y, dScope;
  do {
    x = bounded_rand(_seed, min, max);
    y = normal(x, miu, sigma);
    dScope = bounded_rand(_seed, 0.0, normal(miu, miu, sigma));
  } while (dScope > y);
  return x;
}

inline void norm (floatT *vec, int _dimension) {
  floatT x = 0.0;
  for (int i = 0; i < _dimension; i++) {
    x += vec[i]*vec[i];
  }
  x = sqrt(x);
  if (x > 1) {
    for (int i = 0; i < _dimension; i++) {
      vec[i] /= x;
    }
  }
}

#endif // !UTIL_H