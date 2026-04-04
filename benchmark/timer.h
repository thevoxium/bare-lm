#ifndef TIMER_H
#define TIMER_H

#include "../src/bare.h"
#include <stdio.h>
#include <time.h>

#define PRINT_TIME(S, N, T) printf("%s, N = %zu, time -> %.3f ms\n", S, N, T);
#define ITER 100

typedef struct Timer {
  struct timespec start;
  struct timespec end;
} Timer;

static inline void timer_start(Timer *t) {
  clock_gettime(CLOCK_MONOTONIC, &t->start);
}

static inline double timer_stop(Timer *t) {
  clock_gettime(CLOCK_MONOTONIC, &t->end);
  double sec = t->end.tv_sec - t->start.tv_sec;
  double nsec = t->end.tv_nsec - t->start.tv_nsec;
  return sec * 1000.0 + nsec / 1000000.0;
}

#endif // !TIMER_H
