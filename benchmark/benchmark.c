#include "../src/bare.h"
#include <stdio.h>
#include <time.h>

#define PRINT_TIME(S, N, T) printf("%s, N = %d, time -> %.3f ms\n", S, N, T);
#define ITER 100

typedef struct Timer {
  struct timespec start;
  struct timespec end;
} Timer;

static void timer_start(Timer *t) { clock_gettime(CLOCK_MONOTONIC, &t->start); }

static double timer_stop(Timer *t) {
  clock_gettime(CLOCK_MONOTONIC, &t->end);
  double sec = t->end.tv_sec - t->start.tv_sec;
  double nsec = t->end.tv_nsec - t->start.tv_nsec;
  return sec * 1000.0 + nsec / 1000000.0;
}

Timer t;

static void run_add_t(Memory *mem) {
  int N = 1000000;
  double total_time = 0.0f;
  for (int i = 0; i < ITER; i++) {
    reset_temp_mem(mem);
    int shape_a[] = {N};
    int shape_b[] = {N};
    Tensor *a = tensor_randn(mem, shape_a, 1, TEMP);
    Tensor *b = tensor_randn(mem, shape_b, 1, TEMP);
    timer_start(&t);
    volatile Tensor *r = add_t(mem, a, b);
    double elapsed = timer_stop(&t);
    total_time += elapsed;
  }
  PRINT_TIME("add_t", N, total_time / ITER);
}

int main(void) {
  Memory *mem = create_global_mem(1UL << 32);

  run_add_t(mem);

  free_global_mem(mem);
  return 0;
}
