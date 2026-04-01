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

typedef Tensor *(*BinaryOp)(Memory *, Tensor *, Tensor *);
typedef Tensor *(*UnaryOp)(Memory *, Tensor *);

static double run_binary_benchmark(Memory *mem, const char *name, int N, BinaryOp op) {
  double total_time = 0.0f;
  for (int i = 0; i < ITER; i++) {
    reset_temp_mem(mem);
    int shape[] = {N};
    Tensor *a = tensor_randn(mem, shape, 1, TEMP);
    Tensor *b = tensor_randn(mem, shape, 1, TEMP);
    timer_start(&t);
    volatile Tensor *r = op(mem, a, b);
    total_time += timer_stop(&t);
  }
  PRINT_TIME(name, N, total_time / ITER);
  return total_time / ITER;
}

static double run_unary_benchmark(Memory *mem, const char *name, int N, UnaryOp op) {
  double total_time = 0.0f;
  for (int i = 0; i < ITER; i++) {
    reset_temp_mem(mem);
    int shape[] = {N};
    Tensor *a = tensor_randn(mem, shape, 1, TEMP);
    timer_start(&t);
    volatile Tensor *r = op(mem, a);
    total_time += timer_stop(&t);
  }
  PRINT_TIME(name, N, total_time / ITER);
  return total_time / ITER;
}

int main(void) {
  Memory *mem = create_global_mem(1UL << 32);

  run_binary_benchmark(mem, "add_t", 1000000, add_t);
  run_binary_benchmark(mem, "sub_t", 1000000, sub_t);
  run_binary_benchmark(mem, "mul_t", 1000000, mul_t);
  run_binary_benchmark(mem, "divide_t", 1000000, divide_t);
  run_unary_benchmark(mem, "neg_t", 1000000, neg_t);

  free_global_mem(mem);
  return 0;
}