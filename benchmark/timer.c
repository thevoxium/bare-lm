#include "../src/bare.h"
#include <stdio.h>
#include <time.h>

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

int main(void) {
  Memory *mem = create_global_mem(1 << 30);

  int N = 1024;
  int shape_a[] = {N, N};
  int shape_b[] = {N, N};
  Tensor *a = tensor_randn(mem, shape_a, 2, TEMP);
  Tensor *b = tensor_randn(mem, shape_b, 2, TEMP);

  Timer t;
  timer_start(&t);
  Tensor *c = matmul_t(mem, a, b);
  double elapsed = timer_stop(&t);

  printf("matmul (%d,%d) @ (%d,%d): %.3f ms\n", shape_a[0], shape_a[1],
         shape_b[0], shape_b[1], elapsed);

  free_global_mem(mem);
  return 0;
}
