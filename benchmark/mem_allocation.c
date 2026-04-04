#include "../src/bare.h"
#include "timer.h"
#include <stdio.h>

int main() {
  Memory *mem = create_global_mem(1ULL << 31);

  Timer t;

  size_t N = 256 * 1024 * 1024;
  for (size_t i = 1; i <= N; i *= 2) {
    Timer t;

    timer_start(&t);
    volatile float *m = (float *)malloc(i * sizeof(float));
    if (m == NULL) {
      fprintf(stderr, "Malloc failed allocation for %zu", i);
      return 1;
    }
    size_t step = 4096 / sizeof(float);
    for (size_t j = 0; j < i; j += step) {
      m[j] = 1.0f;
    }

    volatile float sink = m[i / 2];
    free((void *)m);
    double end_time = timer_stop(&t);
    PRINT_TIME("malloc", i, end_time);

    timer_start(&t);
    volatile float *a = (float *)allocate_mem(mem, i * sizeof(float), TEMP);
    if (m == NULL) {
      fprintf(stderr, "Malloc failed allocation for %zu", i);
      return 1;
    }
    for (size_t j = 0; j < i; j += step) {
      a[j] = 1.0f;
    }

    sink = a[i / 2];
    reset_temp_mem(mem);

    end_time = timer_stop(&t);
    PRINT_TIME("allocate_mem", i, end_time);
  }

  free_global_mem(mem);
}
