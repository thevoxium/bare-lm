#include "../src/bare.h"

int main() {
  Memory *mem = create_global_mem(1 << 20);

  Tensor *a = tensor_init(mem, S(1, 3), 2, PERM);
  print_t(a, 0);

  Tensor *t = transpose_t(mem, a);
  print_t(t, 0);

  backward(mem, t);

  reset_temp_mem(mem);
  free_global_mem(mem);
  return 0;
}
