// ignore this file, this will be a lot in commit history, do not worry about
// this. peace out.
#include "../src/bare.h"

int main() {
  Memory *mem = create_global_mem(1 << 20);

  Tensor *a = tensor_zeros(mem, S(1, 3), 2, PERM);
  Tensor *t = transpose_t(mem, a);
  Tensor *b = tensor_full_like(mem, t, 4.0, TEMP);
  Tensor *out = add_t(mem, b, t);

  print_t(out, 0);

  backward(mem, out);

  print_t(a, 1);

  reset_temp_mem(mem);
  free_global_mem(mem);
  return 0;
}
