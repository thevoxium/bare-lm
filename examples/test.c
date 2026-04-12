// ignore this file, this will be a lot in commit history, do not worry about
// this. peace out.
#include "../src/bare.h"

int main() {
  Memory *mem = create_global_mem(1 << 20);

  Tensor *b = tensor_randn(mem, S(3), 1, TEMP);
  Tensor *c = broadcast_t(mem, b, S(4, 3), 2);
  Tensor *d = permute_t(mem, c, S(1, 0), 2);

  backward(mem, d);
  print_t(b, 1);
  print_t(c, 0);
  print_t(d, 0);

  reset_temp_mem(mem);
  free_global_mem(mem);
  return 0;
}
