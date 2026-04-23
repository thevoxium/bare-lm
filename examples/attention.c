// WORK IN PROGRESS

#include "../src/bare.h"

Tensor *create_positional_encodings(Memory *mem, int T, int D) {
  Tensor *pe = tensor_zeros(mem, S(T, D), 2, PERM);
  Tensor *position = tensor_arange(mem, 0, T, 1, PERM);
  Tensor *div_term = tensor_arange(mem, 0, D, 2, PERM);
  div_term = scale_t(mem, div_term, -log(10000) / D);
  return pe;
}

int main() {
  Memory *mem = create_global_mem(1024 * 1024 * 1024);
  ParameterList *pl = create_param_list(mem);

  int B = 32;
  int T_src = 1024;
  int T_tgt = 1024;
  int D = 512;
  int H = 8;
  int d_k = D / H;
  int V = 512;

  free_global_mem(mem);
  return 0;
}
