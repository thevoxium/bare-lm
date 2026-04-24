// WORK IN PROGRESS

#include "../src/bare.h"

Tensor *create_positional_encodings(Memory *mem, int T, int D) {
  CHECK(D % 2 == 0, "create_positional_encodings: D should be even");

  Tensor *pe = tensor_zeros(mem, S(T, D), 2, PERM);
  Tensor *position = tensor_arange(mem, 0, T, 1, TEMP);
  position = unsqueeze_t(mem, position, 1);
  Tensor *div_term = tensor_arange(mem, 0, D, 2, TEMP);
  div_term = scale_t(mem, div_term, (-log(10000)) / D);
  div_term = exp_t(mem, div_term);
  position = broadcast_t(mem, position, S(T, D / 2), 2);
  div_term = broadcast_t(mem, div_term, S(T, D / 2), 2);
  Tensor *position_div_mul = mul_t(mem, position, div_term);

  for (int i = 0; i < T; i++) {
    for (int j = 0; j < D / 2; j++) {
      pe->data[i * D + 2 * j] = sin(position_div_mul->data[i * D / 2 + j]);
      pe->data[i * D + 2 * j + 1] = cos(position_div_mul->data[i * D / 2 + j]);
    }
  }
  detach_t(pe);
  return pe;
}

Pair_T scaled_dot_product_attention(Tensor *Q, Tensor *K, Tensor *V) {
  Pair_T result;
  print_tensor_shape(Q);

  return result;
}

int main() {
  Memory *mem = create_global_mem(1024 * 1024 * 1024);
  ParameterList *pl = create_param_list(mem);

  int B = 32;
  int T = 1024;
  int D = 512;
  int H = 8;
  int d_k = D / H;
  int Vocab = 512;

  Tensor *pe = create_positional_encodings(mem, T, D);

  Tensor *Q = tensor_xavier(mem, S(B, H, T, d_k), 4, PERM);
  Tensor *K = tensor_xavier(mem, S(B, H, T, d_k), 4, PERM);
  Tensor *V = tensor_xavier(mem, S(B, H, T, d_k), 4, PERM);

  Pair_T attention_result = scaled_dot_product_attention(Q, K, V);

  free_global_mem(mem);
  return 0;
}
