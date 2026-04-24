// WORK IN PROGRESS

#include "../src/bare.h"

// #define SCALE 8
#define B 4
#define T 128
#define D 64
#define H 1
#define d_k 64
#define Vocab 64
#define D_ff 32

Tensor *create_positional_encodings(Memory *mem) {
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

Pair_T scaled_dot_product_attention(Memory *mem, Tensor *Q, Tensor *K,
                                    Tensor *V, Tensor *mask) {
  Pair_T result;

  Tensor *K_ = permute_t(mem, K, S(0, 1, 3, 2), 4);
  K_ = reshape_t(mem, K_, S(B * H, d_k, T), 3);
  Tensor *Q_ = reshape_t(mem, Q, S(B * H, T, d_k), 3);

  Tensor *scores = bmm_t(mem, Q_, K_);
  scores = reshape_t(mem, scores, S(B, H, T, T), 4);
  scores = scale_t(mem, scores, 1.0f / sqrt(1.0f * d_k));

  if (mask != NULL) {
    scores = mask_t(mem, scores, mask, -1e9);
  }

  Tensor *attn = softmax_t(mem, scores, scores->ndim - 1);
  attn = reshape_t(mem, attn, S(B * H, T, T), 3);

  Tensor *V_ = reshape_t(mem, V, S(B * H, T, d_k), 3);

  Tensor *out = bmm_t(mem, attn, V_);
  out = reshape_t(mem, out, S(B, H, T, d_k), 4);
  attn = reshape_t(mem, attn, S(B, H, T, T), 4);

  result.F = out;
  result.S = attn;
  return result;
}

Tensor *feed_forward(Memory *mem, Tensor *x, Linear *w1, Linear *w2) {
  Tensor *x_ = reshape_t(mem, x, S(B * T, D), 2);
  Tensor *out = linear_t(mem, w1, x_);
  out = relu_t(mem, out);
  out = linear_t(mem, w2, out);
  out = reshape_t(mem, out, S(B, T, D), 3);
  return out;
}

int main() {
  Memory *mem = create_global_mem(1ULL * 10 * 1024 * 1024 * 1024);
  ParameterList *pl = create_param_list(mem);

  Tensor *pe = create_positional_encodings(mem);
  reset_temp_mem(mem);

  Tensor *Q = tensor_xavier(mem, S(B, H, T, d_k), 4, PERM);
  Tensor *K = tensor_xavier(mem, S(B, H, T, d_k), 4, PERM);
  Tensor *V = tensor_xavier(mem, S(B, H, T, d_k), 4, PERM);

  Pair_T attention_result = scaled_dot_product_attention(mem, Q, K, V, NULL);

  Linear *w1 = create_linear(mem, pl, D, D_ff);
  Linear *w2 = create_linear(mem, pl, D_ff, D);

  Tensor *x = tensor_xavier(mem, S(B, T, D), 3, PERM);
  Tensor *out = feed_forward(mem, x, w1, w2);

  print_tensor_shape(out);

  free_global_mem(mem);
  return 0;
}
