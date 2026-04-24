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

Tensor *generate_causal_mask(Memory *mem) {
  Tensor *mask = tensor_zeros(mem, S(B, H, T, T), 4, PERM);
  CHECK(mask, "generate_causal_mask: failed to create mask tensor");

  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      for (int i = 0; i < T; i++) {
        for (int j = i + 1; j < T; j++) {
          int idx = b * H * T * T + h * T * T + i * T + j;
          mask->data[idx] = 1.0f;
        }
      }
    }
  }
  detach_t(mask);
  return mask;
}

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

Tensor *multi_head_attention(Memory *mem, Tensor *x, Linear *wq, Linear *wk,
                             Linear *wv, Linear *wo, Tensor *mask) {
  Tensor *x_ = reshape_t(mem, x, S(B * T, D), 2);
  Tensor *Q = linear_t(mem, wq, x_);
  Tensor *K = linear_t(mem, wk, x_);
  Tensor *V = linear_t(mem, wv, x_);

  Q = reshape_t(mem, Q, S(B, T, H, d_k), 4);
  K = reshape_t(mem, K, S(B, T, H, d_k), 4);
  V = reshape_t(mem, V, S(B, T, H, d_k), 4);

  Q = permute_t(mem, Q, S(0, 2, 1, 3), 4);
  K = permute_t(mem, K, S(0, 2, 1, 3), 4);
  V = permute_t(mem, V, S(0, 2, 1, 3), 4);

  Pair_T result = scaled_dot_product_attention(mem, Q, K, V, mask);
  Tensor *out = result.F;
  out = permute_t(mem, out, S(0, 2, 1, 3), 4);
  out = reshape_t(mem, out, S(B * T, D), 2);

  Tensor *out_final = linear_t(mem, wo, out);
  out_final = reshape_t(mem, out_final, S(B, T, D), 3);

  return out_final;
}

Tensor *add_norm(Memory *mem, Tensor *x, Tensor *sub_layer, LayerNorm *norm) {
  Tensor *out = add_t(mem, x, sub_layer);
  out = layernorm_t(mem, norm, out);
  return out;
}

Tensor *encoder(Memory *mem, Tensor *x, Linear *wq, Linear *wk, Linear *wv,
                Linear *wo, Linear *w1, Linear *w2, LayerNorm *norm1,
                LayerNorm *norm2, Tensor *mask) {
  Tensor *attn_out = multi_head_attention(mem, x, wq, wk, wv, wo, mask);
  Tensor *x_ = add_norm(mem, x, attn_out, norm1);
  Tensor *ff_out = feed_forward(mem, x_, w1, w2);
  Tensor *final = add_norm(mem, x, ff_out, norm2);
  return final;
}

Tensor *decoder(Memory *mem, Tensor *x, Tensor *enc_out, Tensor *tgt_mask,
                Tensor *src_mask, Linear *wq1, Linear *wk1, Linear *wv1,
                Linear *wo1, Linear *wq2, Linear *wk2, Linear *wv2, Linear *wo2,
                Linear *w1, Linear *w2, LayerNorm *norm1, LayerNorm *norm2,
                LayerNorm *norm3) {
  Tensor *attn1 = multi_head_attention(mem, x, wq1, wk1, wv1, wo1, tgt_mask);
  Tensor *x_ = add_norm(mem, x, attn1, norm1);

  Tensor *x_new = reshape_t(mem, x, S(B * T, D), 2);

  Tensor *Q = linear_t(mem, wq2, x_new);
  Q = reshape_t(mem, Q, S(B, T, H, d_k), 4);

  Tensor *K = linear_t(mem, wk2, x_new);
  K = reshape_t(mem, K, S(B, T, H, d_k), 4);

  Tensor *V = linear_t(mem, wv2, x_new);
  V = reshape_t(mem, V, S(B, T, H, d_k), 4);

  Q = permute_t(mem, Q, S(0, 2, 1, 3), 4);
  K = permute_t(mem, K, S(0, 2, 1, 3), 4);
  V = permute_t(mem, V, S(0, 2, 1, 3), 4);

  Pair_T result_ = scaled_dot_product_attention(mem, Q, K, V, src_mask);
  Tensor *attn2 = result_.S;
  attn2 = reshape_t(mem, attn2, S(B * T, D), 2);
  attn2 = linear_t(mem, wo2, attn2);
  attn2 = reshape_t(mem, attn2, S(B, T, D), 3);

  Tensor *xx = add_norm(mem, x_, attn2, norm2);
  Tensor *ffout = feed_forward(mem, xx, w1, w2);

  Tensor *x_final = add_norm(mem, xx, ffout, norm3);
  return x_final;
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
  // Tensor *out = feed_forward(mem, x, w1, w2);

  Linear *wq = create_linear(mem, pl, D, D);
  Linear *wk = create_linear(mem, pl, D, D);
  Linear *wv = create_linear(mem, pl, D, D);
  Linear *wo = create_linear(mem, pl, D, D);

  // Tensor *out = multi_head_attention(mem, x, wq, wk, wv, wo, NULL);

  Tensor *sub_layer = tensor_xavier(mem, S(B, T, D), 3, PERM);
  LayerNorm *l = create_layernorm(mem, pl, D, 1e-5);
  Tensor *out = add_norm(mem, x, sub_layer, l);
  backward(mem, out);

  print_tensor_shape(out);
  free_global_mem(mem);
  return 0;
}
