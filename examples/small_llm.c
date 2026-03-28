#include "../src/bare.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_ARRAY_LEN 512

typedef struct Config {
  int batch_size;
  int block_size;
  int n_embed;
  int max_iters;
  float lr;
} Config;

Config config = {
    .batch_size = 4,
    .block_size = 8,
    .n_embed = 32,
    .max_iters = 500,
    .lr = 1e-3,
};

int cmp_char(const void *a, const void *b) { return (*(char *)a - *(char *)b); }

int encode(int *array, char *str) {
  int len = strlen(str);
  for (int i = 0; i < len; i++) {
    if (i < MAX_ARRAY_LEN) {
      array[i] = (unsigned char)str[i];
    } else {
      printf("increase array len\n");
      exit(1);
    }
  }
  return len;
}

char *decode(int *array, int len) {
  char *str = malloc(len + 1);
  for (int i = 0; i < len; i++) {
    str[i] = (char)array[i];
  }
  str[len] = '\0';
  return str;
}

void get_batch(Memory *mem, Pair_T *pt, Tensor *data) {
  int idx[config.batch_size];
  int N = data->numel - config.block_size - 1;

  for (int i = 0; i < config.batch_size; i++) {
    idx[i] = rand() % (N + 1);
  }

  int out_shape[] = {config.batch_size, config.block_size};
  pt->F = tensor_zeros(mem, out_shape, 2, PERM);
  pt->S = tensor_zeros(mem, out_shape, 2, PERM);

  for (int i = 0; i < config.batch_size; i++) {
    int current_idx = idx[i];
    for (int j = 0; j < config.block_size; j++) {
      pt->F->data[i * out_shape[1] + j] = 1.0f * data->data[current_idx + j];
      pt->S->data[i * out_shape[1] + j] =
          1.0f * data->data[current_idx + j + 1];
    }
  }

  return;
}

int main() {
  srand(time(NULL));
  Memory *mem = create_global_mem(1 << 30);
  ParameterList *pl = create_param_list(mem);

  char text[] = "hello world, this is a tiny gpt";
  int seen[256] = {0};
  int len = strlen(text);

  for (int i = 0; i < len; i++) {
    unsigned char c = text[i];
    seen[c] = 1;
  }

  char unique_chars[256];
  int vocab_size = 0;
  for (int i = 0; i < 256; i++) {
    if (seen[i]) {
      unique_chars[vocab_size++] = (char)i;
    }
  }

  qsort(unique_chars, vocab_size, sizeof(char), cmp_char);

  // printf("Unique sorted chars:\n");
  // for (int i = 0; i < vocab_size; i++) {
  //   printf("%c ", unique_chars[i]);
  // }
  // printf("\nVocab size: %d\n", vocab_size);

  int input_tokens[MAX_ARRAY_LEN];
  int input_tokens_len = encode(input_tokens, text);
  // for (int i = 0; i < l; i++) {
  //   printf("%d, ", input_tokens[i]);
  // }
  // printf("\n");

  char *s = decode(input_tokens, input_tokens_len);
  // printf("%s", s);
  // printf("\n");
  //

  int data_shape[] = {input_tokens_len};
  Tensor *data = tensor_zeros(mem, data_shape, 1, PERM);
  for (int i = 0; i < data->numel; i++) {
    data->data[i] = input_tokens[i];
  }

  // token_embeddings
  // int tok_shape[] = {vocab_size, config.n_embed};
  // Tensor *token_embedding_table = tensor_randn(mem, tok_shape, 2, PERM);
  // param_list_add(mem, pl, token_embedding_table);
  //
  // // position_embeddings
  // int pos_shape[] = {config.block_size, config.n_embed};
  // Tensor *position_embedding_table = tensor_randn(mem, pos_shape, 2, PERM);
  // param_list_add(mem, pl, position_embedding_table);
  //
  // // creaing position indices
  // int pos_index_shape[] = {1, T};
  // Tensor *pos_indices = tensor_zeros(mem, pos_index_shape, 2, PERM);
  // for (int i = 0; i < pos_indices->numel; i++) {
  //   pos_indices->data[i] = i;
  // }
  //
  // Tensor *tok_emb = embedding_t(mem, token_embedding_table, x);
  // Tensor *pos_emb = embedding_t(mem, position_embedding_table, pos_indices);
  //
  // int pos_b_shape[] = {B, T, config.n_embed};
  // Tensor *pos_emb_b = broadcast_t(mem, pos_emb, pos_b_shape, 3);
  //
  // Tensor *x_emb = add_t(mem, tok_emb, pos_emb_b);

  int shape[] = {config.n_embed, config.n_embed};
  Tensor *Wq = tensor_randn(mem, shape, 2, PERM);
  Tensor *Wk = tensor_randn(mem, shape, 2, PERM);
  Tensor *Wv = tensor_randn(mem, shape, 2, PERM);

  int shape2[] = {config.n_embed, 4 * config.n_embed};
  Tensor *W1 = tensor_randn(mem, shape2, 2, PERM);

  int shape3[] = {4 * config.n_embed};
  Tensor *B1 = tensor_zeros(mem, shape3, 1, PERM);

  int shape4[] = {4 * config.n_embed, config.n_embed};
  Tensor *W2 = tensor_randn(mem, shape4, 2, PERM);

  int shape5[] = {config.n_embed};
  Tensor *B2 = tensor_zeros(mem, shape5, 1, PERM);

  int shape6[] = {config.n_embed, vocab_size};
  Tensor *W_out = tensor_randn(mem, shape6, 2, PERM);

  int shape7[] = {vocab_size};
  Tensor *B_out = tensor_zeros(mem, shape7, 1, PERM);

  int shape8[] = {vocab_size, config.n_embed};
  Tensor *token_embeddings = tensor_randn(mem, shape8, 2, PERM);

  int shape9[] = {config.block_size, config.n_embed};
  Tensor *position_embeddings = tensor_randn(mem, shape9, 2, PERM);

  param_list_add(mem, pl, token_embeddings);
  param_list_add(mem, pl, position_embeddings);
  param_list_add(mem, pl, Wq);
  param_list_add(mem, pl, Wk);
  param_list_add(mem, pl, Wv);
  param_list_add(mem, pl, W1);
  param_list_add(mem, pl, B1);
  param_list_add(mem, pl, W2);
  param_list_add(mem, pl, B2);
  param_list_add(mem, pl, W_out);
  param_list_add(mem, pl, B_out);

  for (int i = 0; i < config.max_iters; i++) {
    printf("Step: %d \n", i);
    reset_temp_mem(mem);
    zero_grad(pl);

    Pair_T pt;
    get_batch(mem, &pt, data);

    Tensor *x = pt.F;
    Tensor *y = pt.S;

    Pair_T result;

    int B = x->shape[0];    // 4
    int T = x->shape[1];    // 8
    int C = config.n_embed; // 32

    int pos_index_shape[] = {1, T};
    Tensor *pos_indices = tensor_zeros(mem, pos_index_shape, 2, PERM);
    for (int i = 0; i < pos_indices->numel; i++) {
      pos_indices->data[i] = i;
    }
    Tensor *tok_emb = embedding_t(mem, token_embeddings, x);
    Tensor *pos_emb = embedding_t(mem, position_embeddings, pos_indices);
    int pos_b_shape[] = {B, T, config.n_embed};

    Tensor *pos_emb_b = broadcast_t(mem, pos_emb, pos_b_shape, 3);
    Tensor *x_emb = add_t(mem, tok_emb, pos_emb_b);

    Tensor *q = bmm_t(mem, x_emb, Wq);
    Tensor *k = bmm_t(mem, x_emb, Wk);
    Tensor *v = bmm_t(mem, x_emb, Wv);

    int k_shape[] = {B, C, T};
    Tensor *k_t = reshape_t(mem, k, k_shape, 3);

    Tensor *scores = bmm_t(mem, q, k_t);
    Tensor *scaled_scores = scale_t(mem, scores, 1 / pow(config.n_embed, 0.5));

    int mask_shape[] = {B, T, T};
    Tensor *causal_mask = tensor_zeros(mem, mask_shape, 3, TEMP);
    for (int b = 0; b < B; b++) {
      for (int i = 0; i < T; i++) {
        for (int j = 0; j < T; j++) {
          if (i < j) {
            causal_mask->data[b * T * T + i * T + j] = 1.0f;
          }
        }
      }
    }
    Tensor *masked_scores = mask_t(mem, scaled_scores, causal_mask, -INFINITY);
    Tensor *weights = softmax_t(mem, masked_scores, 2);
    Tensor *attn_out = bmm_t(mem, weights, v);
    // print_t(W1, 0);

    Tensor *ff1_w = bmm_t(mem, attn_out, W1);
    int ff1_b_shape[] = {B, T, 4 * config.n_embed};
    Tensor *B1_b = broadcast_t(mem, B1, ff1_b_shape, 3);
    Tensor *ff1 = add_t(mem, ff1_w, B1_b);
    Tensor *relu1 = relu_t(mem, ff1);

    Tensor *ff2_w = bmm_t(mem, relu1, W2);
    int ff2_b_shape[] = {B, T, config.n_embed};
    Tensor *B2_b = broadcast_t(mem, B2, ff2_b_shape, 3);
    Tensor *ff2 = add_t(mem, ff2_w, B2_b);

    Tensor *logits_w = bmm_t(mem, ff2, W_out);
    int logits_b_shape[] = {B, T, vocab_size};
    Tensor *B_out_b = broadcast_t(mem, B_out, logits_b_shape, 3);
    Tensor *logits = add_t(mem, logits_w, B_out_b);

    int l_shape[] = {B * T, vocab_size};
    Tensor *logits_flat = reshape_t(mem, logits, l_shape, 2);

    int targets_flat_shape[] = {B * T};
    Tensor *targets_flat = reshape_t(mem, y, targets_flat_shape, 1);

    Tensor *loss = crossentropyloss_t(mem, logits_flat, targets_flat);

    backward(mem, loss);

    sgd_step(pl, config.lr);
    print_t(loss, 0);
  }

  free_global_mem(mem);

  return 0;
}
