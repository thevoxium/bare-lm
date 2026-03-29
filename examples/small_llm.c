#include "../src/bare.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_ARRAY_LEN 2000000

typedef struct Config {
  int batch_size;
  int block_size;
  int n_embed;
  int max_iters;
  int vocab_size;
} Config;

Config config = {
    .batch_size = 4,
    .block_size = 64,
    .n_embed = 256,
    .max_iters = 1000,
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

void get_batch(Memory *mem, Pair_T *pt, Tensor *data, int *char_to_idx) {
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
      pt->F->data[i * out_shape[1] + j] =
          char_to_idx[(int)data->data[current_idx + j]];
      pt->S->data[i * out_shape[1] + j] =
          char_to_idx[(int)data->data[current_idx + j + 1]];
    }
  }

  return;
}

typedef struct ForwardResult {
  Tensor *logits;
} ForwardResult;

ForwardResult forward(Memory *mem, Tensor *x, Tensor *token_embeddings,
                      Tensor *position_embeddings, LayerNorm *ln1,
                      LayerNorm *ln2, Tensor *Wq, Tensor *Bq, Tensor *Wk,
                      Tensor *Bk, Tensor *Wv, Tensor *Bv, Tensor *W1,
                      Tensor *B1, Tensor *W2, Tensor *B2, Tensor *W_out,
                      Tensor *B_out) {
  ForwardResult result;

  int B = x->shape[0];
  int T = x->shape[1];
  int C = config.n_embed;

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

  Tensor *x_norm = layernorm_t(mem, ln1, x_emb);

  Tensor *q = bmm_t(mem, x_norm, Wq);
  int q_b_shape[] = {B, T, config.n_embed};
  Tensor *Bq_b = broadcast_t(mem, Bq, q_b_shape, 3);
  q = add_t(mem, q, Bq_b);

  Tensor *k = bmm_t(mem, x_norm, Wk);
  Tensor *Bk_b = broadcast_t(mem, Bk, q_b_shape, 3);
  k = add_t(mem, k, Bk_b);

  Tensor *v = bmm_t(mem, x_norm, Wv);
  Tensor *Bv_b = broadcast_t(mem, Bv, q_b_shape, 3);
  v = add_t(mem, v, Bv_b);

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

  Tensor *attn_residual = add_t(mem, x_emb, attn_out);

  Tensor *ff1_w = bmm_t(mem, attn_residual, W1);
  int ff1_b_shape[] = {B, T, 4 * config.n_embed};
  Tensor *B1_b = broadcast_t(mem, B1, ff1_b_shape, 3);
  Tensor *ff1 = add_t(mem, ff1_w, B1_b);
  Tensor *relu1 = relu_t(mem, ff1);

  Tensor *ff2_w = bmm_t(mem, relu1, W2);
  int ff2_b_shape[] = {B, T, config.n_embed};
  Tensor *B2_b = broadcast_t(mem, B2, ff2_b_shape, 3);
  Tensor *ff2 = add_t(mem, ff2_w, B2_b);

  Tensor *ff_residual = add_t(mem, attn_residual, ff2);

  Tensor *x_norm2 = layernorm_t(mem, ln2, ff_residual);

  Tensor *logits_w = bmm_t(mem, x_norm2, W_out);
  int logits_b_shape[] = {B, T, config.vocab_size};
  Tensor *B_out_b = broadcast_t(mem, B_out, logits_b_shape, 3);
  Tensor *logits = add_t(mem, logits_w, B_out_b);

  result.logits = logits;
  return result;
}

int *generate(Memory *mem, int *start_tokens, int start_len, int max_new_tokens,
              Tensor *token_embeddings, Tensor *position_embeddings,
              LayerNorm *ln1, LayerNorm *ln2, Tensor *Wq, Tensor *Bq,
              Tensor *Wk, Tensor *Bk, Tensor *Wv, Tensor *Bv, Tensor *W1,
              Tensor *B1, Tensor *W2, Tensor *B2, Tensor *W_out,
              Tensor *B_out) {
  int *result = malloc((start_len + max_new_tokens) * sizeof(int));
  for (int i = 0; i < start_len; i++) {
    result[i] = start_tokens[i];
  }

  int current_len = start_len;

  for (int iter = 0; iter < max_new_tokens; iter++) {
    int seq_len = current_len;
    if (seq_len > config.block_size) {
      seq_len = config.block_size;
    }

    int idx_cond_len = seq_len;
    int x_shape[] = {1, idx_cond_len};
    Tensor *x = tensor_zeros(mem, x_shape, 2, PERM);
    int start_offset = current_len - idx_cond_len;
    for (int i = 0; i < idx_cond_len; i++) {
      x->data[i] = result[start_offset + i];
    }

    ForwardResult fr =
        forward(mem, x, token_embeddings, position_embeddings, ln1, ln2, Wq, Bq,
                Wk, Bk, Wv, Bv, W1, B1, W2, B2, W_out, B_out);

    int logits_shape[] = {1, idx_cond_len, config.vocab_size};
    Tensor *logits = reshape_t(mem, fr.logits, logits_shape, 3);

    int last_logits_shape[] = {1, config.vocab_size};
    Tensor *last_logits = tensor_zeros(mem, last_logits_shape, 2, PERM);
    for (int i = 0; i < config.vocab_size; i++) {
      last_logits->data[i] =
          logits->data[(idx_cond_len - 1) * config.vocab_size + i];
    }

    Tensor *probs = softmax_t(mem, last_logits, 1);

    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    int next_token = 0;
    for (int i = 0; i < config.vocab_size; i++) {
      cumsum += probs->data[i];
      if (r <= cumsum) {
        next_token = i;
        break;
      }
    }

    result[current_len] = next_token;
    current_len++;

    reset_temp_mem(mem);
  }

  return result;
}

int main() {
  srand(time(NULL));
  Memory *mem = create_global_mem(1ULL << 31);
  ParameterList *pl = create_param_list(mem);

  FILE *fp = fopen("data/input.txt", "r");
  if (!fp) {
    printf("Error: Could not open data/input.txt\n");
    return 1;
  }
  fseek(fp, 0, SEEK_END);
  long file_size = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  char *text = malloc(file_size + 1);
  fread(text, 1, file_size, fp);
  text[file_size] = '\0';
  fclose(fp);

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

  config.vocab_size = vocab_size;

  int char_to_idx[256] = {0};
  for (int i = 0; i < vocab_size; i++) {
    char_to_idx[(unsigned char)unique_chars[i]] = i;
  }

  int input_tokens[MAX_ARRAY_LEN];
  int input_tokens_len = encode(input_tokens, text);

  int data_shape[] = {input_tokens_len};
  Tensor *data = tensor_zeros(mem, data_shape, 1, PERM);
  for (int i = 0; i < data->numel; i++) {
    data->data[i] = input_tokens[i];
  }

  int shape[] = {config.n_embed, config.n_embed};
  Tensor *Wq = tensor_xavier(mem, shape, 2, PERM);
  Tensor *Wk = tensor_xavier(mem, shape, 2, PERM);
  Tensor *Wv = tensor_xavier(mem, shape, 2, PERM);

  int qkv_bias_shape[] = {config.n_embed};
  Tensor *Bq = tensor_zeros(mem, qkv_bias_shape, 1, PERM);
  Tensor *Bk = tensor_zeros(mem, qkv_bias_shape, 1, PERM);
  Tensor *Bv = tensor_zeros(mem, qkv_bias_shape, 1, PERM);

  int shape2[] = {config.n_embed, 4 * config.n_embed};
  Tensor *W1 = tensor_xavier(mem, shape2, 2, PERM);

  int shape3[] = {4 * config.n_embed};
  Tensor *B1 = tensor_zeros(mem, shape3, 1, PERM);

  int shape4[] = {4 * config.n_embed, config.n_embed};
  Tensor *W2 = tensor_xavier(mem, shape4, 2, PERM);

  int shape5[] = {config.n_embed};
  Tensor *B2 = tensor_zeros(mem, shape5, 1, PERM);

  int shape6[] = {config.n_embed, vocab_size};
  Tensor *W_out = tensor_xavier(mem, shape6, 2, PERM);

  int shape7[] = {vocab_size};
  Tensor *B_out = tensor_zeros(mem, shape7, 1, PERM);

  int shape8[] = {vocab_size, config.n_embed};
  Tensor *token_embeddings = tensor_xavier(mem, shape8, 2, PERM);

  int shape9[] = {config.block_size, config.n_embed};
  Tensor *position_embeddings = tensor_xavier(mem, shape9, 2, PERM);

  LayerNorm *ln1 = create_layernorm(mem, pl, config.n_embed, 1e-5);
  LayerNorm *ln2 = create_layernorm(mem, pl, config.n_embed, 1e-5);

  param_list_add(mem, pl, token_embeddings);
  param_list_add(mem, pl, position_embeddings);
  param_list_add(mem, pl, Wq);
  param_list_add(mem, pl, Bq);
  param_list_add(mem, pl, Wk);
  param_list_add(mem, pl, Bk);
  param_list_add(mem, pl, Wv);
  param_list_add(mem, pl, Bv);
  param_list_add(mem, pl, W1);
  param_list_add(mem, pl, B1);
  param_list_add(mem, pl, W2);
  param_list_add(mem, pl, B2);
  param_list_add(mem, pl, W_out);
  param_list_add(mem, pl, B_out);

  AdamW *optim = adamw_init(mem, pl, 3e-4, 0.9, 0.95, 1e-8, 0.01, 0);

  for (int i = 0; i < config.max_iters; i++) {
    printf("Step: %d \n", i);
    reset_temp_mem(mem);
    zero_grad(pl);

    Pair_T pt;
    get_batch(mem, &pt, data, char_to_idx);

    Tensor *x = pt.F;
    Tensor *y = pt.S;

    ForwardResult fr =
        forward(mem, x, token_embeddings, position_embeddings, ln1, ln2, Wq, Bq,
                Wk, Bk, Wv, Bv, W1, B1, W2, B2, W_out, B_out);
    Tensor *logits = fr.logits;

    int B = x->shape[0];
    int T = x->shape[1];

    int l_shape[] = {B * T, config.vocab_size};
    Tensor *logits_flat = reshape_t(mem, logits, l_shape, 2);

    int targets_flat_shape[] = {B * T};
    Tensor *targets_flat = reshape_t(mem, y, targets_flat_shape, 1);

    Tensor *loss = crossentropyloss_t(mem, logits_flat, targets_flat);

    backward(mem, loss);

    clip_gradients(pl, 5.0f);

    adamw_step(optim, pl);
    print_t(loss, 0);
  }

  int start_len = 32;
  int *start_tokens = malloc(start_len * sizeof(int));
  for (int i = 0; i < start_len; i++) {
    start_tokens[i] = char_to_idx[(unsigned char)text[i]];
  }
  int max_new_tokens = 2000;

  int *generated_tokens =
      generate(mem, start_tokens, start_len, max_new_tokens, token_embeddings,
               position_embeddings, ln1, ln2, Wq, Bq, Wk, Bk, Wv, Bv, W1, B1,
               W2, B2, W_out, B_out);

  printf("Generated: ");
  for (int i = 0; i < start_len + max_new_tokens; i++) {
    putchar(unique_chars[generated_tokens[i]]);
  }
  printf("\n");

  free(generated_tokens);
  free(start_tokens);
  free(text);

  free_global_mem(mem);

  return 0;
}
