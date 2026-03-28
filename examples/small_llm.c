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
  Memory *mem = create_global_mem(1 << 28);
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

  Pair_T pt;
  get_batch(mem, &pt, data);

  Tensor *x = pt.F;
  Tensor *y = pt.S;

  int B = x->shape[0];
  int T = x->shape[1];

  // token_embeddings
  int tok_shape[] = {vocab_size, config.n_embed};
  Tensor *token_embedding_table = tensor_randn(mem, tok_shape, 2, PERM);
  param_list_add(mem, pl, token_embedding_table);

  // position_embeddings
  int pos_shape[] = {config.block_size, config.n_embed};
  Tensor *position_embedding_table = tensor_randn(mem, pos_shape, 2, PERM);
  param_list_add(mem, pl, position_embedding_table);

  // creaing position indices
  int pos_index_shape[] = {1, T};
  Tensor *pos_indices = tensor_zeros(mem, pos_index_shape, 2, PERM);
  for (int i = 0; i < pos_indices->numel; i++) {
    pos_indices->data[i] = i;
  }

  Tensor *tok_emb = embedding_t(mem, token_embedding_table, x);
  Tensor *pos_emb = embedding_t(mem, position_embedding_table, pos_indices);

  int pos_b_shape[] = {B, T, config.n_embed};
  Tensor *pos_emb_b = broadcast_t(mem, pos_emb, pos_b_shape, 3);

  Tensor *x_emb = add_t(mem, tok_emb, pos_emb_b);

  return 0;
}
