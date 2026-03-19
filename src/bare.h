#ifndef BARE_H
#define BARE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ERROR(msg) fprintf(stderr, "ERROR-> %s\n", msg)
#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ERROR(msg);                                                              \
      return NULL;                                                             \
    }                                                                          \
  } while (0)
#define CHECK_VOID(cond, msg)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      ERROR(msg);                                                              \
      return;                                                                  \
    }                                                                          \
  } while (0)

typedef enum Op {
  NONE,
  ADD,
  SUB,
  MUL,
  DIV,
  NEG,
  POW,
  EXP,
  LOG,
  SUM_REDUCTION,
  MEAN_REDUCTION,
  DOT,
  MAX,
  RELU,
  GELU,
  SIGMOID,
  TANH,
  MSELOSS,
  MATMUL,
  TRANSPOSE,
  RESHAPE,
  BROADCAST,
  CROSSENTROPY,
} Op;

typedef struct Tensor {
  float *data;
  float *grad;
  int64_t *shape;
  int64_t *strides;
  struct Tensor *parents[2];
  int ndim;
  int numel;
  Op op;
  void (*backward)(struct Tensor *self);
  float op_params[2];
} Tensor;

typedef struct Dt_array {
  Tensor **t;
  int count;
  int capacity;
} Dt_array;

typedef struct GraphContext {
  Dt_array *dt_array;
} GraphContext;

GraphContext *create_global_ctx();
void ctx_free(GraphContext *ctx);

Dt_array *dt_array_create();
void dt_array_free(Dt_array *a);
void dt_array_push(Dt_array *a, Tensor *t);

void backward(Tensor *root);

Tensor *tensor_init(int64_t *shape, int ndim, GraphContext *ctx);
Tensor *tensor_zeros(int64_t *shape, int ndim, GraphContext *ctx);
Tensor *tensor_ones(int64_t *shape, int ndim, GraphContext *ctx);
Tensor *tensor_randn(int64_t *shape, int ndim, GraphContext *ctx);
float tensor_get(Tensor *t, int64_t *indices);
void print_t(Tensor *t, uint8_t grad);
void tensor_free(Tensor *t);

Tensor *add_t(Tensor *a, Tensor *b, GraphContext *ctx);
Tensor *sub_t(Tensor *a, Tensor *b, GraphContext *ctx);
Tensor *mul_t(Tensor *a, Tensor *b, GraphContext *ctx);
Tensor *divi_t(Tensor *a, Tensor *b, GraphContext *ctx);
Tensor *neg_t(Tensor *a, GraphContext *ctx);
Tensor *pow_t(Tensor *a, float exponent, GraphContext *ctx);
Tensor *exp_t(Tensor *a, GraphContext *ctx);
Tensor *log_t(Tensor *a, GraphContext *ctx);
Tensor *sum_t(Tensor *a, int dim, GraphContext *ctx);
Tensor *mean_t(Tensor *a, int dim, GraphContext *ctx);
Tensor *dot_t(Tensor *a, Tensor *b, GraphContext *ctx);
Tensor *max_t(Tensor *a, int dim, GraphContext *ctx);
Tensor *relu_t(Tensor *a, GraphContext *ctx);
Tensor *gelu_t(Tensor *a, GraphContext *ctx);
Tensor *sigmoid_t(Tensor *a, GraphContext *ctx);
Tensor *tanh_t(Tensor *a, GraphContext *ctx);
Tensor *mseloss_t(Tensor *a, Tensor *b, GraphContext *ctx);
Tensor *crossentropyloss_t(Tensor *a, Tensor *b, GraphContext *ctx);
Tensor *matmul_t(Tensor *a, Tensor *b, GraphContext *ctx);
Tensor *transpose_t(Tensor *a, GraphContext *ctx);
Tensor *reshape_t(Tensor *a, int64_t *shape, int ndim, GraphContext *ctx);
Tensor *squeeze_t(Tensor *a, int dim, GraphContext *ctx);
Tensor *unsqueeze_t(Tensor *a, int dim, GraphContext *ctx);
Tensor *broadcast_t(Tensor *a, int64_t *shape, int tar_dim, GraphContext *ctx);

#endif // !BARE_H
