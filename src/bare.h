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
  int *shape;
  int *strides;
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

Dt_array *dt_array_create();
void dt_array_free(Dt_array *a);
void dt_array_push(Dt_array *a, Tensor *t);

typedef struct Linear {
  Tensor *weights;
  Tensor *bias;
} Linear;

Linear *create_linear(int d_in, int d_out);
Tensor *linear_t(Linear *l, Tensor *x);
void linear_free(Linear *l);

void backward(Tensor *root);

Tensor *tensor_init(int *shape, int ndim);
Tensor *tensor_zeros(int *shape, int ndim);
Tensor *tensor_ones(int *shape, int ndim);
Tensor *tensor_randn(int *shape, int ndim);
float tensor_get(Tensor *t, int *indices);
void print_t(Tensor *t, uint8_t grad);
void tensor_free(Tensor *t);

Tensor *add_t(Tensor *a, Tensor *b);
Tensor *sub_t(Tensor *a, Tensor *b);
Tensor *mul_t(Tensor *a, Tensor *b);
Tensor *divi_t(Tensor *a, Tensor *b);
Tensor *neg_t(Tensor *a);
Tensor *pow_t(Tensor *a, float exponent);
Tensor *exp_t(Tensor *a);
Tensor *log_t(Tensor *a);
Tensor *sum_t(Tensor *a, int dim);
Tensor *mean_t(Tensor *a, int dim);
Tensor *dot_t(Tensor *a, Tensor *b);
Tensor *max_t(Tensor *a, int dim);
Tensor *relu_t(Tensor *a);
Tensor *gelu_t(Tensor *a);
Tensor *sigmoid_t(Tensor *a);
Tensor *tanh_t(Tensor *a);
Tensor *mseloss_t(Tensor *a, Tensor *b);
Tensor *crossentropyloss_t(Tensor *a, Tensor *b);
Tensor *matmul_t(Tensor *a, Tensor *b);
Tensor *transpose_t(Tensor *a);
Tensor *reshape_t(Tensor *a, int *shape, int ndim);
Tensor *squeeze_t(Tensor *a, int dim);
Tensor *unsqueeze_t(Tensor *a, int dim);
Tensor *broadcast_t(Tensor *a, int *shape, int tar_dim);

#endif // !BARE_H
