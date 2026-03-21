#ifndef BARE_H
#define BARE_H

#include <math.h>
#include <stddef.h>
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

#define PERM 1
#define TEMP 0

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

typedef struct Arena {
  uint8_t *buffer;
  size_t size;
  size_t used;
} Arena;

typedef struct Memory {
  Arena *perm;
  Arena *temp;
} Memory;

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

typedef Dt_array ParameterList;

typedef struct Linear {
  Tensor *weights;
  Tensor *bias;
} Linear;

Memory *create_global_mem(size_t size);
void reset_temp_mem(Memory *mem);
void *allocate_mem(Memory *mem, size_t size, uint8_t perm);
void free_global_mem(Memory *mem);

Dt_array *dt_array_create(Memory *mem, uint8_t perm);
void dt_array_push(Memory *mem, Dt_array *a, Tensor *t, uint8_t perm);

ParameterList *create_param_list(Memory *mem);
void param_list_add(Memory *mem, ParameterList *pl, Tensor *t);

Linear *create_linear(Memory *mem, ParameterList *pl, int d_in, int d_out);
Tensor *linear_t(Memory *mem, Linear *l, Tensor *x);

void backward(Memory *mem, Tensor *root);

Tensor *tensor_init(Memory *mem, int *shape, int ndim, uint8_t perm);
Tensor *tensor_zeros(Memory *mem, int *shape, int ndim, uint8_t perm);
Tensor *tensor_ones(Memory *mem, int *shape, int ndim, uint8_t perm);
Tensor *tensor_randn(Memory *mem, int *shape, int ndim, uint8_t perm);
float tensor_get(Tensor *t, int *indices);
void print_t(Tensor *t, uint8_t grad);
void zero_grad(ParameterList *pl);

Tensor *add_t(Memory *mem, Tensor *a, Tensor *b);
Tensor *sub_t(Memory *mem, Tensor *a, Tensor *b);
Tensor *mul_t(Memory *mem, Tensor *a, Tensor *b);
Tensor *divide_t(Memory *mem, Tensor *a, Tensor *b);
Tensor *neg_t(Memory *mem, Tensor *a);
Tensor *pow_t(Memory *mem, Tensor *a, float exponent);
Tensor *exp_t(Memory *mem, Tensor *a);
Tensor *log_t(Memory *mem, Tensor *a);
Tensor *sum_t(Memory *mem, Tensor *a, int dim);
Tensor *mean_t(Memory *mem, Tensor *a, int dim);
Tensor *dot_t(Memory *mem, Tensor *a, Tensor *b);
Tensor *max_t(Memory *mem, Tensor *a, int dim);
Tensor *relu_t(Memory *mem, Tensor *a);
Tensor *gelu_t(Memory *mem, Tensor *a);
Tensor *sigmoid_t(Memory *mem, Tensor *a);
Tensor *tanh_t(Memory *mem, Tensor *a);
Tensor *mseloss_t(Memory *mem, Tensor *a, Tensor *b);
Tensor *crossentropyloss_t(Memory *mem, Tensor *a, Tensor *b);
Tensor *matmul_t(Memory *mem, Tensor *a, Tensor *b);
Tensor *transpose_t(Memory *mem, Tensor *a);
Tensor *reshape_t(Memory *mem, Tensor *a, int *shape, int ndim);
Tensor *squeeze_t(Memory *mem, Tensor *a, int dim);
Tensor *unsqueeze_t(Memory *mem, Tensor *a, int dim);
Tensor *broadcast_t(Memory *mem, Tensor *a, int *shape, int tar_dim);

void sgd_step(ParameterList *pl, float lr);

#endif // !BARE_H
