#ifndef BARE_H
#define BARE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define ERROR(msg) fprintf(stderr, "ERROR-> %s\n", msg)

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
} Tensor;

void backward(Tensor *root);

Tensor *tensor_init(int64_t *shape, int ndim);
Tensor *tensor_zeros(int64_t *shape, int ndim);
Tensor *tensor_ones(int64_t *shape, int ndim);
float tensor_get(Tensor *t, int64_t *indices);
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

#endif // !BARE_H
