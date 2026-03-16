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
} Op;

typedef struct Tensor {
  float *data;
  float *grad;
  int64_t *shape;
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
void print_t(Tensor *t, uint8_t grad);

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

#endif // !BARE_H
