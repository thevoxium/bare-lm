#ifndef BARE_H
#define BARE_H

#include <cblas.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAGIC_NUMBER 0x42415245
#define VERSION 1

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

#define ALIGNMENT 32

#define S(...)                                                                 \
  (int[]) { __VA_ARGS__ }

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
  BMM,
  TRANSPOSE,
  RESHAPE,
  BROADCAST,
  CROSSENTROPY,
  MASK,
  SCALE,
  CONCAT,
  SLICE,
  PERMUTE,
  SOFTMAX,
  EMBEDDING,
} Op;

typedef enum DType {
  fp32,
} DType;

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
  uint32_t visited_gen;
  uint8_t contiguous;
  Op op;
  void (*backward)(struct Tensor *self);
  float op_params[4];
  DType dt;
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

typedef struct LayerNorm {
  Tensor *weight;
  Tensor *bias;
  float eps;
} LayerNorm;

typedef struct Pair_T {
  Tensor *F;
  Tensor *S;
} Pair_T;

typedef struct Adam {
  float lr;
  float beta1;
  float beta2;
  float eps;
  float **m;
  float **v;
  int t;
} Adam;

typedef struct AdamW {
  float lr;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  float **m;
  float **v;
  int t;
} AdamW;

Adam *adam_init(Memory *mem, ParameterList *pl, float lr, float beta1,
                float beta2, float eps, int t);
void adam_step(Adam *optim, ParameterList *pl);

AdamW *adamw_init(Memory *mem, ParameterList *pl, float lr, float beta1,
                  float beta2, float eps, float weight_decay, int t);
void adamw_step(AdamW *optim, ParameterList *pl);

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

LayerNorm *create_layernorm(Memory *mem, ParameterList *pl,
                            int normalized_shape, float eps);
Tensor *layernorm_t(Memory *mem, LayerNorm *ln, Tensor *x);

void backward(Memory *mem, Tensor *root);

int element_size(Tensor *a);
int nbytes(Tensor *a);
uint8_t is_floating_point(Tensor *a);
int *tensor_size(Memory *mem, Tensor *a);
float item(Tensor *a);

Tensor *tensor_init(Memory *mem, int *shape, int ndim, uint8_t perm);
Tensor *tensor_zeros(Memory *mem, int *shape, int ndim, uint8_t perm);
Tensor *tensor_ones(Memory *mem, int *shape, int ndim, uint8_t perm);
Tensor *tensor_randn(Memory *mem, int *shape, int ndim, uint8_t perm);
Tensor *tensor_xavier(Memory *mem, int *shape, int ndim, uint8_t perm);
Tensor *tensor_eye(Memory *mem, int n, int m, uint8_t perm);
Tensor *tensor_full_like(Memory *mem, Tensor *a, float fill_value,
                         uint8_t perm);
Tensor *tensor_zeros_like(Memory *mem, Tensor *a, uint8_t perm);
Tensor *tensor_ones_like(Memory *mem, Tensor *a, uint8_t perm);
Tensor *tensor_randn_like(Memory *mem, Tensor *a, uint8_t perm);
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
Tensor *bmm_t(Memory *mem, Tensor *a, Tensor *b);
Tensor *transpose_t(Memory *mem, Tensor *a);
Tensor *reshape_t(Memory *mem, Tensor *a, int *shape, int ndim);
Tensor *view_t(Memory *mem, Tensor *a, int *shape, int ndim);
Tensor *squeeze_t(Memory *mem, Tensor *a, int dim);
Tensor *unsqueeze_t(Memory *mem, Tensor *a, int dim);
Tensor *broadcast_t(Memory *mem, Tensor *a, int *shape, int tar_dim);
Tensor *expand_t(Memory *mem, Tensor *a, int *shape, int tar_dim);
Tensor *mask_t(Memory *mem, Tensor *a, Tensor *b, float val);
Tensor *scale_t(Memory *mem, Tensor *a, float v);
Tensor *concat_t(Memory *mem, Tensor *a, Tensor *b, int dim);
Pair_T *slice_t(Memory *mem, Tensor *a, int dim, int split_size);
Tensor *permute_t(Memory *mem, Tensor *a, int *dims, int total_dim);
Tensor *softmax_t(Memory *mem, Tensor *a, int dim);
void replace_t(Tensor *a, Tensor *b);
void detach_t(Tensor *a);

void sgd_step(ParameterList *pl, float lr);
void clip_gradients(ParameterList *pl, float threshold);

void save_checkpoint(ParameterList *pl, const char *path);
void load_checkpoint(ParameterList *pl, const char *path);

void mem_stats(Memory *mem);

#endif //! BARE_H
