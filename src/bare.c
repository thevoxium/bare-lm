#include "bare.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static void print_data(float *data, int64_t *shape, int ndim, int dim, int *idx,
                       int indent) {
  printf("%*s[", indent, "");
  if (dim == ndim - 1) {
    for (int i = 0; i < shape[dim]; i++) {
      printf("%.4f", data[*idx]);
      (*idx)++;
      if (i < shape[dim] - 1)
        printf(", ");
    }
  } else {
    printf("\n");
    for (int i = 0; i < shape[dim]; i++) {
      print_data(data, shape, ndim, dim + 1, idx, indent + 2);
      if (i < shape[dim] - 1)
        printf(",\n");
    }
    printf("\n%*s]", indent, "");
    return;
  }
  printf("]");
}

void print_t(Tensor *t, uint8_t grad) {
  if (!t) {
    printf("Tensor(NULL)\n");
    return;
  }

  printf("Tensor(\n");
  int idx = 0;
  print_data(t->data, t->shape, t->ndim, 0, &idx, 2);
  printf(",\n  shape=[");
  for (int i = 0; i < t->ndim; i++) {
    printf("%lld", (long long)t->shape[i]);
    if (i < t->ndim - 1)
      printf(", ");
  }
  printf("]\n");

  if (grad) {
    printf(",\n  grad=\n");
    idx = 0;
    print_data(t->grad, t->shape, t->ndim, 0, &idx, 2);
    printf("\n)\n");
  } else {
    printf("\n)\n");
  }
}

static void build_topo(Tensor *root, Tensor ***result, int *result_count,
                       int *result_capacity, Tensor ***visited,
                       int *visited_count, int *visited_capacity) {
  if (!root) {
    return;
  }

  for (int i = 0; i < *visited_count; i++) {
    if ((*visited)[i] == root) {
      return;
    }
  }

  if (*visited_count >= *visited_capacity) {
    *visited_capacity *= 2;
    Tensor **tmp = realloc(*visited, sizeof(Tensor *) * (*visited_capacity));
    if (tmp) {
      *visited = tmp;
    } else {
      ERROR("build_topo: realloc failed");
      return;
    }
  }

  (*visited)[(*visited_count)++] = root;

  for (int i = 0; i < 2; i++) {
    if (root->parents[i]) {
      build_topo(root->parents[i], result, result_count, result_capacity,
                 visited, visited_count, visited_capacity);
    }
  }

  if (*result_count >= *result_capacity) {
    *result_capacity *= 2;
    Tensor **tmp = realloc(*result, sizeof(Tensor *) * (*result_capacity));
    if (tmp) {
      *result = tmp;
    } else {
      ERROR("build_topo: realloc failed");
      return;
    }
  }
  (*result)[(*result_count)++] = root;
}

void backward(Tensor *root) {
  if (!root) {
    ERROR("backward: root is NULL");
    return;
  }

  int result_count = 0;
  int result_capacity = 16;
  Tensor **result = calloc(result_capacity, sizeof(Tensor *));
  if (!result) {
    ERROR("backward: result malloc failed");
    return;
  }

  int visited_count = 0;
  int visited_capacity = 16;
  Tensor **visited = calloc(visited_capacity, sizeof(Tensor *));
  if (!visited) {
    free(result);
    ERROR("backward: visited malloc failed");
    return;
  }

  build_topo(root, &result, &result_count, &result_capacity, &visited,
             &visited_count, &visited_capacity);

  for (int i = 0; i < root->numel; i++) {
    root->grad[i] = 1.0f;
  }

  for (int i = result_count - 1; i >= 0; i--) {
    if (!result[i]) {
      printf("bad topo node\n");
      continue;
    }
    if (result[i]->backward) {
      result[i]->backward(result[i]);
    }
  }
  free(result);
  free(visited);
}

Tensor *tensor_init(int64_t *shape, int ndim) {
  if (!shape || ndim <= 0) {
    ERROR("tensor_init: param is invalid");
    return NULL;
  }

  Tensor *t = malloc(sizeof(Tensor));
  if (!t) {
    ERROR("tensor_init: malloc failed");
    return NULL;
  }

  t->ndim = ndim;
  t->numel = 1;
  t->shape = malloc(ndim * sizeof(int64_t));
  if (!t->shape) {
    ERROR("tensor_init: malloc failed");
    free(t);
    return NULL;
  }

  t->strides = malloc(ndim * sizeof(int64_t));
  if (!t->strides) {
    ERROR("tensor_init: malloc failed");
    free(t->shape);
    free(t);
    return NULL;
  }

  for (int i = 0; i < ndim; i++) {
    t->shape[i] = shape[i];
    t->numel *= shape[i];
  }

  t->strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--) {
    t->strides[i] = t->strides[i + 1] * shape[i + 1];
  }

  t->data = malloc(t->numel * sizeof(float));
  t->grad = malloc(t->numel * sizeof(float));

  if (!t->data || !t->grad) {
    ERROR("tensor_init: malloc failed");
    free(t->shape);
    free(t->strides);
    free(t->data);
    free(t->grad);
    free(t);
    return NULL;
  }

  for (int i = 0; i < t->numel; i++) {
    t->data[i] = 0.0f;
    t->grad[i] = 0.0f;
  }

  t->op = NONE;
  t->parents[0] = NULL;
  t->parents[1] = NULL;
  t->backward = NULL;
  return t;
}

float tensor_get(Tensor *t, int64_t *indices) {
  if (!t || !indices) {
    ERROR("get: invalid param");
    return 0.0f;
  }
  int64_t idx = 0;
  for (int i = 0; i < t->ndim; i++) {
    idx += indices[i] * t->strides[i];
  }
  return t->data[idx];
}

void tensor_free(Tensor *t) {
  if (!t) {
    ERROR("tensor_free: tensor is NULL");
    return;
  }
  free(t->shape);
  free(t->strides);
  free(t->data);
  free(t->grad);
  free(t);
}

Tensor *tensor_zeros(int64_t *shape, int ndim) {
  Tensor *t = tensor_init(shape, ndim);
  if (!t) {
    ERROR("tensor_zeros: tensor_init failed");
    return NULL;
  }
  return t;
}

Tensor *tensor_ones(int64_t *shape, int ndim) {
  Tensor *t = tensor_init(shape, ndim);
  if (!t) {
    ERROR("tensor_ones: tensor_init failed");
    return NULL;
  }
  for (int i = 0; i < t->numel; i++) {
    t->data[i] = 1.0f;
  }
  return t;
}

Tensor *tensor_randn(int64_t *shape, int ndim) {
  Tensor *t = tensor_init(shape, ndim);
  if (!t) {
    ERROR("tensor_randn: tensor_init failed");
    return NULL;
  }

  for (int i = 0; i < t->numel; i += 2) {
    float u1, u2;
    do {
      u1 = (float)rand() / (float)RAND_MAX;
    } while (u1 == 0.0f);
    u2 = (float)rand() / (float)RAND_MAX;

    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * (float)M_PI * u2;

    t->data[i] = r * cosf(theta);
    if (i + 1 < t->numel) {
      t->data[i + 1] = r * sinf(theta);
    }
  }
  return t;
}

static void backward_add(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *b = self->parents[1];
  for (int i = 0; i < self->numel; i++) {
    if (a)
      a->grad[i] += self->grad[i];
    if (b)
      b->grad[i] += self->grad[i];
  }
}

Tensor *add_t(Tensor *a, Tensor *b) {
  if (!a || !b || a->numel != b->numel) {
    ERROR("add: param is invalid");
    return NULL;
  }

  Tensor *r = tensor_init(a->shape, a->ndim);
  if (!r) {
    ERROR("add: tensor_init failed");
    return NULL;
  }

  for (int i = 0; i < r->numel; i++) {
    r->data[i] = a->data[i] + b->data[i];
  }

  r->parents[0] = a;
  r->parents[1] = b;
  r->op = ADD;
  r->backward = backward_add;

  return r;
}

static void backward_sub(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *b = self->parents[1];
  for (int i = 0; i < self->numel; i++) {
    if (a)
      a->grad[i] += self->grad[i];
    if (b)
      b->grad[i] -= self->grad[i];
  }
}

Tensor *sub_t(Tensor *a, Tensor *b) {
  if (!a || !b || a->numel != b->numel) {
    ERROR("sub: param is invalid");
    return NULL;
  }

  Tensor *r = tensor_init(a->shape, a->ndim);
  if (!r) {
    ERROR("sub: tensor_init failed");
    return NULL;
  }

  for (int i = 0; i < r->numel; i++) {
    r->data[i] = a->data[i] - b->data[i];
  }

  r->parents[0] = a;
  r->parents[1] = b;
  r->op = SUB;
  r->backward = backward_sub;

  return r;
}

static void backward_mul(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *b = self->parents[1];
  for (int i = 0; i < self->numel; i++) {
    if (a)
      a->grad[i] += self->grad[i] * b->data[i];
    if (b)
      b->grad[i] += self->grad[i] * a->data[i];
  }
}

Tensor *mul_t(Tensor *a, Tensor *b) {
  if (!a || !b || a->numel != b->numel) {
    ERROR("mul: param is invalid");
    return NULL;
  }

  Tensor *r = tensor_init(a->shape, a->ndim);
  if (!r) {
    ERROR("mul: tensor_init failed");
    return NULL;
  }

  for (int i = 0; i < r->numel; i++) {
    r->data[i] = a->data[i] * b->data[i];
  }

  r->parents[0] = a;
  r->parents[1] = b;
  r->op = MUL;
  r->backward = backward_mul;

  return r;
}

static void backward_div(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *b = self->parents[1];
  for (int i = 0; i < self->numel; i++) {
    if (a)
      a->grad[i] += self->grad[i] / b->data[i];
    if (b)
      b->grad[i] -= self->grad[i] * a->data[i] / (b->data[i] * b->data[i]);
  }
}

Tensor *divi_t(Tensor *a, Tensor *b) {
  if (!a || !b || a->numel != b->numel) {
    ERROR("div_op: param is invalid");
    return NULL;
  }

  Tensor *r = tensor_init(a->shape, a->ndim);
  if (!r) {
    ERROR("div_op: tensor_init failed");
    return NULL;
  }

  for (int i = 0; i < r->numel; i++) {
    r->data[i] = a->data[i] / b->data[i];
  }

  r->parents[0] = a;
  r->parents[1] = b;
  r->op = DIV;
  r->backward = backward_div;

  return r;
}

static void backward_neg(Tensor *self) {
  Tensor *a = self->parents[0];
  for (int i = 0; i < self->numel; i++) {
    if (a)
      a->grad[i] -= self->grad[i];
  }
}

Tensor *neg_t(Tensor *a) {
  if (!a) {
    ERROR("neg: param is invalid");
    return NULL;
  }

  Tensor *r = tensor_init(a->shape, a->ndim);
  if (!r) {
    ERROR("neg: tensor_init failed");
    return NULL;
  }

  for (int i = 0; i < r->numel; i++) {
    r->data[i] = -a->data[i];
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = NEG;
  r->backward = backward_neg;

  return r;
}

static float pow_exponent;
static void backward_pow(Tensor *self) {
  Tensor *a = self->parents[0];
  for (int i = 0; i < self->numel; i++) {
    if (a)
      a->grad[i] +=
          self->grad[i] * pow_exponent * powf(a->data[i], pow_exponent - 1);
  }
}

Tensor *pow_t(Tensor *a, float exponent) {
  if (!a) {
    ERROR("pow_op: param is invalid");
    return NULL;
  }

  for (int i = 0; i < a->numel; i++) {
    if (a->data[i] < 0.0f && exponent != (int)exponent) {
      ERROR("pow_op: negative base with non-integer exponent");
      return NULL;
    }
    if (a->data[i] == 0.0f && exponent < 0.0f) {
      ERROR("pow_op: zero base with negative exponent");
      return NULL;
    }
  }

  Tensor *r = tensor_init(a->shape, a->ndim);
  if (!r) {
    ERROR("pow_op: tensor_init failed");
    return NULL;
  }

  pow_exponent = exponent;

  for (int i = 0; i < r->numel; i++) {
    r->data[i] = powf(a->data[i], exponent);
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = POW;
  r->backward = backward_pow;

  return r;
}

static void backward_exp(Tensor *self) {
  Tensor *a = self->parents[0];
  for (int i = 0; i < self->numel; i++) {
    if (a)
      a->grad[i] += self->grad[i] * self->data[i];
  }
}

Tensor *exp_t(Tensor *a) {
  if (!a) {
    ERROR("exp_op: param is invalid");
    return NULL;
  }

  Tensor *r = tensor_init(a->shape, a->ndim);
  if (!r) {
    ERROR("exp_op: tensor_init failed");
    return NULL;
  }

  for (int i = 0; i < r->numel; i++) {
    r->data[i] = expf(a->data[i]);
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = EXP;
  r->backward = backward_exp;

  return r;
}

static void backward_log(Tensor *self) {
  Tensor *a = self->parents[0];
  for (int i = 0; i < self->numel; i++) {
    if (a)
      a->grad[i] += self->grad[i] / a->data[i];
  }
}

Tensor *log_t(Tensor *a) {
  if (!a) {
    ERROR("log_op: param is invalid");
    return NULL;
  }

  Tensor *r = tensor_init(a->shape, a->ndim);
  if (!r) {
    ERROR("log_op: tensor_init failed");
    return NULL;
  }

  for (int i = 0; i < r->numel; i++) {
    r->data[i] = logf(a->data[i]);
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = LOG;
  r->backward = backward_log;

  return r;
}

static int find_reduced_dim(Tensor *a, Tensor *out) {
  for (int i = 0, j = 0; i < a->ndim; i++) {
    if (j >= out->ndim || a->shape[i] != out->shape[j]) {
      return i;
    }
    j++;
  }
  return -1;
}

static void backward_sum(Tensor *self) {
  Tensor *a = self->parents[0];

  int dim = find_reduced_dim(a, self);
  if (dim < 0) {
    ERROR("backward_sum: could not determine reduced dimension");
    return;
  }

  int outer = 1, inner = 1;
  int reduce = a->shape[dim];

  for (int i = 0; i < dim; i++)
    outer *= a->shape[i];

  for (int i = dim + 1; i < a->ndim; i++)
    inner *= a->shape[i];

  for (int o = 0; o < outer; o++) {
    int base = o * reduce * inner;
    for (int i = 0; i < inner; i++) {
      float grad = self->grad[o * inner + i];
      for (int r = 0; r < reduce; r++) {
        int idx = base + r * inner + i;
        a->grad[idx] += grad;
      }
    }
  }
}

Tensor *sum_t(Tensor *a, int dim) {
  if (!a || dim >= a->ndim || dim < 0) {
    ERROR("sum_t: param invalid");
    return NULL;
  }

  int out_ndim;
  int64_t shape[a->ndim];

  if (a->ndim == 1) {
    out_ndim = 1;
    shape[0] = 1;
  } else {
    int j = 0;
    for (int i = 0; i < a->ndim; i++) {
      if (i != dim) {
        shape[j++] = a->shape[i];
      }
    }
    out_ndim = a->ndim - 1;
  }

  Tensor *out = tensor_init(shape, out_ndim);
  if (!out) {
    ERROR("sum_t: out failed");
    return NULL;
  }

  int outer = 1, inner = 1, reduce = a->shape[dim];
  for (int i = 0; i < dim; i++)
    outer *= a->shape[i];
  for (int i = dim + 1; i < a->ndim; i++)
    inner *= a->shape[i];

  for (int o = 0; o < outer; o++) {
    for (int i = 0; i < inner; i++) {
      float sum = 0.0f;
      for (int r = 0; r < reduce; r++) {
        sum += a->data[o * reduce * inner + r * inner + i];
      }
      out->data[o * inner + i] = sum;
    }
  }

  out->parents[0] = a;
  out->parents[1] = NULL;
  out->op = SUM_REDUCTION;
  out->backward = backward_sum;

  return out;
}

static void backward_mean(Tensor *self) {
  Tensor *a = self->parents[0];

  int dim = find_reduced_dim(a, self);
  if (dim < 0) {
    ERROR("backward_mean: could not determine reduced dimension");
    return;
  }
  int R = a->shape[dim];

  int outer = 1, inner = 1;
  int reduce = a->shape[dim];

  for (int i = 0; i < dim; i++)
    outer *= a->shape[i];

  for (int i = dim + 1; i < a->ndim; i++)
    inner *= a->shape[i];

  for (int o = 0; o < outer; o++) {
    int base = o * reduce * inner;
    for (int i = 0; i < inner; i++) {
      float grad = self->grad[o * inner + i];
      for (int r = 0; r < reduce; r++) {
        int idx = base + r * inner + i;
        a->grad[idx] += (grad / R);
      }
    }
  }
}
Tensor *mean_t(Tensor *a, int dim) {
  Tensor *r = sum_t(a, dim);
  if (!r) {
    ERROR("mean_t: sum_t failed");
    return NULL;
  }
  int R = a->shape[dim];
  for (int i = 0; i < r->numel; i++)
    r->data[i] /= R;

  r->op = MEAN_REDUCTION;
  r->backward = backward_mean;
  return r;
}

static void backward_dot(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *b = self->parents[1];

  float grad = self->grad[0];
  for (int i = 0; i < a->numel; i++) {
    a->grad[i] += (grad * b->data[i]);
    b->grad[i] += (grad * a->data[i]);
  }
}

Tensor *dot_t(Tensor *a, Tensor *b) {
  if (!a || !b || a->ndim != b->ndim || a->ndim != 1 ||
      a->shape[0] != b->shape[0]) {
    ERROR("dot_t: invalid param");
    return NULL;
  }

  int64_t shape[] = {1};
  Tensor *r = tensor_zeros(shape, 1);
  for (int i = 0; i < a->numel; i++) {
    r->data[0] += (a->data[i] * b->data[i]);
  }

  r->op = DOT;
  r->parents[0] = a;
  r->parents[1] = b;
  r->backward = backward_dot;
  return r;
}

static inline float max_f(float a, float b) { return a > b ? a : b; }
static inline float min_f(float a, float b) { return a < b ? a : b; }

static void backward_max(Tensor *self) {
  Tensor *a = self->parents[0];
  int dim = find_reduced_dim(a, self);
  int outer = 1, inner = 1;
  int reduce = a->shape[dim];

  for (int i = 0; i < dim; i++)
    outer *= a->shape[i];

  for (int i = dim + 1; i < a->ndim; i++)
    inner *= a->shape[i];

  for (int o = 0; o < outer; o++) {
    int base = o * reduce * inner;
    for (int i = 0; i < inner; i++) {
      float grad = self->grad[o * inner + i];
      float max_val = -INFINITY;
      int max_r = 0;

      for (int r = 0; r < reduce; r++) {
        float v = a->data[base + r * inner + i];
        if (v > max_val) {
          max_val = v;
          max_r = r;
        }
      }
      int idx = base + max_r * inner + i;
      a->grad[idx] += grad;
    }
  }
}

Tensor *max_t(Tensor *a, int dim) {
  if (!a || dim >= a->ndim || dim < 0) {
    ERROR("max_t: param invalid");
    return NULL;
  }
  int out_ndim;
  int64_t shape[a->ndim];

  if (a->ndim == 1) {
    out_ndim = 1;
    shape[0] = 1;
  } else {
    int j = 0;
    for (int i = 0; i < a->ndim; i++) {
      if (i != dim) {
        shape[j++] = a->shape[i];
      }
    }
    out_ndim = a->ndim - 1;
  }

  Tensor *out = tensor_init(shape, out_ndim);
  if (!out) {
    ERROR("max_t: out failed");
    return NULL;
  }

  int outer = 1, inner = 1, reduce = a->shape[dim];
  for (int i = 0; i < dim; i++)
    outer *= a->shape[i];
  for (int i = dim + 1; i < a->ndim; i++)
    inner *= a->shape[i];

  for (int o = 0; o < outer; o++) {
    for (int i = 0; i < inner; i++) {
      float m = -INFINITY;
      for (int r = 0; r < reduce; r++) {
        m = max_f(m, a->data[o * reduce * inner + r * inner + i]);
      }
      out->data[o * inner + i] = m;
    }
  }

  out->parents[0] = a;
  out->parents[1] = NULL;
  out->op = MAX;
  out->backward = backward_max;
  return out;
}

static void backward_relu(Tensor *self) {
  Tensor *a = self->parents[0];
  for (int i = 0; i < self->numel; i++) {
    if (a)
      a->grad[i] += self->grad[i] * (a->data[i] > 0.0f ? 1.0f : 0.0f);
  }
}

Tensor *relu_t(Tensor *a) {
  if (!a) {
    ERROR("relu_t: param is invalid");
    return NULL;
  }

  Tensor *r = tensor_init(a->shape, a->ndim);
  if (!r) {
    ERROR("relu_t: tensor_init failed");
    return NULL;
  }

  for (int i = 0; i < r->numel; i++) {
    r->data[i] = a->data[i] > 0.0f ? a->data[i] : 0.0f;
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = RELU;
  r->backward = backward_relu;

  return r;
}

static void backward_gelu(Tensor *self) {
  Tensor *a = self->parents[0];
  static const float SQRT_2_OVER_PI = 0.7978845608028654f;
  static const float COEFF = 0.044715f;

  for (int i = 0; i < self->numel; i++) {
    if (a) {
      float x = a->data[i];
      float x3 = x * x * x;
      float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
      float tanh_inner = tanhf(inner);
      float sech2 = 1.0f - tanh_inner * tanh_inner;
      float d_inner = SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * x * x);
      float grad = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner;
      a->grad[i] += self->grad[i] * grad;
    }
  }
}

Tensor *gelu_t(Tensor *a) {
  if (!a) {
    ERROR("gelu_t: param is invalid");
    return NULL;
  }

  Tensor *r = tensor_init(a->shape, a->ndim);
  if (!r) {
    ERROR("gelu_t: tensor_init failed");
    return NULL;
  }

  static const float SQRT_2_OVER_PI = 0.7978845608028654f;
  static const float COEFF = 0.044715f;

  for (int i = 0; i < r->numel; i++) {
    float x = a->data[i];
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    r->data[i] = 0.5f * x * (1.0f + tanhf(inner));
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = GELU;
  r->backward = backward_gelu;

  return r;
}

static void backward_sigmoid(Tensor *self) {
  Tensor *a = self->parents[0];
  for (int i = 0; i < self->numel; i++) {
    if (a) {
      float sig = self->data[i];
      a->grad[i] += self->grad[i] * sig * (1.0f - sig);
    }
  }
}

Tensor *sigmoid_t(Tensor *a) {
  if (!a) {
    ERROR("sigmoid_t: param is invalid");
    return NULL;
  }

  Tensor *r = tensor_init(a->shape, a->ndim);
  if (!r) {
    ERROR("sigmoid_t: tensor_init failed");
    return NULL;
  }

  for (int i = 0; i < r->numel; i++) {
    r->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = SIGMOID;
  r->backward = backward_sigmoid;

  return r;
}

static void backward_tanh(Tensor *self) {
  Tensor *a = self->parents[0];
  for (int i = 0; i < self->numel; i++) {
    if (a) {
      float th = self->data[i];
      a->grad[i] += self->grad[i] * (1.0f - th * th);
    }
  }
}

Tensor *tanh_t(Tensor *a) {
  if (!a) {
    ERROR("tanh_t: param is invalid");
    return NULL;
  }

  Tensor *r = tensor_init(a->shape, a->ndim);
  if (!r) {
    ERROR("tanh_t: tensor_init failed");
    return NULL;
  }

  for (int i = 0; i < r->numel; i++) {
    r->data[i] = tanhf(a->data[i]);
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = TANH;
  r->backward = backward_tanh;

  return r;
}

static void backward_mse(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *b = self->parents[1];

  float scale = self->grad[0] * (2.0f / a->numel);

  for (int i = 0; i < a->numel; i++) {
    float diff = a->data[i] - b->data[i];

    a->grad[i] += scale * diff;
    b->grad[i] -= scale * diff;
  }
}

Tensor *mseloss_t(Tensor *a, Tensor *b) {
  if (!a || !b || a->numel != b->numel) {
    ERROR("mseloss_t: invalid params");
    return NULL;
  }

  int64_t shape[] = {1};
  Tensor *r = tensor_zeros(shape, 1);
  for (int i = 0; i < a->numel; i++) {
    r->data[0] += ((a->data[i] - b->data[i]) * (a->data[i] - b->data[i]));
  }
  r->data[0] /= a->numel;
  r->op = MSELOSS;
  r->parents[0] = a;
  r->parents[1] = b;
  r->backward = backward_mse;
  return r;
}

static void backward_matmul(Tensor *self) {
  Tensor *a = self->parents[0]; // (m, n)
  Tensor *b = self->parents[1]; // (n, p)

  int m = a->shape[0];
  int n = a->shape[1];
  int p = b->shape[1];

  // dA = dC @ B^T
  for (int i = 0; i < m; i++) {
    for (int k = 0; k < n; k++) {
      float sum = 0.0f;
      for (int j = 0; j < p; j++) {
        sum += self->grad[i * p + j] * b->data[k * p + j]; // B[k,j]
      }
      a->grad[i * n + k] += sum;
    }
  }

  // dB = A^T @ dC
  for (int k = 0; k < n; k++) {
    for (int j = 0; j < p; j++) {
      float sum = 0.0f;
      for (int i = 0; i < m; i++) {
        sum += a->data[i * n + k] * self->grad[i * p + j];
      }
      b->grad[k * p + j] += sum;
    }
  }
}

Tensor *matmul_t(Tensor *a, Tensor *b) {
  if (!a || !b || a->ndim != b->ndim || a->ndim != 2 ||
      a->shape[1] != b->shape[0]) {
    ERROR("matmul_t: invalid param");
    return NULL;
  }
  int64_t result_shape[] = {a->shape[0], b->shape[1]};
  Tensor *r = tensor_zeros(result_shape, 2);

  for (int i = 0; i < a->shape[0]; i++) {
    for (int j = 0; j < b->shape[1]; j++) {
      float sum = 0.0f;
      for (int k = 0; k < a->shape[1]; k++) {
        sum += (a->data[i * a->shape[1] + k] * b->data[k * b->shape[1] + j]);
      }
      r->data[i * result_shape[1] + j] = sum;
    }
  }

  r->op = MATMUL;
  r->parents[0] = a;
  r->parents[1] = b;
  r->backward = backward_matmul;
  return r;
}

void backward_transpose(Tensor *self) {
  Tensor *a = self->parents[0];

  int n = self->shape[0];
  int m = self->shape[1];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      // a[j][i] += self.grad[i][j]
      a->grad[j * a->shape[1] + i] += self->grad[i * m + j];
    }
  }
}

Tensor *transpose_t(Tensor *a) {
  if (!a || a->ndim != 2) {
    ERROR("transpose_t: invalid param");
    return NULL;
  }
  int64_t result_shape[] = {a->shape[1], a->shape[0]};
  Tensor *r = tensor_zeros(result_shape, 2);
  if (!r) {
    ERROR("transpose_t: result tensor failed");
    return NULL;
  }

  for (int i = 0; i < result_shape[0]; i++) {
    for (int j = 0; j < result_shape[1]; j++) {
      r->data[i * result_shape[1] + j] = a->data[j * result_shape[0] + i];
    }
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->backward = backward_transpose;
  r->op = TRANSPOSE;
  return r;
}

static void backward_reshape(Tensor *self) {
  Tensor *a = self->parents[0];
  for (int i = 0; i < self->numel; i++) {
    a->grad[i] += self->grad[i];
  }
}

Tensor *reshape_t(Tensor *a, int64_t *shape, int ndim) {
  if (!a) {
    ERROR("reshape_t: invalid param");
    return NULL;
  }

  int numel = 1;
  for (int i = 0; i < ndim; i++) {
    numel *= shape[i];
  }
  if (numel != a->numel) {
    ERROR("reshape_t: numel does not match");
    return NULL;
  }

  Tensor *r = tensor_zeros(shape, ndim);
  if (!r) {
    ERROR("reshape_t: result tensor failed");
    return NULL;
  }
  for (int i = 0; i < numel; i++) {
    r->data[i] = a->data[i];
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = RESHAPE;
  r->backward = backward_reshape;
  return r;
}

Tensor *squeeze_t(Tensor *a, int dim) {
  if (!a || dim >= a->ndim || dim < 0) {
    ERROR("squeeze_t: invalid params");
    return NULL;
  }

  if (a->shape[dim] != 1) {
    ERROR("squeeze_t: dim != 1");
    return NULL;
  }

  int64_t result_shape[a->ndim - 1];
  for (int i = 0, j = 0; i < a->ndim; i++) {
    if (i != dim) {
      result_shape[j++] = a->shape[i];
    }
  }

  return reshape_t(a, result_shape, a->ndim - 1);
}

Tensor *unsqueeze_t(Tensor *a, int dim) {
  if (!a || dim > a->ndim || dim < 0) {
    ERROR("unsqueeze_t: invalid params");
    return NULL;
  }

  int64_t result_shape[a->ndim + 1];
  for (int i = 0, j = 0; i < a->ndim + 1; i++) {
    if (i == dim) {
      result_shape[i] = 1;
    } else {
      result_shape[i] = a->shape[j++];
    }
  }

  return reshape_t(a, result_shape, a->ndim + 1);
}

static void backward_broadcast(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *r = self;
  int tar_dim = r->ndim;
  int offset = tar_dim - a->ndim;

  int64_t align_shape[tar_dim];
  for (int i = 0; i < tar_dim; i++) {
    align_shape[i] = 1;
  }
  for (int i = tar_dim - a->ndim; i < tar_dim; i++) {
    align_shape[i] = a->shape[i - offset];
  }

  for (int i = 0; i < r->numel; i++) {
    int curr = i;
    int64_t mapped_idx[r->ndim];

    for (int j = r->ndim - 1; j >= 0; j--) {
      int idx = curr % r->shape[j];
      curr = curr / r->shape[j];

      if (align_shape[j] == 1) {
        mapped_idx[j] = 0;
      } else {
        mapped_idx[j] = idx;
      }
    }

    int a_idx = 0;
    for (int k = 0; k < a->ndim; k++) {
      a_idx += a->strides[k] * mapped_idx[k + offset];
    }

    a->grad[a_idx] += r->grad[i];
  }
}

/*
1. Right-align input shape with target shape by padding leading 1s.
2. For each output index, compute its multi-dimensional index.
3. For each dimension:
   - If input dim == 1 → use index 0 (repeat value)
   - Else → use the same index as output
4. Map this to input index and copy value. Ignore the extra dim in align shape
for getting value from a->data Output index → collapse broadcasted dims to 0 →
read from input.
*/
Tensor *broadcast_t(Tensor *a, int64_t *shape, int tar_dim) {
  if (!a || !shape || tar_dim < a->ndim) {
    ERROR("broadcast_t: invalid param");
    return NULL;
  }

  int64_t align_shape[tar_dim];
  for (int i = 0; i < tar_dim; i++) {
    align_shape[i] = 1;
  }

  for (int i = tar_dim - a->ndim; i < tar_dim; i++) {
    align_shape[i] = a->shape[i - (tar_dim - a->ndim)];
  }

  for (int i = 0; i < tar_dim; i++) {
    if (align_shape[i] == 1 || align_shape[i] == shape[i]) {
      continue;
    } else {
      ERROR("broadcast_t: not compatible");
      return NULL;
    }
  }

  Tensor *r = tensor_zeros(shape, tar_dim);

  for (int i = 0; i < r->numel; i++) {
    int curr = i;
    int64_t mapped_idx[r->ndim];

    for (int j = r->ndim - 1; j >= 0; j--) {
      int idx = curr % r->shape[j];
      curr = curr / r->shape[j];

      if (align_shape[j] == 1) {
        mapped_idx[j] = 0;
      } else {
        mapped_idx[j] = idx;
      }
    }

    int a_idx = 0;
    for (int k = a->ndim - 1; k >= 0; k--) {
      a_idx += (a->strides[k] * mapped_idx[k + (tar_dim - a->ndim)]);
    }

    r->data[i] = a->data[a_idx];
  }

  r->op = BROADCAST;
  r->parents[0] = a;
  r->parents[1] = NULL;
  r->backward = backward_broadcast;
  return r;
}

static void backward_crossentropy(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *b = self->parents[1];

  int N = a->shape[0];
  int C = a->shape[1];

  float upstream = self->grad[0];

  for (int i = 0; i < N; i++) {
    float max = a->data[i * C];
    for (int j = 1; j < C; j++) {
      float val = a->data[i * C + j];
      if (val > max)
        max = val;
    }

    float sum = 0.0f;
    float exp_buf[C];

    for (int j = 0; j < C; j++) {
      float e = expf(a->data[i * C + j] - max);
      exp_buf[j] = e;
      sum += e;
    }

    for (int j = 0; j < C; j++) {
      float p = exp_buf[j] / sum;
      float grad = p;

      if (j == (int)b->data[i]) {
        grad -= 1.0f;
      }
      grad = (grad / N) * upstream;
      a->grad[i * C + j] += grad;
    }
  }
}

Tensor *crossentropyloss_t(Tensor *a, Tensor *b) {
  if (!a || !b || a->ndim != 2 || b->ndim != 1 || a->shape[0] != b->shape[0]) {
    ERROR("crossentropyloss_t: invalid params");
    return NULL;
  }

  int N = a->shape[0];
  int C = a->shape[1];

  int64_t shape[] = {1};
  Tensor *result = tensor_zeros(shape, 1);
  if (!result) {
    ERROR("crossentropyloss_t: result allocation failed");
    return NULL;
  }

  float loss = 0.0f;

  for (int i = 0; i < N; i++) {
    float max = a->data[i * C];
    for (int j = 1; j < C; j++) {
      float val = a->data[i * C + j];
      if (val > max)
        max = val;
    }

    float sum = 0.0f;
    for (int j = 0; j < C; j++) {
      sum += expf(a->data[i * C + j] - max);
    }
    float log_sum_exp = max + logf(sum);
    int target = (int)b->data[i];
    loss += -a->data[i * C + target] + log_sum_exp;
  }

  loss /= N;
  result->data[0] = loss;
  result->op = CROSSENTROPY;
  result->parents[0] = a;
  result->parents[1] = b;
  result->backward = backward_crossentropy;
  return result;
}
