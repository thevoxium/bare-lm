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

  if (*visited_count >= *visited_capacity) {
    *visited_capacity *= 2;
    Tensor **tmp = realloc(*visited, sizeof(Tensor *) * (*visited_capacity));
    if (tmp) {
      *result = tmp;
    } else {
      ERROR("build_topo: realloc failed");
      return;
    }
  }

  for (int i = 0; i < *visited_count; i++) {
    if ((*visited)[i] == root) {
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

  (*result)[(*result_count)++] = root;
}

void backward(Tensor *root) {
  if (!root) {
    ERROR("backward: root is NULL");
    return;
  }

  int result_count = 0;
  int result_capacity = 16;
  Tensor **result = malloc(sizeof(Tensor *) * result_capacity);
  if (!result) {
    ERROR("backward: result malloc failed");
    return;
  }

  int visited_count = 0;
  int visited_capacity = 16;
  Tensor **visited = malloc(sizeof(Tensor *) * visited_capacity);
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

  for (int i = 0; i < ndim; i++) {
    t->shape[i] = shape[i];
    t->numel *= shape[i];
  }

  t->data = malloc(t->numel * sizeof(float));
  t->grad = malloc(t->numel * sizeof(float));

  if (!t->data || !t->grad) {
    ERROR("tensor_init: malloc failed");
    free(t->shape);
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
