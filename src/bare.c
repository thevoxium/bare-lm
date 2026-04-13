#include "bare.h"
#include <cblas.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>

Memory *create_global_mem(size_t size) {
  CHECK(size > 0, "create_global_mem: size <= 0");
  Memory *mem = (Memory *)malloc(sizeof(Memory));
  CHECK(mem, "create_global_mem: failed to allocate Memory struct");
  size = (size + ALIGNMENT - 1) & (~(ALIGNMENT - 1));
  CHECK(size <= SIZE_MAX - (ALIGNMENT - 1),
        "create_global_mem: overflow in alignment");

  mem->perm = (Arena *)malloc(sizeof(Arena));
  mem->temp = (Arena *)malloc(sizeof(Arena));
  CHECK(mem->perm && mem->temp, "create_global_mem: could not create arenas");

  mem->perm->buffer = (uint8_t *)aligned_alloc(ALIGNMENT, size);
  mem->temp->buffer = (uint8_t *)aligned_alloc(ALIGNMENT, size);
  CHECK(mem->perm->buffer && mem->temp->buffer,
        "create_global_mem: error allocating buffer");

  mem->perm->size = size;
  mem->perm->used = 0;
  mem->temp->size = size;
  mem->temp->used = 0;

  return mem;
}

void reset_temp_mem(Memory *mem) {
  CHECK_VOID(mem, "reset_temp_mem: mem is NULL");
  mem->temp->used = 0;
}

void *allocate_mem(Memory *mem, size_t size, uint8_t perm) {
  CHECK(mem && size > 0, "allocate_mem: invalid params");

  Arena *arena = perm ? mem->perm : mem->temp;
  CHECK(arena && arena->buffer, "allocate_mem: invalid arena");

  CHECK(arena->used <= SIZE_MAX - (ALIGNMENT - 1),
        "allocate_mem: overflow in alignment");

  size_t aligned_used = (arena->used + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);

  CHECK(aligned_used <= arena->size, "allocate_mem: used > size");
  CHECK(size <= arena->size - aligned_used, "allocate_mem: not enough memory");

  void *ptr = (uint8_t *)arena->buffer + aligned_used;

  arena->used = aligned_used + size;

  return ptr;
}

void free_global_mem(Memory *mem) {
  CHECK_VOID(mem, "free_global_mem: mem is NULL");
  free(mem->perm->buffer);
  free(mem->temp->buffer);
  free(mem->temp);
  free(mem->perm);
  free(mem);
}

ParameterList *create_param_list(Memory *mem) {
  CHECK(mem, "create_param_list: mem is NULL");
  return dt_array_create(mem, PERM);
}

void param_list_add(Memory *mem, ParameterList *pl, Tensor *t) {
  CHECK_VOID(pl && t, "param_list_add: pl or t is NULL");
  return dt_array_push(mem, pl, t, PERM);
}

void zero_grad(ParameterList *pl) {
  CHECK_VOID(pl, "zero_grad: pl is NULL");
  for (int i = 0; i < pl->count; i++) {
    Tensor *t = pl->t[i];
    memset(t->grad, 0, sizeof(float) * t->numel);
  }
}

Dt_array *dt_array_create(Memory *mem, uint8_t perm) {
  CHECK(mem, "dt_array_create: mem is NULL");
  Dt_array *a = (Dt_array *)allocate_mem(mem, sizeof(Dt_array), perm);
  CHECK(a, "dt_array_create: array creation failed");

  a->count = 0;
  a->capacity = 16;
  a->t = (Tensor **)allocate_mem(mem, a->capacity * sizeof(Tensor *), perm);
  CHECK(a->t, "dt_array_create: tensor array creation failed");
  return a;
}

void dt_array_push(Memory *mem, Dt_array *a, Tensor *t, uint8_t perm) {
  CHECK_VOID(mem && a && t, "dt_array_push: NULL params");
  if (a->count >= a->capacity) {
    a->capacity = a->capacity * 2;
    Tensor **tmp = allocate_mem(mem, sizeof(Tensor *) * a->capacity, perm);
    CHECK_VOID(tmp, "dt_array_push: alloc failed");
    for (int i = 0; i < a->count; i++)
      tmp[i] = a->t[i];
    a->t = tmp;
  }
  a->t[a->count++] = t;
}

static void print_data(float *data, int *shape, int *strides, int ndim, int dim,
                       int *indices, int indent) {
  printf("%*s[", indent, "");
  if (dim == ndim - 1) {
    for (int i = 0; i < shape[dim]; i++) {
      indices[dim] = i;
      int idx = 0;
      for (int d = 0; d < ndim; d++)
        idx += indices[d] * strides[d];
      printf("%.4f", data[idx]);
      if (i < shape[dim] - 1)
        printf(", ");
    }
  } else {
    printf("\n");
    for (int i = 0; i < shape[dim]; i++) {
      indices[dim] = i;
      print_data(data, shape, strides, ndim, dim + 1, indices, indent + 2);
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

  int indices[t->ndim];
  for (int i = 0; i < t->ndim; i++)
    indices[i] = 0;

  printf("Tensor(\n");
  print_data(t->data, t->shape, t->strides, t->ndim, 0, indices, 2);
  printf(",\n  shape=[");
  for (int i = 0; i < t->ndim; i++) {
    printf("%lld", (long long)t->shape[i]);
    if (i < t->ndim - 1)
      printf(", ");
  }
  printf("]\n");

  if (grad) {
    printf(",\n  grad=\n");
    for (int i = 0; i < t->ndim; i++)
      indices[i] = 0;
    print_data(t->grad, t->shape, t->strides, t->ndim, 0, indices, 2);
    printf("\n)\n");
  } else {
    printf("\n)\n");
  }
}

static void build_topo(Memory *mem, Tensor *root, Dt_array *result,
                       uint32_t *current_gen) {
  if (!root) {
    return;
  }

  if (root->visited_gen == *current_gen) {
    return;
  } else {
    root->visited_gen = *current_gen;
  }

  for (int i = 0; i < 2; i++) {
    if (root->parents[i]) {
      build_topo(mem, root->parents[i], result, current_gen);
    }
  }

  dt_array_push(mem, result, root, TEMP);
}

void backward(Memory *mem, Tensor *root) {
  CHECK_VOID(root, "backward: root is NULL");

  Dt_array *result = dt_array_create(mem, TEMP);
  CHECK_VOID(result, "backward: failed to create result array");

  static uint32_t gen = 1;
  uint32_t current_gen = gen++;
  build_topo(mem, root, result, &current_gen);

  for (int i = 0; i < root->numel; i++) {
    root->grad[i] = 1.0f;
  }

  for (int i = result->count - 1; i >= 0; i--) {
    if (!result->t[i]) {
      continue;
    }
    if (result->t[i]->backward) {
      result->t[i]->backward(result->t[i]);
    }
  }
}

int element_size(Tensor *a) {
  if (a) {
    return sizeof(a->data[0]);
  }
  return 0;
}

int nbytes(Tensor *a) {
  if (a) {
    return a->numel * sizeof(a->data[0]);
  }
  return 0;
}

uint8_t is_floating_point(Tensor *a) {
  if (a) {
    return (a->dt == fp32) ? 1 : 0;
  }
  return 0;
}

int *tensor_size(Memory *mem, Tensor *a) {
  if (a) {
    int *shape = (int *)allocate_mem(mem, a->ndim * sizeof(int), TEMP);
    for (int i = 0; i < a->ndim; i++) {
      shape[i] = a->shape[i];
    }
    return shape;
  }
  return NULL;
}

float item(Tensor *a) {
  if (a && a->numel == 1) {
    return a->data[0];
  }
  ERROR("item: numel > 1");
  return 0.0f;
}

Tensor *tensor_init(Memory *mem, int *shape, int ndim, uint8_t perm) {
  CHECK(mem && shape && ndim > 0,
        "tensor_init: mem is NULL, shape is NULL, or ndim <= 0");

  Tensor *t = allocate_mem(mem, sizeof(Tensor), perm);
  CHECK(t, "tensor_init: failed to allocate Tensor struct");

  t->ndim = ndim;
  t->numel = 1;

  t->shape = allocate_mem(mem, ndim * sizeof(int), perm);
  CHECK(t->shape, "tensor_init: alloc shape failed");

  t->strides = allocate_mem(mem, ndim * sizeof(int), perm);
  CHECK(t->strides, "tensor_init: alloc strides failed");

  for (int i = 0; i < ndim; i++) {
    CHECK(shape[i] > 0, "tensor_init: why do you have negative shapes?");
    t->shape[i] = shape[i];
    t->numel *= shape[i];
  }

  t->strides[ndim - 1] = 1;
  for (int i = ndim - 2; i >= 0; i--) {
    t->strides[i] = t->strides[i + 1] * shape[i + 1];
  }

  t->data = allocate_mem(mem, t->numel * sizeof(float), perm);
  CHECK(t->data, "tensor_init: alloc data failed");

  t->grad = allocate_mem(mem, t->numel * sizeof(float), perm);
  CHECK(t->grad, "tensor_init: alloc grad failed");

  for (int i = 0; i < t->numel; i++) {
    t->data[i] = 0.0f;
    t->grad[i] = 0.0f;
  }

  t->op = NONE;
  t->parents[0] = NULL;
  t->parents[1] = NULL;
  t->backward = NULL;
  t->dt = fp32;
  t->contiguous = 1;

  return t;
}

Tensor *tensor_zeros(Memory *mem, int *shape, int ndim, uint8_t perm) {
  Tensor *t = tensor_init(mem, shape, ndim, perm);
  CHECK(t, "tensor_zeros: tensor_init failed");
  memset(t->data, 0, sizeof(float) * t->numel);
  return t;
}

Tensor *tensor_ones(Memory *mem, int *shape, int ndim, uint8_t perm) {
  Tensor *t = tensor_init(mem, shape, ndim, perm);
  CHECK(t, "tensor_ones: tensor_init failed");
  for (int i = 0; i < t->numel; i++) {
    t->data[i] = 1.0f;
  }
  return t;
}

// this is called box muller transform, learn about it later on.
Tensor *tensor_randn(Memory *mem, int *shape, int ndim, uint8_t perm) {
  Tensor *t = tensor_init(mem, shape, ndim, perm);
  CHECK(t, "tensor_randn: tensor_init failed");

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

Tensor *tensor_xavier(Memory *mem, int *shape, int ndim, uint8_t perm) {
  CHECK(ndim >= 2, "xavier_init: ndim < 2");
  Tensor *r = tensor_randn(mem, shape, ndim, perm);
  CHECK(r, "xavier_init: tensor_randn failed");

  int f_in, f_out;
  if (ndim == 2) {
    f_in = shape[0];
    f_out = shape[1];
  } else {
    int rf = 1;
    for (int i = 2; i < ndim; i++) {
      rf *= shape[i];
    }
    f_in = shape[0] * rf;
    f_out = shape[1] * rf;
  }

  float scale = sqrtf(2.0f / (f_in + f_out));
  for (int i = 0; i < r->numel; i++) {
    r->data[i] *= scale;
  }

  return r;
}

Tensor *tensor_eye(Memory *mem, int n, int m, uint8_t perm) {
  CHECK(n > 0, "tensor_eye: n must be positive");
  int rows = n;
  int cols = (m > 0) ? m : n;

  int shape[] = {rows, cols};
  Tensor *r = tensor_zeros(mem, shape, 2, perm);
  CHECK(r, "tensor_eye: tensor_zeros failed");

  int min_nm = (rows < cols) ? rows : cols;
  for (int i = 0; i < min_nm; i++) {
    r->data[i * cols + i] = 1.0f;
  }

  return r;
}

Tensor *tensor_full_like(Memory *mem, Tensor *a, float fill_value,
                         uint8_t perm) {
  CHECK(a, "tensor_full_like: a is NULL");
  Tensor *r = tensor_init(mem, a->shape, a->ndim, perm);
  CHECK(r, "tensor_full_like: tensor_init failed");

  for (int i = 0; i < r->numel; i++) {
    r->data[i] = fill_value;
  }

  return r;
}

Tensor *tensor_zeros_like(Memory *mem, Tensor *a, uint8_t perm) {
  CHECK(a, "tensor_zeros_like: a is NULL");
  return tensor_zeros(mem, a->shape, a->ndim, perm);
}

Tensor *tensor_ones_like(Memory *mem, Tensor *a, uint8_t perm) {
  CHECK(a, "tensor_ones_like: a is NULL");
  return tensor_ones(mem, a->shape, a->ndim, perm);
}

Tensor *tensor_randn_like(Memory *mem, Tensor *a, uint8_t perm) {
  CHECK(a, "tensor_ones_like: a is NULL");
  return tensor_randn(mem, a->shape, a->ndim, perm);
}

static inline uint8_t check_op_compatibilty(Tensor *a, Tensor *b) {
  if (!a || !b || a->ndim != b->ndim) {
    return 0;
  }

  for (int i = 0; i < a->ndim; i++) {
    if (a->shape[i] != b->shape[i])
      return 0;
  }
  return 1;
}

static inline void advance_offsets(int ndim, int *idx, int *shape, int n_offs,
                                   int *offs, int **strides) {
  for (int d = ndim - 1; d >= 0; d--) {
    idx[d]++;
    for (int k = 0; k < n_offs; k++) {
      offs[k] += strides[k][d];
    }
    if (idx[d] < shape[d]) {
      break;
    }
    idx[d] = 0;
    for (int k = 0; k < n_offs; k++) {
      offs[k] -= shape[d] * strides[k][d];
    }
  }
}

static void backward_add(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *b = self->parents[1];
  Tensor *r = self;

  int N = r->numel;

  if (a->contiguous && b->contiguous && r->contiguous) {
    for (int i = 0; i < N; i++) {
      a->grad[i] += r->grad[i];
      b->grad[i] += r->grad[i];
    }
    return;
  }

  int ndim = r->ndim;
  int idx[ndim];
  memset(idx, 0, sizeof(idx));
  int offs[2] = {0, 0};
  int *strides[2] = {a->strides, b->strides};

  for (int i = 0; i < N; i++) {
    float g = r->grad[i];
    a->grad[offs[0]] += g;
    b->grad[offs[1]] += g;
    advance_offsets(ndim, idx, r->shape, 2, offs, strides);
  }
}

Tensor *add_t(Memory *mem, Tensor *a, Tensor *b) {
  CHECK(a && b && check_op_compatibilty(a, b),
        "add_t: a is NULL, b is NULL, or tensor sizes do not match");

  Tensor *r = tensor_init(mem, a->shape, a->ndim, TEMP);
  CHECK(r, "add_t: tensor_init failed");

  int N = r->numel;

  if (a->contiguous && b->contiguous) {
    for (int i = 0; i < N; i++) {
      r->data[i] = a->data[i] + b->data[i];
    }
  } else {
    int idx[a->ndim];
    memset(idx, 0, sizeof(idx));
    int offs[2] = {0, 0};
    int *strides[2] = {a->strides, b->strides};
    for (int i = 0; i < N; i++) {
      r->data[i] = a->data[offs[0]] + b->data[offs[1]];
      advance_offsets(a->ndim, idx, r->shape, 2, offs, strides);
    }
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
  Tensor *r = self;

  int N = r->numel;

  if (a->contiguous && b->contiguous && r->contiguous) {
    for (int i = 0; i < N; i++) {
      a->grad[i] += r->grad[i];
      b->grad[i] -= r->grad[i];
    }
    return;
  }

  int ndim = r->ndim;
  int idx[ndim];
  memset(idx, 0, sizeof(idx));
  int offs[2] = {0, 0};
  int *strides[2] = {a->strides, b->strides};

  for (int i = 0; i < N; i++) {
    float g = r->grad[i];
    a->grad[offs[0]] += g;
    b->grad[offs[1]] -= g;
    advance_offsets(ndim, idx, r->shape, 2, offs, strides);
  }
}

Tensor *sub_t(Memory *mem, Tensor *a, Tensor *b) {
  CHECK(a && b && check_op_compatibilty(a, b),
        "sub_t: a is NULL, b is NULL, or tensor sizes do not match");

  Tensor *r = tensor_init(mem, a->shape, a->ndim, TEMP);
  CHECK(r, "sub_t: tensor_init failed");

  int N = r->numel;

  if (a->contiguous && b->contiguous) {
    for (int i = 0; i < N; i++) {
      r->data[i] = a->data[i] - b->data[i];
    }
  } else {
    int idx[a->ndim];
    memset(idx, 0, sizeof(idx));
    int offs[2] = {0, 0};
    int *strides[2] = {a->strides, b->strides};
    for (int i = 0; i < N; i++) {
      r->data[i] = a->data[offs[0]] - b->data[offs[1]];
      advance_offsets(a->ndim, idx, r->shape, 2, offs, strides);
    }
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
  Tensor *r = self;

  int N = r->numel;

  if (a->contiguous && b->contiguous && r->contiguous) {
    for (int i = 0; i < N; i++) {
      a->grad[i] += (r->grad[i] * b->data[i]);
      b->grad[i] += (r->grad[i] * a->data[i]);
    }
    return;
  }

  int ndim = r->ndim;
  int idx[ndim];
  memset(idx, 0, sizeof(idx));
  int offs[2] = {0, 0};
  int *strides[2] = {a->strides, b->strides};

  for (int i = 0; i < N; i++) {
    float g = r->grad[i];
    a->grad[offs[0]] += g * b->data[offs[1]];
    b->grad[offs[1]] += g * a->data[offs[0]];
    advance_offsets(ndim, idx, r->shape, 2, offs, strides);
  }
}

Tensor *mul_t(Memory *mem, Tensor *a, Tensor *b) {
  CHECK(a && b && check_op_compatibilty(a, b),
        "mul_t: a is NULL, b is NULL, or tensor sizes do not match");

  Tensor *r = tensor_init(mem, a->shape, a->ndim, TEMP);
  CHECK(r, "mul_t: tensor_init failed");

  int N = r->numel;

  if (a->contiguous && b->contiguous) {
    for (int i = 0; i < N; i++) {
      r->data[i] = a->data[i] * b->data[i];
    }
  } else {
    int idx[a->ndim];
    memset(idx, 0, sizeof(idx));
    int offs[2] = {0, 0};
    int *strides[2] = {a->strides, b->strides};
    for (int i = 0; i < N; i++) {
      r->data[i] = a->data[offs[0]] * b->data[offs[1]];
      advance_offsets(a->ndim, idx, r->shape, 2, offs, strides);
    }
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
  Tensor *r = self;

  int N = r->numel;

  if (a->contiguous && b->contiguous && r->contiguous) {
    for (int i = 0; i < N; i++) {
      float denom = b->data[i] + 1e-6f;
      a->grad[i] += self->grad[i] / denom;
      b->grad[i] -= self->grad[i] * a->data[i] / (denom * denom);
    }
    return;
  }

  int ndim = r->ndim;
  int idx[ndim];
  memset(idx, 0, sizeof(idx));
  int offs[2] = {0, 0};
  int *strides[2] = {a->strides, b->strides};

  for (int i = 0; i < N; i++) {
    float g = r->grad[i];
    float denom = b->data[offs[1]] + 1e-6f;

    a->grad[offs[0]] += g / denom;
    b->grad[offs[1]] -= g * a->data[offs[0]] / (denom * denom);
    advance_offsets(ndim, idx, r->shape, 2, offs, strides);
  }
}
Tensor *divide_t(Memory *mem, Tensor *a, Tensor *b) {
  CHECK(a && b && check_op_compatibilty(a, b),
        "divide_t: a is NULL, b is NULL, or tensor sizes do not match");

  Tensor *r = tensor_init(mem, a->shape, a->ndim, TEMP);
  CHECK(r, "divide_t: tensor_init failed");

  int N = r->numel;

  if (a->contiguous && b->contiguous) {
    for (int i = 0; i < N; i++) {
      r->data[i] = a->data[i] / (b->data[i] + 1e-6);
    }
  } else {
    int idx[a->ndim];
    memset(idx, 0, sizeof(idx));
    int offs[2] = {0, 0};
    int *strides[2] = {a->strides, b->strides};
    for (int i = 0; i < N; i++) {
      r->data[i] = a->data[offs[0]] / (b->data[offs[1]] + 1e-6);
      advance_offsets(a->ndim, idx, r->shape, 2, offs, strides);
    }
  }

  r->parents[0] = a;
  r->parents[1] = b;
  r->op = DIV;
  r->backward = backward_div;
  return r;
}

static void backward_neg(Tensor *self) {
  Tensor *a = self->parents[0];

  int N = self->numel;
  if (a->contiguous && self->contiguous) {
    for (int i = 0; i < N; i++) {
      a->grad[i] -= self->grad[i];
    }
    return;
  }

  int ndim = self->ndim;
  int idx[ndim];
  memset(idx, 0, sizeof(idx));
  int offs[2] = {0, 0};
  int *strides[2] = {a->strides, self->strides};

  for (int i = 0; i < N; i++) {
    a->grad[offs[0]] -= self->grad[offs[1]];
    advance_offsets(ndim, idx, self->shape, 2, offs, strides);
  }
}

Tensor *neg_t(Memory *mem, Tensor *a) {
  CHECK(a, "neg_t: a is NULL");

  Tensor *r = tensor_init(mem, a->shape, a->ndim, TEMP);
  CHECK(r, "neg_t: tensor_init failed");

  int N = r->numel;

  if (a->contiguous) {
    float *__restrict__ r_data = r->data;
    float *__restrict__ a_data = a->data;
    for (int i = 0; i < N; i++) {
      r_data[i] = -a_data[i];
    }
  } else {
    int idx[a->ndim];
    memset(idx, 0, sizeof(idx));
    int offs[1] = {0};
    int *strides[1] = {a->strides};
    for (int i = 0; i < N; i++) {
      r->data[i] = -a->data[offs[0]];
      advance_offsets(a->ndim, idx, r->shape, 1, offs, strides);
    }
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = NEG;
  r->backward = backward_neg;

  return r;
}

static void backward_pow(Tensor *self) {
  Tensor *a = self->parents[0];
  float pow_exponent = self->op_params[0];

  int N = self->numel;
  if (a->contiguous && self->contiguous) {
    for (int i = 0; i < N; i++) {
      a->grad[i] +=
          self->grad[i] * pow_exponent * powf(a->data[i], pow_exponent - 1);
    }
    return;
  }

  int ndim = self->ndim;
  int idx[ndim];
  memset(idx, 0, sizeof(idx));
  int offs[2] = {0, 0};
  int *strides[2] = {a->strides, self->strides};

  for (int i = 0; i < N; i++) {
    float x = a->data[offs[0]];
    a->grad[offs[0]] +=
        self->grad[offs[1]] * pow_exponent * powf(x, pow_exponent - 1);
    advance_offsets(ndim, idx, self->shape, 2, offs, strides);
  }
}

Tensor *pow_t(Memory *mem, Tensor *a, float exponent) {
  CHECK(a, "pow_t: a is NULL");

  if (a->contiguous) {
    for (int i = 0; i < a->numel; i++) {
      CHECK(!(a->data[i] < 0.0f && exponent != (int)exponent),
            "pow_t: negative base with non-integer exponent");
      CHECK(!(a->data[i] == 0.0f && exponent < 0.0f),
            "pow_t: zero base with negative exponent");
    }
  } else {
    int idx[a->ndim];
    memset(idx, 0, sizeof(idx));
    int offs[1] = {0};
    int *strides[1] = {a->strides};
    for (int i = 0; i < a->numel; i++) {
      float x = a->data[offs[0]];
      CHECK(!(x < 0.0f && exponent != (int)exponent),
            "pow_t: negative base with non-integer exponent");
      CHECK(!(x == 0.0f && exponent < 0.0f),
            "pow_t: zero base with negative exponent");
      advance_offsets(a->ndim, idx, a->shape, 1, offs, strides);
    }
  }

  Tensor *r = tensor_init(mem, a->shape, a->ndim, TEMP);
  CHECK(r, "pow_t: tensor_init failed");

  int N = r->numel;
  if (a->contiguous) {
    for (int i = 0; i < N; i++) {
      r->data[i] = powf(a->data[i], exponent);
    }
  } else {
    int idx[a->ndim];
    memset(idx, 0, sizeof(idx));
    int offs[1] = {0};
    int *strides[1] = {a->strides};
    for (int i = 0; i < N; i++) {
      r->data[i] = powf(a->data[offs[0]], exponent);
      advance_offsets(a->ndim, idx, r->shape, 1, offs, strides);
    }
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = POW;
  r->backward = backward_pow;
  r->op_params[0] = exponent;
  return r;
}

static void backward_exp(Tensor *self) {
  Tensor *a = self->parents[0];

  int N = self->numel;
  if (a->contiguous && self->contiguous) {
    for (int i = 0; i < N; i++) {
      a->grad[i] += self->grad[i] * self->data[i];
    }
    return;
  }

  int ndim = self->ndim;
  int idx[ndim];
  memset(idx, 0, sizeof(idx));
  int offs[2] = {0, 0};
  int *strides[2] = {a->strides, self->strides};

  for (int i = 0; i < N; i++) {
    a->grad[offs[0]] += self->grad[offs[1]] * self->data[offs[1]];
    advance_offsets(ndim, idx, self->shape, 2, offs, strides);
  }
}

Tensor *exp_t(Memory *mem, Tensor *a) {
  CHECK(a, "exp_t: a is NULL");

  Tensor *r = tensor_init(mem, a->shape, a->ndim, TEMP);
  CHECK(r, "exp_t: tensor_init failed");

  int N = r->numel;
  if (a->contiguous) {
    for (int i = 0; i < N; i++) {
      r->data[i] = expf(a->data[i]);
    }
  } else {
    int idx[a->ndim];
    memset(idx, 0, sizeof(idx));
    int offs[1] = {0};
    int *strides[1] = {a->strides};
    for (int i = 0; i < N; i++) {
      r->data[i] = expf(a->data[offs[0]]);
      advance_offsets(a->ndim, idx, r->shape, 1, offs, strides);
    }
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = EXP;
  r->backward = backward_exp;

  return r;
}

static void backward_log(Tensor *self) {
  Tensor *a = self->parents[0];

  int N = self->numel;
  if (a->contiguous && self->contiguous) {
    for (int i = 0; i < N; i++) {
      CHECK_VOID(a->data[i] != 0.0f, "backward_log: 0.0 denominator");
      a->grad[i] += self->grad[i] / a->data[i];
    }
    return;
  }

  int ndim = self->ndim;
  int idx[ndim];
  memset(idx, 0, sizeof(idx));
  int offs[2] = {0, 0};
  int *strides[2] = {a->strides, self->strides};

  for (int i = 0; i < N; i++) {
    float x = a->data[offs[0]];
    CHECK_VOID(x != 0.0f, "backward_log: 0.0 denominator");
    a->grad[offs[0]] += self->grad[offs[1]] / x;
    advance_offsets(ndim, idx, self->shape, 2, offs, strides);
  }
}

Tensor *log_t(Memory *mem, Tensor *a) {
  CHECK(a, "log_t: a is NULL");

  Tensor *r = tensor_init(mem, a->shape, a->ndim, TEMP);
  CHECK(r, "log_t: tensor_init failed");

  int N = r->numel;
  if (a->contiguous) {
    for (int i = 0; i < N; i++) {
      CHECK(a->data[i] > 0.0f, "log_t: negative data");
      r->data[i] = logf(a->data[i]);
    }
  } else {
    int idx[a->ndim];
    memset(idx, 0, sizeof(idx));
    int offs[1] = {0};
    int *strides[1] = {a->strides};
    for (int i = 0; i < N; i++) {
      float x = a->data[offs[0]];
      CHECK(x > 0.0f, "log_t: negative data");
      r->data[i] = logf(x);
      advance_offsets(a->ndim, idx, r->shape, 1, offs, strides);
    }
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = LOG;
  r->backward = backward_log;

  return r;
}

static void backward_sum(Tensor *self) {
  Tensor *r = self;
  Tensor *a = self->parents[0];
  int dim = self->op_params[0];
  int out_dim = self->ndim;

  for (int i = 0; i < r->numel; i++) {
    int curr = i;
    int cord;
    int a_idx = 0;
    for (int d = out_dim - 1; d >= 0; d--) {
      cord = curr % r->shape[d];
      a_idx += cord * a->strides[(d < dim) ? d : d + 1];
      curr /= r->shape[d];
    }

    float g = r->grad[i];
    for (int k = 0; k < a->shape[dim]; k++) {
      int final_idx = a_idx + k * a->strides[dim];
      a->grad[final_idx] += g;
    }
  }
}

Tensor *sum_t(Memory *mem, Tensor *a, int dim) {
  CHECK(a && dim >= 0 && dim < a->ndim,
        "sum_t: a is NULL or dim out of bounds");

  int out_shape[a->ndim];
  int out_dim;
  if (a->ndim == 1) {
    out_shape[0] = 1;
    out_dim = 1;
  } else {
    for (int d = 0, j = 0; d < a->ndim; d++) {
      if (d != dim) {
        out_shape[j++] = a->shape[d];
      }
    }
    out_dim = a->ndim - 1;
  }

  Tensor *r = tensor_zeros(mem, out_shape, out_dim, TEMP);
  CHECK(r, "sum_t: failed to create output tensor");

  for (int i = 0; i < r->numel; i++) {
    int curr = i;
    int cord;
    int a_idx = 0;
    for (int d = out_dim - 1; d >= 0; d--) {
      cord = curr % r->shape[d];
      a_idx += cord * a->strides[(d < dim) ? d : d + 1];
      curr /= r->shape[d];
    }

    float sum = 0.0f;
    for (int k = 0; k < a->shape[dim]; k++) {
      int final_idx = a_idx + k * a->strides[dim];
      sum += a->data[final_idx];
    }
    r->data[i] = sum;
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op_params[0] = dim;
  r->op = SUM_REDUCTION;
  r->backward = backward_sum;

  return r;
}

static void backward_mean(Tensor *self) {
  Tensor *r = self;
  Tensor *a = self->parents[0];
  int dim = self->op_params[0];
  int out_dim = self->ndim;
  int R = a->shape[dim];

  for (int i = 0; i < r->numel; i++) {
    int curr = i;
    int cord;
    int a_idx = 0;
    for (int d = out_dim - 1; d >= 0; d--) {
      cord = curr % r->shape[d];
      a_idx += cord * a->strides[(d < dim) ? d : d + 1];
      curr /= r->shape[d];
    }

    float g = r->grad[i];
    for (int k = 0; k < R; k++) {
      int final_idx = a_idx + k * a->strides[dim];
      a->grad[final_idx] += g / R;
    }
  }
}

Tensor *mean_t(Memory *mem, Tensor *a, int dim) {
  Tensor *r = sum_t(mem, a, dim);
  CHECK(r, "mean_t: sum_t failed");
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

Tensor *dot_t(Memory *mem, Tensor *a, Tensor *b) {
  CHECK(
      a && b && a->ndim == b->ndim && a->ndim == 1 &&
          a->shape[0] == b->shape[0],
      "dot_t: a is NULL, b is NULL, or tensors are not 1D with matching sizes");

  int shape[] = {1};
  Tensor *r = tensor_zeros(mem, shape, 1, TEMP);
  for (int i = 0; i < a->numel; i++) {
    r->data[0] += (a->data[i] * b->data[i]);
  }

  r->op = DOT;
  r->parents[0] = a;
  r->parents[1] = b;
  r->backward = backward_dot;
  return r;
}

static void backward_max(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *r = self;
  int out_ndim = r->ndim;
  int dim = self->op_params[0];

  for (int i = 0; i < r->numel; i++) {
    int curr = i;
    int cord;
    int a_idx = 0;
    for (int d = out_ndim - 1; d >= 0; d--) {
      cord = curr % r->shape[d];
      a_idx += cord * a->strides[(d < dim) ? d : d + 1];
      curr /= r->shape[d];
    }

    float m = -INFINITY;
    int m_idx = -1;
    for (int k = 0; k < a->shape[dim]; k++) {
      int final_idx = a_idx + k * a->strides[dim];
      if (m < a->data[final_idx]) {
        m = a->data[final_idx];
        m_idx = final_idx;
      }
    }
    if (m_idx != -1)
      a->grad[m_idx] += r->grad[i];
  }
}

Tensor *max_t(Memory *mem, Tensor *a, int dim) {
  CHECK(a && dim >= 0 && dim < a->ndim,
        "max_t: a is NULL or dim out of bounds");
  int out_ndim;
  int shape[a->ndim];

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

  Tensor *r = tensor_init(mem, shape, out_ndim, TEMP);
  CHECK(r, "max_t: failed to create out tensor");

  for (int i = 0; i < r->numel; i++) {
    int curr = i;
    int cord;
    int a_idx = 0;
    for (int d = out_ndim - 1; d >= 0; d--) {
      cord = curr % r->shape[d];
      a_idx += cord * a->strides[(d < dim) ? d : d + 1];
      curr /= r->shape[d];
    }

    float m = -INFINITY;
    for (int k = 0; k < a->shape[dim]; k++) {
      int final_idx = a_idx + k * a->strides[dim];
      if (m < a->data[final_idx])
        m = a->data[final_idx];
    }
    r->data[i] = m;
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op_params[0] = dim;
  r->op = MAX;
  r->backward = backward_max;
  return r;
}

static void backward_relu(Tensor *self) {
  Tensor *a = self->parents[0];
  int N = self->numel;

  if (a->contiguous && self->contiguous) {
    for (int i = 0; i < N; i++) {
      a->grad[i] += self->grad[i] * (a->data[i] > 0.0f ? 1.0f : 0.0f);
    }
    return;
  }

  int ndim = self->ndim;
  int idx[ndim];
  memset(idx, 0, sizeof(idx));
  int offs[2] = {0, 0};
  int *strides[2] = {a->strides, self->strides};

  for (int i = 0; i < N; i++) {
    a->grad[offs[0]] +=
        self->grad[offs[1]] * (a->data[offs[0]] > 0.0f ? 1.0f : 0.0f);
    advance_offsets(ndim, idx, self->shape, 2, offs, strides);
  }
}

Tensor *relu_t(Memory *mem, Tensor *a) {
  CHECK(a, "relu_t: a is NULL");

  Tensor *r = tensor_init(mem, a->shape, a->ndim, TEMP);
  CHECK(r, "relu_t: tensor_init failed");

  int N = r->numel;

  if (a->contiguous) {
    float *__restrict__ r_data = r->data;
    float *__restrict__ a_data = a->data;
    for (int i = 0; i < N; i++) {
      r_data[i] = a_data[i] > 0.0f ? a_data[i] : 0.0f;
    }
  } else {
    int idx[a->ndim];
    memset(idx, 0, sizeof(idx));
    int offs[1] = {0};
    int *strides[1] = {a->strides};
    for (int i = 0; i < N; i++) {
      float x = a->data[offs[0]];
      r->data[i] = x > 0.0f ? x : 0.0f;
      advance_offsets(a->ndim, idx, r->shape, 1, offs, strides);
    }
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

  int N = self->numel;
  if (a->contiguous && self->contiguous) {
    for (int i = 0; i < N; i++) {
      float x = a->data[i];
      float x3 = x * x * x;
      float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
      float tanh_inner = tanhf(inner);
      float sech2 = 1.0f - tanh_inner * tanh_inner;
      float d_inner = SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * x * x);
      float grad = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner;
      a->grad[i] += self->grad[i] * grad;
    }
    return;
  }

  int ndim = self->ndim;
  int idx[ndim];
  memset(idx, 0, sizeof(idx));
  int offs[2] = {0, 0};
  int *strides[2] = {a->strides, self->strides};

  for (int i = 0; i < N; i++) {
    float x = a->data[offs[0]];
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    float tanh_inner = tanhf(inner);
    float sech2 = 1.0f - tanh_inner * tanh_inner;
    float d_inner = SQRT_2_OVER_PI * (1.0f + 3.0f * COEFF * x * x);
    float grad = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2 * d_inner;
    a->grad[offs[0]] += self->grad[offs[1]] * grad;
    advance_offsets(ndim, idx, self->shape, 2, offs, strides);
  }
}

Tensor *gelu_t(Memory *mem, Tensor *a) {
  CHECK(a, "gelu_t: a is NULL");

  Tensor *r = tensor_init(mem, a->shape, a->ndim, TEMP);
  CHECK(r, "gelu_t: tensor_init failed");

  static const float SQRT_2_OVER_PI = 0.7978845608028654f;
  static const float COEFF = 0.044715f;

  int N = r->numel;
  if (a->contiguous) {
    for (int i = 0; i < N; i++) {
      float x = a->data[i];
      float x3 = x * x * x;
      float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
      r->data[i] = 0.5f * x * (1.0f + tanhf(inner));
    }
  } else {
    int idx[a->ndim];
    memset(idx, 0, sizeof(idx));
    int offs[1] = {0};
    int *strides[1] = {a->strides};
    for (int i = 0; i < N; i++) {
      float x = a->data[offs[0]];
      float x3 = x * x * x;
      float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
      r->data[i] = 0.5f * x * (1.0f + tanhf(inner));
      advance_offsets(a->ndim, idx, r->shape, 1, offs, strides);
    }
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = GELU;
  r->backward = backward_gelu;

  return r;
}

static void backward_sigmoid(Tensor *self) {
  Tensor *a = self->parents[0];
  int N = self->numel;

  if (a->contiguous && self->contiguous) {
    for (int i = 0; i < N; i++) {
      float sig = self->data[i];
      a->grad[i] += self->grad[i] * sig * (1.0f - sig);
    }
    return;
  }

  int ndim = self->ndim;
  int idx[ndim];
  memset(idx, 0, sizeof(idx));
  int offs[2] = {0, 0};
  int *strides[2] = {a->strides, self->strides};

  for (int i = 0; i < N; i++) {
    float sig = self->data[offs[1]];
    a->grad[offs[0]] += self->grad[offs[1]] * sig * (1.0f - sig);
    advance_offsets(ndim, idx, self->shape, 2, offs, strides);
  }
}

Tensor *sigmoid_t(Memory *mem, Tensor *a) {
  CHECK(a, "sigmoid_t: a is NULL");

  Tensor *r = tensor_init(mem, a->shape, a->ndim, TEMP);
  CHECK(r, "sigmoid_t: tensor_init failed");

  int N = r->numel;
  if (a->contiguous) {
    for (int i = 0; i < N; i++) {
      r->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
    }
  } else {
    int idx[a->ndim];
    memset(idx, 0, sizeof(idx));
    int offs[1] = {0};
    int *strides[1] = {a->strides};
    for (int i = 0; i < N; i++) {
      float x = a->data[offs[0]];
      r->data[i] = 1.0f / (1.0f + expf(-x));
      advance_offsets(a->ndim, idx, r->shape, 1, offs, strides);
    }
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = SIGMOID;
  r->backward = backward_sigmoid;

  return r;
}

static void backward_tanh(Tensor *self) {
  Tensor *a = self->parents[0];
  int N = self->numel;

  if (a->contiguous && self->contiguous) {
    for (int i = 0; i < N; i++) {
      float th = self->data[i];
      a->grad[i] += self->grad[i] * (1.0f - th * th);
    }
    return;
  }

  int ndim = self->ndim;
  int idx[ndim];
  memset(idx, 0, sizeof(idx));
  int offs[2] = {0, 0};
  int *strides[2] = {a->strides, self->strides};

  for (int i = 0; i < N; i++) {
    float th = self->data[offs[1]];
    a->grad[offs[0]] += self->grad[offs[1]] * (1.0f - th * th);
    advance_offsets(ndim, idx, self->shape, 2, offs, strides);
  }
}

Tensor *tanh_t(Memory *mem, Tensor *a) {
  CHECK(a, "tanh_t: a is NULL");

  Tensor *r = tensor_init(mem, a->shape, a->ndim, TEMP);
  CHECK(r, "tanh_t: tensor_init failed");

  int N = r->numel;
  if (a->contiguous) {
    for (int i = 0; i < N; i++) {
      r->data[i] = tanhf(a->data[i]);
    }
  } else {
    int idx[a->ndim];
    memset(idx, 0, sizeof(idx));
    int offs[1] = {0};
    int *strides[1] = {a->strides};
    for (int i = 0; i < N; i++) {
      r->data[i] = tanhf(a->data[offs[0]]);
      advance_offsets(a->ndim, idx, r->shape, 1, offs, strides);
    }
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

Tensor *mseloss_t(Memory *mem, Tensor *a, Tensor *b) {
  CHECK(a && b && check_op_compatibilty(a, b),
        "mseloss_t: a is NULL, b is NULL, or tensor sizes do not match");

  int shape[] = {1};
  Tensor *r = tensor_zeros(mem, shape, 1, TEMP);
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
  Tensor *a = self->parents[0]; // (M, K)
  Tensor *b = self->parents[1]; // (K, N)

  int M = a->shape[0];
  int K = a->shape[1];
  int N = b->shape[1];

  // dA = dC @ B^T  -> (M, N) @ (N, K) = (M, K)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, K, N, 1.0f,
              self->grad, N, b->data, N, 1.0f, a->grad, K);

  // dB = A^T @ dC  -> (K, M) @ (M, N) = (K, N)
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, N, M, 1.0f, a->data,
              K, self->grad, N, 1.0f, b->grad, N);
}

Tensor *matmul_t(Memory *mem, Tensor *a, Tensor *b) {
  CHECK(a && b && a->ndim == b->ndim && a->ndim == 2 &&
            a->shape[1] == b->shape[0],
        "matmul_t: a is NULL, b is NULL, or tensors are not compatible 2D "
        "matrices");
  int M = a->shape[0];
  int K = a->shape[1];
  int N = b->shape[1];

  int result_shape[] = {M, N};
  Tensor *r = tensor_zeros(mem, result_shape, 2, TEMP);

  float *__restrict__ r_data = r->data;
  float *__restrict__ a_data = a->data;
  float *__restrict__ b_data = b->data;

  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, a_data,
              K, b_data, N, 0.0f, r_data, N);

  r->op = MATMUL;
  r->parents[0] = a;
  r->parents[1] = b;
  r->backward = backward_matmul;
  return r;
}

static void backward_bmm(Tensor *self) {
  Tensor *a = self->parents[0]; // (B, M, K)
  Tensor *b = self->parents[1]; // (B, K, N)

  Tensor *r = self;

  int B = a->shape[0];
  int M = a->shape[1];
  int K = a->shape[2];
  int N = b->shape[b->ndim - 1];

  int a_stride = a->strides[0];
  int b_stride = b->strides[0];
  int r_stride = r->strides[0];

  for (int i = 0; i < B; i++) {
    float *a_grad = a->grad + i * a_stride; // M, k
    float *a_data = a->data + i * a_stride;
    float *b_data = b->data + i * b_stride; // K, N
    float *b_grad = b->grad + i * b_stride;
    float *r_grad = r->grad + i * r_stride; // M, N

    // sgemm(M, N, K)
    // C(M × N) = A(M × K) × B(K × N)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, K, N, 1.0f, r_grad,
                N, b_data, N, 1.0f, a_grad, K);

    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, N, M, 1.0f, a_data,
                K, r_grad, N, 1.0f, b_grad, N);
  }
}

Tensor *bmm_t(Memory *mem, Tensor *a, Tensor *b) {
  CHECK(a && b && a->ndim == 3 && b->ndim == 3, "bmm_t: invalid params");

  CHECK(a->shape[0] == b->shape[0], "bmm_t: batch size mismatch");
  CHECK(a->shape[2] == b->shape[1], "bmm_t: K mismatch");

  int B = a->shape[0];
  int M = a->shape[1];
  int K = a->shape[2];
  int N = b->shape[b->ndim - 1];

  int out_shape[] = {B, M, N};
  Tensor *r = tensor_zeros(mem, out_shape, 3, TEMP);

  int a_stride = a->strides[0];
  int b_stride = b->strides[0];
  int r_stride = r->strides[0];

  for (int i = 0; i < B; i++) {
    float *a_data = a->data + i * a_stride; // M, k
    float *b_data = b->data + i * b_stride; // K, N
    float *r_data = r->data + i * r_stride; // M, N

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f,
                a_data, K, b_data, N, 0.0f, r_data, N);
  }

  r->op = BMM;
  r->parents[0] = a;
  r->parents[1] = b;
  r->backward = backward_bmm;
  return r;
}

static void backward_transpose(Tensor *self) {
  Tensor *a = self->parents[0];

  int n = self->shape[0];
  int m = self->shape[1];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < m; j++) {
      int r_idx = i * self->strides[0] + j * self->strides[1];
      int a_idx = j * a->strides[0] + i * a->strides[1];
      a->grad[a_idx] += self->grad[r_idx];
    }
  }
}

Tensor *transpose_t(Memory *mem, Tensor *a) {
  CHECK(a && a->ndim == 2, "transpose_t: a is NULL or not a 2D tensor");

  Tensor *r = (Tensor *)allocate_mem(mem, sizeof(Tensor), TEMP);
  CHECK(r, "transpose_t: allocate_mem failed");
  r->data = a->data;
  r->grad = (float *)allocate_mem(mem, sizeof(float) * a->numel, TEMP);
  r->shape = (int *)allocate_mem(mem, sizeof(int) * a->ndim, TEMP);
  r->strides = (int *)allocate_mem(mem, sizeof(int) * a->ndim, TEMP);
  CHECK(r->grad && r->shape && r->strides,
        "transpose_t: grad, shape, stride allocation failed");
  r->shape[0] = a->shape[1];
  r->shape[1] = a->shape[0];
  r->strides[0] = a->strides[1];
  r->strides[1] = a->strides[0];
  r->ndim = a->ndim;
  r->numel = a->numel;
  r->visited_gen = a->visited_gen;
  r->contiguous = 0;
  r->dt = fp32;
  r->parents[0] = a;
  r->parents[1] = NULL;
  r->backward = backward_transpose;
  r->op = TRANSPOSE;
  return r;
}

static void backward_reshape(Tensor *self) {
  Tensor *a = self->parents[0];
  if (a->contiguous) {
    for (int i = 0; i < self->numel; i++) {
      a->grad[i] += self->grad[i];
    }
    return;
  }

  int idx[a->ndim];
  memset(idx, 0, sizeof(idx));
  int offs[1] = {0};
  int *strides[1] = {a->strides};
  for (int i = 0; i < self->numel; i++) {
    a->grad[offs[0]] += self->grad[i];
    advance_offsets(a->ndim, idx, a->shape, 1, offs, strides);
  }
}

Tensor *reshape_t(Memory *mem, Tensor *a, int *shape, int ndim) {
  CHECK(a, "reshape_t: a is NULL");

  int numel = 1;
  for (int i = 0; i < ndim; i++) {
    numel *= shape[i];
  }
  CHECK(numel == a->numel, "reshape_t: numel does not match");

  Tensor *r = tensor_zeros(mem, shape, ndim, TEMP);
  CHECK(r, "reshape_t: result tensor failed");

  if (a->contiguous) {
    for (int i = 0; i < numel; i++) {
      r->data[i] = a->data[i];
    }
  } else {
    int idx[a->ndim];
    memset(idx, 0, sizeof(idx));
    int offs[1] = {0};
    int *strides[1] = {a->strides};
    for (int i = 0; i < numel; i++) {
      r->data[i] = a->data[offs[0]];
      advance_offsets(a->ndim, idx, a->shape, 1, offs, strides);
    }
  }

  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op = RESHAPE;
  r->backward = backward_reshape;
  return r;
}

Tensor *view_t(Memory *mem, Tensor *a, int *shape, int ndim) {
  return reshape_t(mem, a, shape, ndim);
}

Tensor *squeeze_t(Memory *mem, Tensor *a, int dim) {
  CHECK(a && dim >= 0 && dim < a->ndim,
        "squeeze_t: a is NULL or dim out of bounds");

  CHECK(a->shape[dim] == 1, "squeeze_t: dimension to squeeze must have size 1");

  int result_shape[a->ndim - 1];
  for (int i = 0, j = 0; i < a->ndim; i++) {
    if (i != dim) {
      result_shape[j++] = a->shape[i];
    }
  }

  return reshape_t(mem, a, result_shape, a->ndim - 1);
}

Tensor *unsqueeze_t(Memory *mem, Tensor *a, int dim) {
  CHECK(a && dim >= 0 && dim <= a->ndim,
        "unsqueeze_t: a is NULL or dim out of bounds");

  int result_shape[a->ndim + 1];
  for (int i = 0, j = 0; i < a->ndim + 1; i++) {
    if (i == dim) {
      result_shape[i] = 1;
    } else {
      result_shape[i] = a->shape[j++];
    }
  }

  return reshape_t(mem, a, result_shape, a->ndim + 1);
}

static void backward_broadcast(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *r = self;
  int tar_dim = r->ndim;
  int offset = tar_dim - a->ndim;

  int idx[tar_dim];
  memset(idx, 0, sizeof(idx));
  for (int i = 0; i < r->numel; i++) {
    int a_idx = 0;
    for (int d = 0; d < tar_dim; d++) {
      int a_dim = d - offset;
      if (a_dim >= 0 && a->shape[a_dim] != 1) {
        a_idx += idx[d] * a->strides[a_dim];
      }
    }
    a->grad[a_idx] += r->grad[i];
    for (int d = r->ndim - 1; d >= 0; d--) {
      idx[d]++;
      if (idx[d] < r->shape[d])
        break;
      idx[d] = 0;
    }
  }
}

Tensor *broadcast_t(Memory *mem, Tensor *a, int *shape, int tar_dim) {
  CHECK(a && shape && tar_dim >= a->ndim,
        "broadcast_t: a is NULL, shape is NULL, or tar_dim < a->ndim");

  int align_shape[tar_dim];
  for (int i = 0; i < tar_dim; i++) {
    align_shape[i] = 1;
  }

  for (int i = tar_dim - a->ndim; i < tar_dim; i++) {
    align_shape[i] = a->shape[i - (tar_dim - a->ndim)];
  }

  for (int i = 0; i < tar_dim; i++) {
    CHECK(align_shape[i] == 1 || align_shape[i] == shape[i],
          "broadcast_t: not compatible");
  }

  int numel = 1;
  for (int i = 0; i < tar_dim; i++) {
    numel *= shape[i];
  }

  Tensor *r = (Tensor *)allocate_mem(mem, sizeof(Tensor), TEMP);
  CHECK(r, "broadcast_t: allocate_mem failed");
  r->data = a->data;
  r->grad = (float *)allocate_mem(mem, sizeof(float) * numel, TEMP);
  r->shape = (int *)allocate_mem(mem, sizeof(int) * tar_dim, TEMP);
  r->strides = (int *)allocate_mem(mem, sizeof(int) * tar_dim, TEMP);
  CHECK(r->grad && r->shape && r->strides,
        "broadcast_t: grad, shape, stride allocation failed");
  memset(r->grad, 0, sizeof(float) * numel);
  for (int i = 0; i < tar_dim; i++) {
    r->shape[i] = shape[i];
  }
  for (int i = 0; i < tar_dim; i++) {
    if (align_shape[i] == 1)
      r->strides[i] = 0;
    else
      r->strides[i] = a->strides[i - (tar_dim - a->ndim)];
  }

  r->ndim = tar_dim;
  r->numel = numel;
  r->visited_gen = a->visited_gen;
  r->contiguous = 0;
  r->dt = fp32;
  r->parents[0] = a;
  r->parents[1] = NULL;
  r->backward = backward_broadcast;
  r->op = BROADCAST;
  return r;
}

Tensor *expand_t(Memory *mem, Tensor *a, int *shape, int tar_dim) {
  return broadcast_t(mem, a, shape, tar_dim);
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

Tensor *crossentropyloss_t(Memory *mem, Tensor *a, Tensor *b) {
  CHECK(a && b && a->ndim == 2 && b->ndim == 1 && a->shape[0] == b->shape[0],
        "crossentropyloss_t: a is NULL, b is NULL, or shapes are incompatible");

  int N = a->shape[0];
  int C = a->shape[1];

  int shape[] = {1};
  Tensor *result = tensor_zeros(mem, shape, 1, TEMP);
  CHECK(result, "crossentropyloss_t: result allocation failed");

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

Linear *create_linear(Memory *mem, ParameterList *pl, int d_in, int d_out) {
  CHECK(d_out > 0 && d_in > 0,
        "create_linear: d_out and d_in must be positive");
  Linear *l = (Linear *)allocate_mem(mem, sizeof(Linear), PERM);
  CHECK(l, "create_linear: failed to allocate Linear struct");

  int weight_shape[] = {d_out, d_in};
  int bias_shape[] = {d_out};
  l->weights = tensor_randn(mem, weight_shape, 2, PERM);
  CHECK(l->weights, "create_linear: failed to create weights tensor");

  l->bias = tensor_randn(mem, bias_shape, 1, PERM);
  CHECK(l->bias, "create_linear: failed to create bias tensor");

  param_list_add(mem, pl, l->weights);
  param_list_add(mem, pl, l->bias);

  return l;
}

Tensor *linear_t(Memory *mem, Linear *l, Tensor *x) {
  CHECK(x, "linear_t: x is NULL");

  Tensor *W = l->weights;
  Tensor *b = l->bias;
  Tensor *wT = transpose_t(mem, W);

  int shape[] = {x->shape[0], l->weights->shape[0]};

  b = broadcast_t(mem, b, shape, 2);
  Tensor *out = matmul_t(mem, x, wT);
  out = add_t(mem, out, b);
  return out;
}

LayerNorm *create_layernorm(Memory *mem, ParameterList *pl,
                            int normalized_shape, float eps) {
  CHECK(normalized_shape > 0,
        "create_layernorm: normalized_shape must be positive");
  LayerNorm *ln = (LayerNorm *)allocate_mem(mem, sizeof(LayerNorm), PERM);
  CHECK(ln, "create_layernorm: failed to allocate LayerNorm struct");

  int shape[] = {normalized_shape};
  ln->weight = tensor_ones(mem, shape, 1, PERM);
  CHECK(ln->weight, "create_layernorm: failed to create weight tensor");

  ln->bias = tensor_zeros(mem, shape, 1, PERM);
  CHECK(ln->bias, "create_layernorm: failed to create bias tensor");

  ln->eps = eps;

  param_list_add(mem, pl, ln->weight);
  param_list_add(mem, pl, ln->bias);

  return ln;
}

Tensor *layernorm_t(Memory *mem, LayerNorm *ln, Tensor *x) {
  CHECK(x, "layernorm_t: x is NULL");

  int last_dim = x->ndim - 1;

  Tensor *mean = mean_t(mem, x, last_dim);

  int mean_b_shape[x->ndim];
  for (int i = 0; i < x->ndim - 1; i++) {
    mean_b_shape[i] = x->shape[i];
  }
  mean_b_shape[x->ndim - 1] = 1;
  Tensor *mean_b = reshape_t(mem, mean, mean_b_shape, x->ndim);
  Tensor *mean_bc = broadcast_t(mem, mean_b, x->shape, x->ndim);

  Tensor *diff = sub_t(mem, x, mean_bc);
  Tensor *diff_sq = pow_t(mem, diff, 2.0f);
  Tensor *var = mean_t(mem, diff_sq, last_dim);

  Tensor *eps_t = tensor_zeros(mem, var->shape, var->ndim, TEMP);
  for (int i = 0; i < eps_t->numel; i++)
    eps_t->data[i] = ln->eps;

  Tensor *var_plus_eps = add_t(mem, var, eps_t);
  Tensor *std = pow_t(mem, var_plus_eps, 0.5f);

  int std_b_shape[x->ndim];
  for (int i = 0; i < x->ndim - 1; i++) {
    std_b_shape[i] = x->shape[i];
  }
  std_b_shape[x->ndim - 1] = 1;
  Tensor *std_b = reshape_t(mem, std, std_b_shape, x->ndim);
  Tensor *std_bc = broadcast_t(mem, std_b, x->shape, x->ndim);

  Tensor *normed = divide_t(mem, diff, std_bc);

  int weight_b_shape[x->ndim];
  for (int i = 0; i < x->ndim - 1; i++) {
    weight_b_shape[i] = 1;
  }
  weight_b_shape[x->ndim - 1] = x->shape[x->ndim - 1];
  Tensor *weight_b = reshape_t(mem, ln->weight, weight_b_shape, x->ndim);
  Tensor *weight_bc = broadcast_t(mem, weight_b, x->shape, x->ndim);
  Tensor *out = mul_t(mem, weight_bc, normed);

  int bias_b_shape[x->ndim];
  for (int i = 0; i < x->ndim - 1; i++) {
    bias_b_shape[i] = 1;
  }
  bias_b_shape[x->ndim - 1] = x->shape[x->ndim - 1];
  Tensor *bias_b = reshape_t(mem, ln->bias, bias_b_shape, x->ndim);
  Tensor *bias_bc = broadcast_t(mem, bias_b, x->shape, x->ndim);
  out = add_t(mem, out, bias_bc);

  return out;
}

void sgd_step(ParameterList *pl, float lr) {
  CHECK_VOID(pl, "sgd_step: pl is NULL");
  for (int i = 0; i < pl->count; i++) {
    Tensor *t = pl->t[i];
    for (int j = 0; j < t->numel; j++) {
      t->data[j] -= (lr * t->grad[j]);
    }
  }
}

static void backward_mask(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *b = self->parents[1];

  for (int i = 0; i < a->numel; i++) {
    if (b->data[i] == 0) {
      a->grad[i] += (self->grad[i]);
    }
  }
}

Tensor *mask_t(Memory *mem, Tensor *a, Tensor *b, float val) {
  CHECK(a && b && a->numel == b->numel && a->ndim == b->ndim,
        "mask_t: a is NULL, b is NULL, or tensor sizes do not match");

  Tensor *r = tensor_zeros(mem, a->shape, a->ndim, TEMP);
  CHECK(r, "mask_t: tensor_zeros failed");

  for (int i = 0; i < a->numel; i++) {
    if (b->data[i] == 1) {
      r->data[i] = val;
    } else {
      r->data[i] = a->data[i];
    }
  }

  r->op = MASK;
  r->parents[0] = a;
  r->parents[1] = b;
  r->backward = backward_mask;
  return r;
}

static void backward_scale(Tensor *self) {
  Tensor *a = self->parents[0];
  float v = self->op_params[0];
  for (int i = 0; i < a->numel; i++) {
    a->grad[i] += (self->grad[i] * v);
  }
}

Tensor *scale_t(Memory *mem, Tensor *a, float v) {
  CHECK(a, "scale_t: a is NULL");
  Tensor *r = tensor_zeros(mem, a->shape, a->ndim, TEMP);
  CHECK(r, "scale_t: failed to create result tensor");
  for (int i = 0; i < a->numel; i++) {
    r->data[i] = a->data[i] * v;
  }

  r->op = SCALE;
  r->parents[0] = a;
  r->parents[1] = NULL;
  r->op_params[0] = v;
  r->backward = backward_scale;

  return r;
}

static void backward_concat(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *b = self->parents[1];
  Tensor *r = self;
  int dim = self->op_params[0];

  for (int i = 0; i < r->numel; i++) {
    int curr_idx = i;
    int idx[r->ndim];

    for (int d = r->ndim - 1; d >= 0; d--) {
      idx[d] = curr_idx % r->shape[d];
      curr_idx /= r->shape[d];
    }

    if (idx[dim] < a->shape[dim]) {
      int a_idx = 0;
      for (int d = 0; d < a->ndim; d++) {
        a_idx += (idx[d] * a->strides[d]);
      }
      a->grad[a_idx] += self->grad[i];
    } else {
      int b_idx = 0;
      for (int d = 0; d < b->ndim; d++) {
        int mapped = idx[d];
        if (d == dim) {
          mapped -= a->shape[dim];
        }
        b_idx += (mapped * b->strides[d]);
      }
      b->grad[b_idx] += self->grad[i];
    }
  }
}

Tensor *concat_t(Memory *mem, Tensor *a, Tensor *b, int dim) {
  CHECK(a && b && dim >= 0 && a->ndim == b->ndim && dim < a->ndim,
        "concat_t: a is NULL, b is NULL, dim out of bounds, or ndim mismatch");

  int out_shape[a->ndim];

  for (int i = 0; i < a->ndim; i++) {
    if (i == dim) {
      out_shape[i] = a->shape[i] + b->shape[i];
    } else {
      CHECK(a->shape[i] == b->shape[i], "concat_t: not compatible for concat");
      out_shape[i] = a->shape[i];
    }
  }

  Tensor *r = tensor_zeros(mem, out_shape, a->ndim, TEMP);
  CHECK(r, "concat_t: failed to create result tensor");

  for (int i = 0; i < r->numel; i++) {
    int curr_idx = i;
    int idx[r->ndim];

    for (int d = r->ndim - 1; d >= 0; d--) {
      idx[d] = curr_idx % r->shape[d];
      curr_idx /= r->shape[d];
    }

    if (idx[dim] < a->shape[dim]) {
      int a_idx = 0;
      for (int d = 0; d < a->ndim; d++) {
        a_idx += (idx[d] * a->strides[d]);
      }
      r->data[i] = a->data[a_idx];
    } else {
      int b_idx = 0;
      for (int d = 0; d < b->ndim; d++) {
        int mapped = idx[d];
        if (d == dim) {
          mapped -= a->shape[dim];
        }
        b_idx += (mapped * b->strides[d]);
      }
      r->data[i] = b->data[b_idx];
    }
  }

  r->op = CONCAT;
  r->parents[0] = a;
  r->parents[1] = b;
  r->op_params[0] = dim;
  r->backward = backward_concat;

  return r;
}

static void backward_slice(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *r = self;
  int dim = r->op_params[0];
  int split_size = r->op_params[2];
  if (r->op_params[1] == 1) {
    for (int i = 0; i < r->numel; i++) {
      int idx[r->ndim];
      int curr = i;
      for (int d = r->ndim - 1; d >= 0; d--) {
        idx[d] = curr % r->shape[d];
        curr /= r->shape[d];
      }

      int a_idx = 0;
      for (int d = 0; d < r->ndim; d++) {
        a_idx += (idx[d] * a->strides[d]);
      }
      a->grad[a_idx] += self->grad[i];
    }
  } else {
    for (int i = 0; i < r->numel; i++) {
      int idx[r->ndim];
      int curr = i;
      for (int d = r->ndim - 1; d >= 0; d--) {
        idx[d] = curr % r->shape[d];
        curr /= r->shape[d];
      }

      int a_idx = 0;
      for (int d = 0; d < r->ndim; d++) {
        int mapped = idx[d];
        if (d == dim) {
          mapped += split_size;
        }
        a_idx += (mapped * a->strides[d]);
      }
      a->grad[a_idx] += self->grad[i];
    }
  }
}

Pair_T *slice_t(Memory *mem, Tensor *a, int dim, int split_size) {
  CHECK(a && dim >= 0 && dim < a->ndim && split_size < a->shape[dim] &&
            split_size > 0,
        "slice_t: a is NULL, dim out of bounds, or invalid split_size");
  Pair_T *r = allocate_mem(mem, sizeof(Pair_T), TEMP);
  CHECK(r, "slice_t: result allocation failed");

  int f_shape[a->ndim];
  int s_shape[a->ndim];

  for (int i = 0; i < a->ndim; i++) {
    if (i == dim) {
      f_shape[i] = split_size;
      s_shape[i] = a->shape[i] - split_size;
    } else {
      f_shape[i] = a->shape[i];
      s_shape[i] = a->shape[i];
    }
  }

  r->F = tensor_zeros(mem, f_shape, a->ndim, TEMP);
  CHECK(r->F, "slice_t: failed to create first slice tensor");
  r->S = tensor_zeros(mem, s_shape, a->ndim, TEMP);
  CHECK(r->S, "slice_t: failed to create second slice tensor");

  for (int i = 0; i < r->F->numel; i++) {
    int idx[r->F->ndim];
    int curr = i;
    for (int d = r->F->ndim - 1; d >= 0; d--) {
      idx[d] = curr % r->F->shape[d];
      curr /= r->F->shape[d];
    }

    int a_idx = 0;
    for (int d = 0; d < r->F->ndim; d++) {
      a_idx += (idx[d] * a->strides[d]);
    }
    r->F->data[i] = a->data[a_idx];
  }

  for (int i = 0; i < r->S->numel; i++) {
    int idx[r->S->ndim];
    int curr = i;
    for (int d = r->S->ndim - 1; d >= 0; d--) {
      idx[d] = curr % r->S->shape[d];
      curr /= r->S->shape[d];
    }

    int a_idx = 0;
    for (int d = 0; d < r->S->ndim; d++) {
      int mapped = idx[d];
      if (d == dim) {
        mapped += r->F->shape[dim];
      }
      a_idx += (mapped * a->strides[d]);
    }
    r->S->data[i] = a->data[a_idx];
  }

  r->F->op_params[0] = dim;
  r->F->op_params[1] =
      1; // 1 if it is the first tensor, F, 2, if it is S. Helpful for backprop
  r->F->op_params[2] = split_size;
  r->F->parents[0] = a;
  r->F->parents[1] = NULL;
  r->F->op = SLICE;
  r->F->backward = backward_slice;

  r->S->op_params[0] = dim;
  r->S->op_params[1] = 2;
  r->S->op_params[2] = split_size;
  r->S->parents[0] = a;
  r->S->parents[1] = NULL;
  r->S->op = SLICE;
  r->S->backward = backward_slice;

  return r;
}

static void backward_permute(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *r = self;

  int idx[a->ndim];
  memset(idx, 0, sizeof(idx));
  int offs[1] = {0};
  int N = r->numel;
  int *strides[1] = {r->strides};
  for (int i = 0; i < N; i++) {
    a->grad[offs[0]] += r->grad[offs[0]];
    advance_offsets(a->ndim, idx, r->shape, 1, offs, strides);
  }
}

Tensor *permute_t(Memory *mem, Tensor *a, int *dims, int total_dim) {
  CHECK(a && dims && total_dim == a->ndim,
        "permute_t: a is NULL, dims is NULL, or invalid dimensions");

  Tensor *r = (Tensor *)allocate_mem(mem, sizeof(Tensor), TEMP);
  CHECK(r, "permute_t: allocate_mem failed");
  r->data = a->data;
  r->grad = (float *)allocate_mem(mem, sizeof(float) * a->numel, TEMP);
  memset(r->grad, 0, sizeof(float) * a->numel);
  r->shape = (int *)allocate_mem(mem, sizeof(int) * a->ndim, TEMP);
  r->strides = (int *)allocate_mem(mem, sizeof(int) * a->ndim, TEMP);
  CHECK(r->grad && r->shape && r->strides,
        "transpose_t: grad, shape, stride allocation failed");

  for (int i = 0; i < total_dim; i++) {
    r->shape[i] = a->shape[dims[i]];
    r->strides[i] = a->strides[dims[i]];
  }

  r->ndim = a->ndim;
  r->numel = a->numel;
  r->visited_gen = a->visited_gen;
  r->contiguous = 0;
  r->dt = fp32;
  r->op = PERMUTE;
  r->parents[0] = a;
  r->parents[1] = NULL;
  r->backward = backward_permute;
  return r;
}

static void backward_softmax(Tensor *self) {
  Tensor *a = self->parents[0];
  Tensor *r = self;
  int dim = self->op_params[0];

  for (int i = 0; i < r->numel; i++) {
    int curr = i;
    int idx[a->ndim];
    for (int d = r->ndim - 1; d >= 0; d--) {
      idx[d] = curr % r->shape[d];
      curr /= r->shape[d];
    }

    int offset = 0;
    for (int d = 0; d < a->ndim; d++) {
      if (d != dim) {
        offset += (idx[d] * a->strides[d]);
      }
    }

    float sum = 0.0f;
    for (int k = 0; k < r->shape[dim]; k++) {
      int final_idx = offset + k * a->strides[dim];
      sum += r->grad[final_idx] * r->data[final_idx];
    }

    int k = idx[dim];
    int final_idx = offset + k * a->strides[dim];
    float data = r->data[final_idx];
    a->grad[final_idx] += data * (r->grad[final_idx] - sum);
  }
}

Tensor *softmax_t(Memory *mem, Tensor *a, int dim) {
  CHECK(a && dim >= 0 && dim < a->ndim,
        "softmax_t: a is NULL or dim out of bounds");

  Tensor *r = tensor_zeros(mem, a->shape, a->ndim, TEMP);
  CHECK(r, "softmax_t: failed to create result tensor");

  int idx[r->ndim];
  memset(idx, 0, sizeof(idx));
  for (int i = 0; i < r->numel; i++) {
    int curr = i;
    for (int d = r->ndim - 1; d >= 0; d--) {
      idx[d] = curr % r->shape[d];
      curr /= r->shape[d];
    }
    int offset = 0;
    for (int d = 0; d < r->ndim; d++) {
      if (d == dim)
        continue;
      offset += idx[d] * a->strides[d];
    }

    float m = -INFINITY;
    for (int k = 0; k < r->shape[dim]; k++) {
      float v = a->data[offset + k * a->strides[dim]];
      if (m < v)
        m = v;
    }

    float sum = 0.0f;
    for (int k = 0; k < r->shape[dim]; k++) {
      sum += expf(a->data[offset + k * a->strides[dim]] - m);
    }

    r->data[i] = expf(a->data[offset + idx[dim] * a->strides[dim]] - m) / sum;
  }

  r->op = SOFTMAX;
  r->op_params[0] = dim;
  r->parents[0] = a;
  r->parents[1] = NULL;
  r->backward = backward_softmax;

  return r;
}

void clip_gradients(ParameterList *pl, float threshold) {
  for (int i = 0; i < pl->count; i++) {
    Tensor *t = pl->t[i];
    for (int j = 0; j < t->numel; j++) {
      float g = t->grad[j];
      if (g > threshold) {
        t->grad[j] = threshold;
      } else if (g < -threshold) {
        t->grad[j] = -threshold;
      }
    }
  }
}

Adam *adam_init(Memory *mem, ParameterList *pl, float lr, float beta1,
                float beta2, float eps, int t) {
  Adam *optim = allocate_mem(mem, sizeof(Adam), PERM);
  CHECK(optim, "adam_init: allocate_mem failed");

  optim->lr = lr;
  optim->beta1 = beta1;
  optim->beta2 = beta2;
  optim->eps = eps;
  optim->t = t;

  optim->v = allocate_mem(mem, pl->count * sizeof(float *), PERM);
  optim->m = allocate_mem(mem, pl->count * sizeof(float *), PERM);
  CHECK(optim->v && optim->m, "adam_init: optim m or v allocation failed");
  for (int i = 0; i < pl->count; i++) {
    Tensor *t = pl->t[i];
    optim->v[i] = allocate_mem(mem, t->numel * sizeof(float), PERM);
    optim->m[i] = allocate_mem(mem, t->numel * sizeof(float), PERM);
    CHECK(optim->v[i] && optim->m[i],
          "adam_init: optim mi or vi allocation failed");
    memset(optim->v[i], 0, t->numel * sizeof(float));
    memset(optim->m[i], 0, t->numel * sizeof(float));
  }

  return optim;
}

void adam_step(Adam *optim, ParameterList *pl) {
  optim->t += 1;
  float lr_t = optim->lr * sqrtf(1.0f - powf(optim->beta2, optim->t)) /
               (1.0f - powf(optim->beta1, optim->t));
  for (int i = 0; i < pl->count; i++) {
    Tensor *t = pl->t[i];
    for (int j = 0; j < t->numel; j++) {
      float grad = t->grad[j];
      optim->m[i][j] =
          optim->beta1 * optim->m[i][j] + (1.0f - optim->beta1) * grad;
      optim->v[i][j] =
          optim->beta2 * optim->v[i][j] + (1.0f - optim->beta2) * grad * grad;
      float m_hat = optim->m[i][j] / (1.0f - powf(optim->beta1, optim->t));
      float v_hat = optim->v[i][j] / (1.0f - powf(optim->beta2, optim->t));
      t->data[j] -= lr_t * m_hat / (sqrtf(v_hat) + optim->eps);
    }
  }
}

AdamW *adamw_init(Memory *mem, ParameterList *pl, float lr, float beta1,
                  float beta2, float eps, float weight_decay, int t) {
  AdamW *optim = allocate_mem(mem, sizeof(AdamW), PERM);
  CHECK(optim, "adamw_init: allocate_mem failed");

  optim->lr = lr;
  optim->beta1 = beta1;
  optim->beta2 = beta2;
  optim->eps = eps;
  optim->weight_decay = weight_decay;
  optim->t = t;

  optim->v = allocate_mem(mem, pl->count * sizeof(float *), PERM);
  optim->m = allocate_mem(mem, pl->count * sizeof(float *), PERM);
  CHECK(optim->v && optim->m, "adamw_init: optim m or v allocation failed");
  for (int i = 0; i < pl->count; i++) {
    Tensor *t = pl->t[i];
    optim->v[i] = allocate_mem(mem, t->numel * sizeof(float), PERM);
    optim->m[i] = allocate_mem(mem, t->numel * sizeof(float), PERM);
    CHECK(optim->v[i] && optim->m[i],
          "adamw_init: optim mi or vi allocation failed");
    memset(optim->v[i], 0, t->numel * sizeof(float));
    memset(optim->m[i], 0, t->numel * sizeof(float));
  }

  return optim;
}

void adamw_step(AdamW *optim, ParameterList *pl) {
  optim->t += 1;
  float lr_t = optim->lr * sqrtf(1.0f - powf(optim->beta2, optim->t)) /
               (1.0f - powf(optim->beta1, optim->t));
  for (int i = 0; i < pl->count; i++) {
    Tensor *t = pl->t[i];
    for (int j = 0; j < t->numel; j++) {
      float grad = t->grad[j];
      optim->m[i][j] =
          optim->beta1 * optim->m[i][j] + (1.0f - optim->beta1) * grad;
      optim->v[i][j] =
          optim->beta2 * optim->v[i][j] + (1.0f - optim->beta2) * grad * grad;
      float m_hat = optim->m[i][j] / (1.0f - powf(optim->beta1, optim->t));
      float v_hat = optim->v[i][j] / (1.0f - powf(optim->beta2, optim->t));
      t->data[j] -= lr_t * (m_hat / (sqrtf(v_hat) + optim->eps) +
                            optim->weight_decay * t->data[j]);
    }
  }
}

void save_checkpoint(ParameterList *pl, const char *path) {
  CHECK_VOID(pl && path, "save_checkpoint: invalid params");

  FILE *f = fopen(path, "wb");
  CHECK_VOID(f, "save_checkpoint: unable to open file");

  uint32_t tensor_count = (uint32_t)pl->count;
  uint32_t magic_number = MAGIC_NUMBER;
  uint32_t version = VERSION;

  if (fwrite(&magic_number, sizeof(uint32_t), 1, f) != 1)
    goto fail;
  if (fwrite(&version, sizeof(uint32_t), 1, f) != 1)
    goto fail;
  if (fwrite(&tensor_count, sizeof(uint32_t), 1, f) != 1)
    goto fail;

  for (uint32_t i = 0; i < tensor_count; i++) {
    Tensor *t = pl->t[i];

    uint32_t ndim = (uint32_t)t->ndim;
    if (fwrite(&ndim, sizeof(uint32_t), 1, f) != 1)
      goto fail;

    for (uint32_t j = 0; j < ndim; j++) {
      uint32_t dim = (uint32_t)t->shape[j];
      if (fwrite(&dim, sizeof(uint32_t), 1, f) != 1)
        goto fail;
    }

    uint32_t numel = (uint32_t)t->numel;
    if (fwrite(&numel, sizeof(uint32_t), 1, f) != 1)
      goto fail;

    if (fwrite(t->data, sizeof(float), numel, f) != numel)
      goto fail;
  }

  fclose(f);
  return;

fail:
  fclose(f);
  CHECK_VOID(0, "save_checkpoint: write failed");
}

void load_checkpoint(ParameterList *pl, const char *path) {
  CHECK_VOID(pl && path, "load_checkpoint: invalid params");

  FILE *f = fopen(path, "rb");
  CHECK_VOID(f, "load_checkpoint: unable to open file");

  uint32_t magic_number;
  uint32_t version;
  uint32_t tensor_count;

  if (fread(&magic_number, sizeof(uint32_t), 1, f) != 1)
    goto fail;
  if (magic_number != MAGIC_NUMBER)
    goto fail;

  if (fread(&version, sizeof(uint32_t), 1, f) != 1)
    goto fail;
  if (version != VERSION)
    goto fail;

  if (fread(&tensor_count, sizeof(uint32_t), 1, f) != 1)
    goto fail;
  if (tensor_count != (uint32_t)pl->count)
    goto fail;

  for (uint32_t i = 0; i < tensor_count; i++) {
    Tensor *t = pl->t[i];

    uint32_t ndim;
    if (fread(&ndim, sizeof(uint32_t), 1, f) != 1)
      goto fail;
    if (ndim != (uint32_t)t->ndim)
      goto fail;

    for (uint32_t j = 0; j < ndim; j++) {
      uint32_t dim;
      if (fread(&dim, sizeof(uint32_t), 1, f) != 1)
        goto fail;
      if (dim != (uint32_t)t->shape[j])
        goto fail;
    }

    uint32_t numel;
    if (fread(&numel, sizeof(uint32_t), 1, f) != 1)
      goto fail;
    if (numel != (uint32_t)t->numel)
      goto fail;

    if (fread(t->data, sizeof(float), numel, f) != numel)
      goto fail;
  }

  fclose(f);
  return;

fail:
  fclose(f);
  CHECK_VOID(0, "load_checkpoint: read failed");
}

void mem_stats(Memory *mem) {
  CHECK_VOID(mem, "mem_stats: invalid params");
  float perm_perc = 1.0 * 100 * mem->perm->used / mem->perm->size;
  float temp_perc = 1.0 * 100 * mem->temp->used / mem->temp->size;
  printf("PERM Memory Used: %zu/%zu bytes -> %f%%\n", mem->perm->used,
         mem->perm->size, perm_perc);
  printf("TEMP Memory Used: %zu/%zu bytes -> %f%%\n", mem->temp->used,
         mem->temp->size, temp_perc);
}

void replace_t(Tensor *a, Tensor *b) {
  CHECK_VOID(check_op_compatibilty(a, b), "replace_t: not compatible");
  if (a->contiguous && b->contiguous) {
    for (int i = 0; i < a->numel; i++) {
      a->data[i] = b->data[i];
    }
  } else {
    int N = a->numel;
    int idx[a->ndim];
    memset(idx, 0, sizeof(idx));
    int offs[2] = {0, 0};
    int *strides[2] = {a->strides, b->strides};
    for (int i = 0; i < N; i++) {
      a->data[offs[0]] = b->data[offs[1]];
      advance_offsets(a->ndim, idx, b->shape, 2, offs, strides);
    }
  }
}

void detach_t(Tensor *a) {
  CHECK_VOID(a, "detach_t: invalid params");
  a->parents[0] = NULL;
  a->parents[1] = NULL;
}
