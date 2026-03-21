# bare-lm

A minimal autograd tensor library in C. Arena-based memory, zero dependencies beyond libc.

NOTE: examples & API will CHANGE a lot. This is under active developement.

## Build & Run

```bash
make run FILE=main.c        # build and run
make asan FILE=main.c       # run with address + undefined behavior sanitizer
```

## Example: XOR with a 2-layer MLP

```c
#include "src/bare.h"

void sgd_step(Tensor *t, float lr) {
  for (int i = 0; i < t->numel; i++) {
    t->data[i] -= (lr * t->grad[i]);
    t->grad[i] = 0.0f;
  }
}

int main() {
  Memory *mem = create_global_mem(1 << 28);

  int x_shape[] = {4, 2};
  int y_shape[] = {4, 1};
  int w1_shape[] = {2, 8};
  int b1_shape[] = {8};
  int w2_shape[] = {8, 1};
  int b2_shape[] = {1};

  Tensor *x = tensor_init(mem, x_shape, 2, PERM);
  Tensor *y = tensor_init(mem, y_shape, 2, PERM);

  float x_data[] = {0, 0, 0, 1, 1, 0, 1, 1};
  float y_data[] = {0, 1, 1, 0};
  for (int i = 0; i < 8; i++) x->data[i] = x_data[i];
  for (int i = 0; i < 4; i++) y->data[i] = y_data[i];

  Tensor *w1 = tensor_randn(mem, w1_shape, 2, PERM);
  Tensor *b1 = tensor_zeros(mem, b1_shape, 1, PERM);
  Tensor *w2 = tensor_randn(mem, w2_shape, 2, PERM);
  Tensor *b2 = tensor_zeros(mem, b2_shape, 1, PERM);

  float lr = 0.1f;

  for (int epoch = 0; epoch < 500; epoch++) {
    Tensor *h = matmul_t(mem, x, w1);
    Tensor *b1_b = broadcast_t(mem, b1, h->shape, h->ndim);
    h = add_t(mem, h, b1_b);
    h = relu_t(mem, h);

    Tensor *o = matmul_t(mem, h, w2);
    Tensor *b2_b = broadcast_t(mem, b2, o->shape, o->ndim);
    o = add_t(mem, o, b2_b);
    o = sigmoid_t(mem, o);

    Tensor *loss = mseloss_t(mem, o, y);
    backward(mem, loss);

    if (epoch % 100 == 0)
      printf("epoch %3d  loss=%.4f\n", epoch, loss->data[0]);

    sgd_step(w1, lr);
    sgd_step(b1, lr);
    sgd_step(w2, lr);
    sgd_step(b2, lr);

    reset_temp_mem(mem);
  }

  free_global_mem(mem);
}
```

### Walkthrough

**Step 1: Create memory.** A single `Memory` object holds two arenas (permanent and temporary). `1 << 28` = 256 MB per arena.

```c
Memory *mem = create_global_mem(1 << 28);
```

**Step 2: Allocate persistent tensors.** Inputs `x`, `y` and all weight/bias tensors live for the entire training run. Pass `PERM` so they survive `reset_temp_mem`.

```c
Tensor *x = tensor_init(mem, x_shape, 2, PERM);
Tensor *w1 = tensor_randn(mem, w1_shape, 2, PERM);
```

**Step 3: Forward pass.** Every operation (`matmul_t`, `add_t`, `relu_t`, `sigmoid_t`, `mseloss_t`) allocates its result from the temp arena. No `malloc` calls, no cleanup code.

```c
Tensor *h = matmul_t(mem, x, w1);
h = relu_t(mem, h);
Tensor *loss = mseloss_t(mem, o, y);
```

**Step 4: Backward pass.** `backward` builds a topological sort of the computation graph (also temp-allocated) and propagates gradients. Intermediate tensor data is still valid at this point.

```c
backward(mem, loss);
```

**Step 5: Update weights and reset.** After gradients are consumed, `reset_temp_mem` zeroes the temp arena in O(1). All intermediate tensors from the forward pass are gone. The next epoch starts with a clean temp arena.

```c
sgd_step(w1, lr);
reset_temp_mem(mem);
```

**Step 6: Cleanup.** `free_global_mem` releases both arenas and the `Memory` struct.

```c
free_global_mem(mem);
```

## Memory Model

All allocations go through `allocate_mem(mem, size, perm)`. There are no individual `free` calls.

| Flag | Lifetime | Used for |
|------|----------|----------|
| `PERM` | Until `free_global_mem` | Weights, biases, persistent inputs/targets |
| `TEMP` | Until `reset_temp_mem` | Operation results, graph traversal arrays, loss |

`reset_temp_mem` resets the temp arena pointer to 0. It is called **after** both forward and backward complete, because backward reads intermediate tensor data.

```
epoch N:  forward (temp fills) → backward (reads temp) → update → reset_temp_mem
epoch N+1: forward (temp fills from 0) → ...
```

## API Reference

### Memory

| Function | Signature | Description |
|----------|-----------|-------------|
| `create_global_mem` | `(size_t size) → Memory*` | Allocate perm + temp arenas |
| `reset_temp_mem` | `(Memory *mem)` | Reset temp arena to empty |
| `allocate_mem` | `(Memory *mem, size_t size, uint8_t perm) → void*` | Arena allocation |
| `free_global_mem` | `(Memory *mem)` | Free both arenas and Memory |

### Tensor Creation

| Function | Signature | Description |
|----------|-----------|-------------|
| `tensor_init` | `(Memory*, int *shape, int ndim, uint8_t perm) → Tensor*` | Zero-initialized tensor |
| `tensor_zeros` | `(Memory*, int *shape, int ndim, uint8_t perm) → Tensor*` | Same as tensor_init |
| `tensor_ones` | `(Memory*, int *shape, int ndim, uint8_t perm) → Tensor*` | All ones |
| `tensor_randn` | `(Memory*, int *shape, int ndim, uint8_t perm) → Tensor*` | Random normal (Box-Muller) |
| `tensor_get` | `(Tensor*, int *indices) → float` | Get element at multi-dim index |
| `print_t` | `(Tensor*, uint8_t grad)` | Print tensor (grad=1 to include gradients) |

### Arithmetic

| Function | Signature | Description |
|----------|-----------|-------------|
| `add_t` | `(Memory*, Tensor *a, Tensor *b) → Tensor*` | Element-wise a + b |
| `sub_t` | `(Memory*, Tensor *a, Tensor *b) → Tensor*` | Element-wise a - b |
| `mul_t` | `(Memory*, Tensor *a, Tensor *b) → Tensor*` | Element-wise a * b |
| `divi_t` | `(Memory*, Tensor *a, Tensor *b) → Tensor*` | Element-wise a / b |
| `neg_t` | `(Memory*, Tensor *a) → Tensor*` | Element-wise -a |
| `pow_t` | `(Memory*, Tensor *a, float exp) → Tensor*` | Element-wise pow(a, exp) |

### Math

| Function | Signature | Description |
|----------|-----------|-------------|
| `exp_t` | `(Memory*, Tensor *a) → Tensor*` | Element-wise exp |
| `log_t` | `(Memory*, Tensor *a) → Tensor*` | Element-wise log |

### Reductions

| Function | Signature | Description |
|----------|-----------|-------------|
| `sum_t` | `(Memory*, Tensor *a, int dim) → Tensor*` | Sum along dimension |
| `mean_t` | `(Memory*, Tensor *a, int dim) → Tensor*` | Mean along dimension |
| `max_t` | `(Memory*, Tensor *a, int dim) → Tensor*` | Max along dimension |
| `dot_t` | `(Memory*, Tensor *a, Tensor *b) → Tensor*` | Dot product (1D) |

### Activations

| Function | Signature | Description |
|----------|-----------|-------------|
| `relu_t` | `(Memory*, Tensor *a) → Tensor*` | ReLU |
| `gelu_t` | `(Memory*, Tensor *a) → Tensor*` | GELU (tanh approximation) |
| `sigmoid_t` | `(Memory*, Tensor *a) → Tensor*` | Sigmoid |
| `tanh_t` | `(Memory*, Tensor *a) → Tensor*` | Tanh |

### Linear Algebra

| Function | Signature | Description |
|----------|-----------|-------------|
| `matmul_t` | `(Memory*, Tensor *a, Tensor *b) → Tensor*` | Matrix multiply (2D) |
| `transpose_t` | `(Memory*, Tensor *a) → Tensor*` | Transpose 2D |

### Shape Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `reshape_t` | `(Memory*, Tensor *a, int *shape, int ndim) → Tensor*` | Reshape (shares data) |
| `squeeze_t` | `(Memory*, Tensor *a, int dim) → Tensor*` | Remove dim of size 1 |
| `unsqueeze_t` | `(Memory*, Tensor *a, int dim) → Tensor*` | Insert dim of size 1 |
| `broadcast_t` | `(Memory*, Tensor *a, int *shape, int tar_dim) → Tensor*` | Broadcast to target shape |

### Loss Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `mseloss_t` | `(Memory*, Tensor *a, Tensor *b) → Tensor*` | Mean squared error |
| `crossentropyloss_t` | `(Memory*, Tensor *logits, Tensor *targets) → Tensor*` | Cross-entropy (logits: [N,C], targets: [N]) |

### Layers

| Function | Signature | Description |
|----------|-----------|-------------|
| `create_linear` | `(Memory*, int d_in, int d_out) → Linear*` | Linear layer (weights + bias, PERM) |
| `linear_t` | `(Memory*, Linear*, Tensor *x) → Tensor*` | Forward: x @ W^T + b |

### Autograd

| Function | Signature | Description |
|----------|-----------|-------------|
| `backward` | `(Memory*, Tensor *root)` | Backpropagate from root |

All operations register backward passes and track parent tensors automatically.
