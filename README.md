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

int main() {
  Memory *mem = create_global_mem(1 << 28);
  ParameterList *pl = create_param_list(mem);

  int x_shape[] = {4, 2};
  int y_shape[] = {4, 1};

  Tensor *x = tensor_init(mem, x_shape, 2, PERM);
  Tensor *y = tensor_init(mem, y_shape, 2, PERM);

  float x_data[] = {0, 0, 0, 1, 1, 0, 1, 1};
  float y_data[] = {0, 1, 1, 0};
  for (int i = 0; i < 8; i++) x->data[i] = x_data[i];
  for (int i = 0; i < 4; i++) y->data[i] = y_data[i];

  Linear *l1 = create_linear(mem, pl, 2, 8);
  Linear *l2 = create_linear(mem, pl, 8, 1);

  for (int epoch = 0; epoch < 500; epoch++) {
    zero_grad(pl);

    Tensor *h = linear_t(mem, l1, x);
    h = relu_t(mem, h);
    Tensor *o = linear_t(mem, l2, h);
    o = sigmoid_t(mem, o);

    Tensor *loss = mseloss_t(mem, o, y);
    backward(mem, loss);

    if (epoch % 100 == 0)
      printf("epoch %3d  loss=%.4f\n", epoch, loss->data[0]);

    sgd_step(pl, 0.1f);
    reset_temp_mem(mem);
  }

  free_global_mem(mem);
}
```

### Walkthrough

**Step 1: Create memory and a parameter list.** A single `Memory` object holds two arenas (permanent and temporary). `1 << 28` = 256 MB per arena. A `ParameterList` tracks all trainable tensors.

```c
Memory *mem = create_global_mem(1 << 28);
ParameterList *pl = create_param_list(mem);
```

**Step 2: Allocate persistent tensors.** Inputs `x` and `y` are plain PERM tensors (not trainable). Pass `PERM` so they survive `reset_temp_mem`.

```c
Tensor *x = tensor_init(mem, x_shape, 2, PERM);
Tensor *y = tensor_init(mem, y_shape, 2, PERM);
```

**Step 3: Create layers with auto-registration.** `create_linear` allocates weights and bias as PERM tensors and automatically adds them to the parameter list.

```c
Linear *l1 = create_linear(mem, pl, 2, 8);
Linear *l2 = create_linear(mem, pl, 8, 1);
// pl now contains l1->weights, l1->bias, l2->weights, l2->bias
```

**Step 4: Forward pass.** Every operation (`linear_t`, `relu_t`, `sigmoid_t`, `mseloss_t`) allocates its result from the temp arena. No `malloc` calls, no cleanup code.

```c
Tensor *h = linear_t(mem, l1, x);
h = relu_t(mem, h);
Tensor *o = linear_t(mem, l2, h);
o = sigmoid_t(mem, o);
Tensor *loss = mseloss_t(mem, o, y);
```

**Step 5: Backward pass.** `backward` builds a topological sort of the computation graph (also temp-allocated) and propagates gradients. Intermediate tensor data is still valid at this point.

```c
backward(mem, loss);
```

**Step 6: Zero grads, update, reset.** `zero_grad` clears all parameter gradients. `sgd_step` applies SGD to all parameters in one call. After gradients are consumed, `reset_temp_mem` zeroes the temp arena in O(1). All intermediate tensors from the forward pass are gone.

```c
zero_grad(pl);
sgd_step(pl, 0.1f);
reset_temp_mem(mem);
```

**Step 7: Cleanup.** `free_global_mem` releases both arenas and the `Memory` struct.

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

## ParameterList

`ParameterList` is a dynamic array of trainable tensors. It is a `typedef` of `Dt_array` allocated in the PERM arena.

```c
ParameterList *pl = create_param_list(mem);

// Manual registration
param_list_add(mem, pl, my_tensor);

// Or automatic — create_linear registers weights and bias for you
Linear *l = create_linear(mem, pl, 128, 64);
```

Once built, the parameter list drives the training loop:

```c
zero_grad(pl);       // zero all parameter gradients
backward(mem, loss); // backprop
sgd_step(pl, 0.01f); // update all parameters
```

## API Reference

### Memory

| Function | Signature | Description |
|----------|-----------|-------------|
| `create_global_mem` | `(size_t size) → Memory*` | Allocate perm + temp arenas |
| `reset_temp_mem` | `(Memory *mem)` | Reset temp arena to empty |
| `allocate_mem` | `(Memory *mem, size_t size, uint8_t perm) → void*` | Arena allocation |
| `free_global_mem` | `(Memory *mem)` | Free both arenas and Memory |

### ParameterList

| Function | Signature | Description |
|----------|-----------|-------------|
| `create_param_list` | `(Memory *mem) → ParameterList*` | Create an empty PERM parameter list |
| `param_list_add` | `(Memory *mem, ParameterList *pl, Tensor *t)` | Add a tensor to the list |
| `zero_grad` | `(ParameterList *pl)` | Zero gradients for all parameters |
| `sgd_step` | `(ParameterList *pl, float lr)` | SGD update: `data -= lr * grad` |

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
| `create_linear` | `(Memory*, ParameterList *pl, int d_in, int d_out) → Linear*` | Linear layer, auto-registers weights + bias |
| `linear_t` | `(Memory*, Linear*, Tensor *x) → Tensor*` | Forward: x @ W^T + b |

### Autograd

| Function | Signature | Description |
|----------|-----------|-------------|
| `backward` | `(Memory*, Tensor *root)` | Backpropagate from root |

All operations register backward passes and track parent tensors automatically.
