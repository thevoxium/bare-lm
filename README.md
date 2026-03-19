# bare-lm

A minimal autograd tensor library in C for building language models.

## Build & Run

```bash
make run FILE=<filename>
```

## API Reference

### Context Management

| Function | Params | Returns | Description |
|----------|--------|---------|-------------|
| `create_global_ctx` | — | `GraphContext*` | Create computation graph context |
| `ctx_free` | `ctx` | `void` | Free context and all tracked tensors |
| `ctx_zero_grad` | `ctx` | `void` | Zero all gradients in context |

### Tensor Creation

| Function | Params | Returns | Description |
|----------|--------|---------|-------------|
| `tensor_init` | `shape`, `ndim`, `ctx` | `Tensor*` | Create zero-initialized tensor |
| `tensor_zeros` | `shape`, `ndim`, `ctx` | `Tensor*` | Create zero tensor |
| `tensor_ones` | `shape`, `ndim`, `ctx` | `Tensor*` | Create ones tensor |
| `tensor_randn` | `shape`, `ndim`, `ctx` | `Tensor*` | Create random normal tensor |
| `tensor_free` | `t` | `void` | Free tensor memory |
| `tensor_get` | `t`, `indices` | `float` | Get element at multi-dim index |
| `print_t` | `t`, `grad` | `void` | Print tensor; `grad=1` to include gradients |

### Arithmetic Ops

| Function | Params | Returns | Description |
|----------|--------|---------|-------------|
| `add_t` | `a`, `b`, `ctx` | `Tensor*` | Element-wise addition |
| `sub_t` | `a`, `b`, `ctx` | `Tensor*` | Element-wise subtraction |
| `mul_t` | `a`, `b`, `ctx` | `Tensor*` | Element-wise multiplication |
| `divi_t` | `a`, `b`, `ctx` | `Tensor*` | Element-wise division |
| `neg_t` | `a`, `ctx` | `Tensor*` | Element-wise negation |
| `pow_t` | `a`, `exponent`, `ctx` | `Tensor*` | Element-wise power |

### Math Functions

| Function | Params | Returns | Description |
|----------|--------|---------|-------------|
| `exp_t` | `a`, `ctx` | `Tensor*` | Element-wise exponential |
| `log_t` | `a`, `ctx` | `Tensor*` | Element-wise natural log |

### Reductions

| Function | Params | Returns | Description |
|----------|--------|---------|-------------|
| `sum_t` | `a`, `dim`, `ctx` | `Tensor*` | Sum along dimension |
| `mean_t` | `a`, `dim`, `ctx` | `Tensor*` | Mean along dimension |
| `max_t` | `a`, `dim`, `ctx` | `Tensor*` | Max along dimension |

### Activations

| Function | Params | Returns | Description |
|----------|--------|---------|-------------|
| `relu_t` | `a`, `ctx` | `Tensor*` | ReLU activation |
| `gelu_t` | `a`, `ctx` | `Tensor*` | GELU activation (tanh approx) |
| `sigmoid_t` | `a`, `ctx` | `Tensor*` | Sigmoid activation |
| `tanh_t` | `a`, `ctx` | `Tensor*` | Tanh activation |

### Linear Algebra

| Function | Params | Returns | Description |
|----------|--------|---------|-------------|
| `dot_t` | `a`, `b`, `ctx` | `Tensor*` | Dot product of 1D tensors |
| `matmul_t` | `a`, `b`, `ctx` | `Tensor*` | Matrix multiplication (2D) |
| `transpose_t` | `a`, `ctx` | `Tensor*` | Transpose 2D tensor |

### Shape Operations

| Function | Params | Returns | Description |
|----------|--------|---------|-------------|
| `reshape_t` | `a`, `shape`, `ndim`, `ctx` | `Tensor*` | Reshape tensor |
| `squeeze_t` | `a`, `dim`, `ctx` | `Tensor*` | Remove dim of size 1 |
| `unsqueeze_t` | `a`, `dim`, `ctx` | `Tensor*` | Insert dim of size 1 |
| `broadcast_t` | `a`, `shape`, `tar_dim`, `ctx` | `Tensor*` | Broadcast to target shape |

### Loss Functions

| Function | Params | Returns | Description |
|----------|--------|---------|-------------|
| `mseloss_t` | `a`, `b`, `ctx` | `Tensor*` | Mean squared error loss |
| `crossentropyloss_t` | `logits`, `targets`, `ctx` | `Tensor*` | Cross-entropy loss (expects `logits`: [N,C], `targets`: [N]) |

### Autograd

| Function | Params | Returns | Description |
|----------|--------|---------|-------------|
| `backward` | `root` | `void` | Run backpropagation from root tensor |

All operations automatically register backward passes and track parents for autograd. Pass `ctx` (non-NULL) to automatically manage tensor lifetime.
