#include "../src/bare.h"

int main() {
  Memory *mem = create_global_mem(1 << 28);
  ParameterList *pl = create_param_list(mem);

  int x_shape[] = {4, 2};
  int y_shape[] = {4, 1};

  Tensor *x = tensor_init(mem, x_shape, 2, PERM);
  Tensor *y = tensor_init(mem, y_shape, 2, PERM);

  float x_data[] = {0, 0, 0, 1, 1, 0, 1, 1};
  float y_data[] = {0, 1, 1, 0};
  for (int i = 0; i < 8; i++)
    x->data[i] = x_data[i];
  for (int i = 0; i < 4; i++)
    y->data[i] = y_data[i];

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
