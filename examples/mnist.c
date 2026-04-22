// Download the dataset from
// https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?resource=download&select=mnist_train.csv
//
#include "../src/bare.h"
#include <time.h>

#define TRAIN_ROWS 60000
#define TEST_ROWS 10000
#define TRAIN_COLS 784
#define LABELS 10

int parse_int(char **ptr) {
  int val = 0;
  while (**ptr >= '0' && **ptr <= '9') {
    val = val * 10 + (**ptr - '0');
    (*ptr)++;
  }
  if (**ptr == ',')
    (*ptr)++;
  return val;
}

void parse_csv(FILE *fp, Tensor *input, Tensor *output) {
  char line[16000];
  int k = 0;
  while (fgets(line, sizeof(line), fp)) {
    char *ptr = line;
    int label = parse_int(&ptr);
    output->data[k * LABELS + label] = 1.0f;

    for (int i = 0; i < TRAIN_COLS; i++) {
      int pixel = parse_int(&ptr);
      input->data[k * TRAIN_COLS + i] = 1.0f * pixel / 255;
    }
    k++;
  }
}

Pair_T get_batch(Memory *mem, int batch_size, Tensor *input, Tensor *output) {
  Pair_T result;

  result.F = tensor_init(mem, S(batch_size, TRAIN_COLS), 2, TEMP);
  result.S = tensor_init(mem, S(batch_size, LABELS), 2, TEMP);

  for (int i = 0; i < batch_size; i++) {
    int num = rand() % TRAIN_ROWS;
    for (int j = 0; j < TRAIN_COLS; j++) {
      result.F->data[i * TRAIN_COLS + j] = input->data[num * TRAIN_COLS + j];
    }
    for (int j = 0; j < LABELS; j++) {
      result.S->data[i * LABELS + j] = output->data[num * LABELS + j];
    }
  }

  return result;
}

int get_max_index(float *data, int size) {
  int max_idx = 0;
  for (int i = 1; i < size; i++) {
    if (data[i] > data[max_idx])
      max_idx = i;
  }
  return max_idx;
}

int main() {
  srand(time(NULL));
  Memory *mem = create_global_mem(1024 * 1024 * 1024);
  ParameterList *pl = create_param_list(mem);

  char *train_data_path = "data/mnist_train.csv";

  FILE *fp = fopen(train_data_path, "r");
  if (!fp) {
    return 1;
  }

  Tensor *input = tensor_init(mem, S(TRAIN_ROWS, TRAIN_COLS), 2, PERM);
  Tensor *output = tensor_init(mem, S(TRAIN_ROWS, LABELS), 2, PERM);

  parse_csv(fp, input, output);
  fclose(fp);

  char *test_data_path = "data/mnist_test.csv";
  FILE *fp_test = fopen(test_data_path, "r");
  if (!fp_test) {
    return 1;
  }

  Tensor *test_input = tensor_init(mem, S(TEST_ROWS, TRAIN_COLS), 2, PERM);
  Tensor *test_output = tensor_init(mem, S(TEST_ROWS, LABELS), 2, PERM);

  parse_csv(fp_test, test_input, test_output);
  fclose(fp_test);

  int hidden1 = 512;
  int hidden2 = 256;
  int hidden3 = 128;
  int steps = 10000;
  int batch_size = 128;
  Linear *l1 = create_linear(mem, pl, TRAIN_COLS, hidden1);
  Linear *l2 = create_linear(mem, pl, hidden1, hidden2);
  Linear *l3 = create_linear(mem, pl, hidden2, hidden3);
  Linear *l4 = create_linear(mem, pl, hidden3, LABELS);
  AdamW *optim = adamw_init(mem, pl, 0.00003, 0.9, 0.99, 1e-8, 1e-2, 0);

  for (int i = 0; i < steps; i++) {
    reset_temp_mem(mem);
    zero_grad(pl);

    Pair_T batch = get_batch(mem, batch_size, input, output);
    Tensor *x = batch.F;
    Tensor *y = batch.S;

    Tensor *h = linear_t(mem, l1, x);
    h = relu_t(mem, h);
    h = linear_t(mem, l2, h);
    h = relu_t(mem, h);
    h = linear_t(mem, l3, h);
    h = relu_t(mem, h);
    Tensor *logits = linear_t(mem, l4, h);

    int label_shape[] = {batch_size};
    Tensor *y_idx = tensor_init(mem, label_shape, 1, TEMP);
    for (int j = 0; j < batch_size; j++) {
      y_idx->data[j] = (float)get_max_index(&y->data[j * LABELS], LABELS);
    }

    Tensor *out = crossentropyloss_t(mem, logits, y_idx);
    backward(mem, out);
    adamw_step(optim, pl);

    float loss = item(out);
    if (i % 100 == 0) {
      printf("STEP: %d, LOSS: %f\n", i, loss);
    }
  }

  int correct = 0;
  for (int i = 0; i < TEST_ROWS; i++) {
    reset_temp_mem(mem);
    Tensor *x = tensor_init(mem, S(1, TRAIN_COLS), 2, TEMP);
    Tensor *y_true = tensor_init(mem, S(1, LABELS), 2, TEMP);

    for (int j = 0; j < TRAIN_COLS; j++) {
      x->data[j] = test_input->data[i * TRAIN_COLS + j];
    }
    for (int j = 0; j < LABELS; j++) {
      y_true->data[j] = test_output->data[i * LABELS + j];
    }

    Tensor *h = linear_t(mem, l1, x);
    h = relu_t(mem, h);
    h = linear_t(mem, l2, h);
    h = relu_t(mem, h);
    h = linear_t(mem, l3, h);
    h = relu_t(mem, h);
    h = linear_t(mem, l4, h);
    Tensor *probs = softmax_t(mem, h, 1);

    int pred = get_max_index(probs->data, LABELS);
    int true_label = get_max_index(y_true->data, LABELS);
    if (pred == true_label)
      correct++;
  }

  float accuracy = 100.0f * correct / TEST_ROWS;
  printf("TEST ACCURACY: %f%% (%d/%d)\n", accuracy, correct, TEST_ROWS);

  free_global_mem(mem);
  return 0;
}
