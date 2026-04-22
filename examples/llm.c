#include "../src/bare.h"
#include "../src/tokenizer/bpe.h"

int main() {
  Memory *mem = create_global_mem(1 << 20);

  char *input = "hello world";
  BPE(input);

  reset_temp_mem(mem);
  free_global_mem(mem);
  return 0;
}
