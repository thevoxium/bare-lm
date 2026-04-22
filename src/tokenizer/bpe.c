#include "bpe.h"

int count_words(char *input) {
  int i = 0;
  int count = 0;
  int in_word = 0;
  while (input[i] != '\0') {
    if (input[i] != ' ') {
      if (in_word == 0) {
        in_word = 1;
        count++;
      }
    } else {
      in_word = 0;
    }
    i++;
  }
  return count;
}

void BPE(char *input) {
  WordFreqMapping word_freq_mapping;
  int word_count = count_words(input);

  printf("%d", word_count);
  return;
}
