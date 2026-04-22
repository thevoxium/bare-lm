#ifndef BPE_H
#define BPE_H

#include "../src/bare.h"

#define MAX_WORDS 1000

typedef struct WordFreqPair {
  char *word;
  int freq;
} WordFreqPair;

typedef struct WordFreqMapping {
  WordFreqPair word_freq_pairs[MAX_WORDS];
} WordFreqMapping;

int count_words(char *input);

void BPE(char *input);

#endif // !BPE_H
