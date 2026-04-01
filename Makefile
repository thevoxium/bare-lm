CC = gcc-15
SANFLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer -g
SRCS = src/bare.c
SRCS_TEST = test/test.c
SRCS_TIMER = benchmark/benchmark.c
BUILD_DIR = build
OPENBLAS_PATH = /opt/homebrew/opt/openblas
CFLAGS = -Wall -I./src -O3 -march=native -ffast-math -fopenmp \
         -I$(OPENBLAS_PATH)/include
LDFLAGS = -L$(OPENBLAS_PATH)/lib -lopenblas

.PHONY: run benchmark test asan clean

run:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(FILE) $(LDFLAGS) -o $(BUILD_DIR)/$(basename $(notdir $(FILE)))
	./$(BUILD_DIR)/$(basename $(notdir $(FILE)))

test:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(SRCS_TEST) $(LDFLAGS) -o $(BUILD_DIR)/test
	./$(BUILD_DIR)/test

benchmark:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(SRCS_TIMER) $(LDFLAGS) -o $(BUILD_DIR)/timer
	./$(BUILD_DIR)/timer

asan:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SANFLAGS) $(SRCS) $(FILE) -o $(BUILD_DIR)/$(basename $(notdir $(FILE)))_san
	 ./$(BUILD_DIR)/$(basename $(notdir $(FILE)))_san $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)
