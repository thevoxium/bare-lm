CC = gcc
CFLAGS = -Wall -I./src -O3 -march=native -ffast-math
SANFLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer -g
SRCS = src/bare.c
SRCS_TEST = test/test.c
SRCS_TIMER = benchmark/timer.c
BUILD_DIR = build

.PHONY: run time test asan clean

run:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(FILE) -o $(BUILD_DIR)/$(basename $(notdir $(FILE)))
	./$(BUILD_DIR)/$(basename $(notdir $(FILE)))

test:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(SRCS_TEST) -o $(BUILD_DIR)/test
	./$(BUILD_DIR)/test

time:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(SRCS_TIMER) -o $(BUILD_DIR)/timer
	./$(BUILD_DIR)/timer

asan:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SANFLAGS) $(SRCS) $(FILE) -o $(BUILD_DIR)/$(basename $(notdir $(FILE)))_san
	 ./$(BUILD_DIR)/$(basename $(notdir $(FILE)))_san

clean:
	rm -rf $(BUILD_DIR)
