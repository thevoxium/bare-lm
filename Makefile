CC = gcc
CFLAGS = -Wall -I./src
SANFLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer -g
SRCS = src/bare.c
BUILD_DIR = build

.PHONY: run run_san clean

run:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(FILE) -o $(BUILD_DIR)/$(basename $(notdir $(FILE)))
	./$(BUILD_DIR)/$(basename $(notdir $(FILE)))

asan:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SANFLAGS) $(SRCS) $(FILE) -o $(BUILD_DIR)/$(basename $(notdir $(FILE)))_san
	 ./$(BUILD_DIR)/$(basename $(notdir $(FILE)))_san

clean:
	rm -rf $(BUILD_DIR)
