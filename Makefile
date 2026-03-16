CC = gcc
CFLAGS = -Wall -I./src
SRCS = src/bare.c
BUILD_DIR = build

.PHONY: run clean

run:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(FILE) -o $(BUILD_DIR)/$(basename $(notdir $(FILE)))
	./$(BUILD_DIR)/$(basename $(notdir $(FILE)))

clean:
	rm -rf $(BUILD_DIR)
