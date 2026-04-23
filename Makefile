CC ?= cc
BUILD_DIR ?= build

SRCS = src/bare.c
SRCS_TEST ?= examples/test.c

# Prefer FILE, keep lowercase `file` for backward compatibility.
FILE ?= $(file)

BASE_CFLAGS = -Wall -I./src -O2
PERF_CFLAGS = -O3 -march=native -ffast-math
DEBUG_CFLAGS = -O0 -g -fno-omit-frame-pointer
SANFLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer

# OpenBLAS flags:
# 1) pkg-config (recommended), 2) OPENBLAS_PATH fallback, 3) linker default path.
OPENBLAS_CFLAGS ?= $(shell pkg-config --cflags openblas 2>/dev/null)
OPENBLAS_LIBS ?= $(shell pkg-config --libs openblas 2>/dev/null)
OPENBLAS_PATH ?= $(shell brew --prefix openblas 2>/dev/null)

ifeq ($(strip $(OPENBLAS_CFLAGS)$(OPENBLAS_LIBS)),)
ifneq ($(strip $(OPENBLAS_PATH)),)
OPENBLAS_CFLAGS := -I$(OPENBLAS_PATH)/include
OPENBLAS_LIBS := -L$(OPENBLAS_PATH)/lib -lopenblas
else
OPENBLAS_LIBS := -lopenblas
endif
endif

CFLAGS = $(BASE_CFLAGS) $(OPENBLAS_CFLAGS)

ifeq ($(perf),1)
CFLAGS += $(PERF_CFLAGS)
endif

LDFLAGS = $(OPENBLAS_LIBS) -lm

BIN = $(BUILD_DIR)/$(basename $(notdir $(FILE)))

.PHONY: help run time test valgrind asan clean

help:
	@printf "Targets:\n"
	@printf "  make run FILE=examples/xor.c [CC=cc] [perf=1]\n"
	@printf "  make asan FILE=examples/xor.c [CC=cc]\n"
	@printf "  make time FILE=examples/xor.c [CC=cc]\n"
	@printf "  make valgrind FILE=examples/xor.c [CC=cc]\n"
	@printf "  make test [CC=cc]\n"
	@printf "  make clean\n\n"
	@printf "Variables:\n"
	@printf "  FILE=<path>            C file to compile with src/bare.c\n"
	@printf "  CC=<compiler>          e.g. gcc, gcc-15, clang\n"
	@printf "  perf=1                 enable -O3 -march=native -ffast-math\n"
	@printf "  OPENBLAS_PATH=<path>   fallback if pkg-config cannot find OpenBLAS\n"

run:
	@if [ -z "$(FILE)" ]; then \
		echo "error: FILE is required (example: make run FILE=examples/xor.c)"; \
		exit 1; \
	fi
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(FILE) $(LDFLAGS) -o $(BIN)
	./$(BIN)

test:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(SRCS_TEST) $(LDFLAGS) -o $(BUILD_DIR)/test
	./$(BUILD_DIR)/test

time:
	@if [ -z "$(FILE)" ]; then \
		echo "error: FILE is required (example: make time FILE=examples/xor.c)"; \
		exit 1; \
	fi
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(FILE) $(LDFLAGS) -o $(BIN)
	/usr/bin/time -p ./$(BIN)

valgrind:
	@if [ -z "$(FILE)" ]; then \
		echo "error: FILE is required (example: make valgrind FILE=examples/xor.c)"; \
		exit 1; \
	fi
	@if ! command -v valgrind >/dev/null 2>&1; then \
		echo "error: valgrind is not installed"; \
		exit 1; \
	fi
	@mkdir -p $(BUILD_DIR)
	$(CC) $(BASE_CFLAGS) $(DEBUG_CFLAGS) $(OPENBLAS_CFLAGS) $(SRCS) $(FILE) $(LDFLAGS) -o $(BIN)
	valgrind --leak-check=full --show-leak-kinds=all ./$(BIN)

asan:
	@if [ -z "$(FILE)" ]; then \
		echo "error: FILE is required (example: make asan FILE=examples/xor.c)"; \
		exit 1; \
	fi
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SANFLAGS) $(SRCS) $(FILE) $(LDFLAGS) -o $(BIN)_san
	./$(BIN)_san

clean:
	rm -rf $(BUILD_DIR)
