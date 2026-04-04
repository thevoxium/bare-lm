CC ?= gcc-15
SANFLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer 
SRCS = src/bare.c
SRCS_TEST = test/test.c
BUILD_DIR = build
OPENBLAS_PATH = /opt/homebrew/opt/openblas
CFLAGS = -Wall -I./src -O3 -march=native -ffast-math -fopenmp \
         -I$(OPENBLAS_PATH)/include
VALGRIND = -Wall -I./src -O0 -march=native -ffast-math -fopenmp \
         -I$(OPENBLAS_PATH)/include
ifeq ($(perf),1)
CFLAGS += -g -fno-omit-frame-pointer 
endif
LDFLAGS = -L$(OPENBLAS_PATH)/lib -lopenblas -lm

.PHONY: run time test asan clean

run:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(file) $(LDFLAGS) -o $(BUILD_DIR)/$(basename $(notdir $(file)))
	./$(BUILD_DIR)/$(basename $(notdir $(file)))

test:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(SRCS_TEST) $(LDFLAGS) -o $(BUILD_DIR)/test
	./$(BUILD_DIR)/test

time:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(file) $(LDFLAGS) -o $(BUILD_DIR)/$(basename $(notdir $(file)))
	./$(BUILD_DIR)/$(basename $(notdir $(file)))

valgrind:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(VALGRIND) $(SRCS) $(file) $(LDFLAGS) -o $(BUILD_DIR)/$(basename $(notdir $(file)))
	./$(BUILD_DIR)/$(basename $(notdir $(file)))

asan:
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SANFLAGS) $(SRCS) $(file) $(LDFLAGS) -o $(BUILD_DIR)/$(basename $(notdir $(file)))_san
	 ./$(BUILD_DIR)/$(basename $(notdir $(file)))_san 

clean:
	rm -rf $(BUILD_DIR)
