CC ?= cc
PREFIX ?= /usr/local

SRC = src/bare.c
BUILD_DIR = build

UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S),Darwin)
    SHARED_LIB = libbare.dylib
    SHARED_LDFLAGS = -dynamiclib -Wl,-install_name,$(PREFIX)/lib/$(SHARED_LIB)
else
    SHARED_LIB = libbare.so
    SHARED_LDFLAGS = -shared -Wl,-soname,$(SHARED_LIB)
endif

OUT = $(BUILD_DIR)/$(SHARED_LIB)

OPENBLAS_PREFIX ?=
CFLAGS = -Wall -O2 -fPIC
LDFLAGS = -lm -lopenblas
ifneq ($(OPENBLAS_PREFIX),)
    CFLAGS += -I$(OPENBLAS_PREFIX)/include
    LDFLAGS += -L$(OPENBLAS_PREFIX)/lib
endif

.PHONY: shared install uninstall clean

shared:
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRC) $(SHARED_LDFLAGS) $(LDFLAGS) -o $(OUT)

install: 
	mkdir -p $(PREFIX)/include
	mkdir -p $(PREFIX)/lib
	cp src/bare.h $(PREFIX)/include/
	cp $(OUT) $(PREFIX)/lib/
uninstall:
	rm -f $(PREFIX)/include/bare.h
	rm -f $(PREFIX)/lib/$(SHARED_LIB)

clean:
	rm -rf $(BUILD_DIR)
