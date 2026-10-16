UNAME := $(shell uname -s)
ARCH := $(shell uname -m)

ifeq ($(UNAME),Darwin)
  CC = clang
  CFLAGS = -O3
  SHARED_FLAGS = -dynamiclib
  LIB = tinygiant/libtinygiant.dylib
  ifeq ($(ARCH),arm64)
    CFLAGS += -mcpu=apple-m1
  endif
  ifeq ($(ARCH),x86_64)
    CFLAGS += -march=native
  endif
else
  CC = gcc
  CFLAGS = -O3
  SHARED_FLAGS = -shared -fPIC
  ifneq (,$(filter $(ARCH),arm64 aarch64))
    CFLAGS += -march=armv8.2-a+dotprod
  endif
  ifeq ($(ARCH),x86_64)
    CFLAGS += -march=native
  endif
  LIB = tinygiant/libtinygiant.so
endif

SRC = csrc/libtinygiant.c

.PHONY: all clean test

all: $(LIB)

$(LIB): $(SRC)
	$(CC) $(SHARED_FLAGS) $(CFLAGS) -o $@ $<

test: csrc/test_libtinygiant.c $(SRC)
	$(CC) $(CFLAGS) -o test_libtinygiant csrc/test_libtinygiant.c -lm
	./test_libtinygiant
nws_q8dot_bench: tools/nws_q8dot_bench.c
	$(CC) $(CFLAGS) -o $@ $< -lm
nws_attention_bench: tools/nws_attention_bench.c $(SRC)
	$(CC) $(CFLAGS) -o $@ $< -lm
clean:
	rm -f tinygiant/libtinygiant.dylib tinygiant/libtinygiant.so test_libtinygiant nws_attention_bench
