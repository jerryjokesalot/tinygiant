UNAME := $(shell uname -s)
ARCH := $(shell uname -m)

ifeq ($(UNAME),Darwin)
  CC = clang
  CFLAGS = -O3
  LIB = tinygiant/libtinygiant.dylib
  ifeq ($(ARCH),arm64)
    CFLAGS += -mcpu=apple-m1
  endif
  ifeq ($(ARCH),x86_64)
    CLFAGS += -march=native
  endif
else
  CC = gcc
  CFLAGS = -O3
  ifeq ($(ARCH),arm64)
    CFLAGS += -march=armv8.2-a+dotprod
  endif
  ifeq ($(ARCH),x86_64)
    CFLAGS += -march=native
  endif
  LIB = tinygiant/libtinygiant.so
endif

LIB = tinygiant/libtinygiant.so

SRC = csrc/libtinygiant.c

.PHONY: all clean test

all: $(LIB)

$(LIB): $(SRC)
	$(CC) -shared -fPIC $(CFLAGS) -o $@ $<

test: csrc/test_libtinygiant.c $(SRC)
	$(CC) $(CFLAGS) -o test_libtinygiant csrc/test_libtinygiant.c -lm
	./test_libtinygiant

clean:
	rm -f tinygiant/libtinygiant.dylib tinygiant/libtinygiant.so test_libtinygiant
