UNAME := $(shell uname -s)
ARCH := $(shell uname -m)

CC = gcc
CFLAGS = -shared -fPIC -O3

ifeq ($(ARCH),x86_64)
  CFLAGS += -mavx2 -mf16c -mfma
endif

LIB = tinygiant/libtinygiant.so

SRC = csrc/libtinygiant.c

.PHONY: all clean test

all: $(LIB)

$(LIB): $(SRC)
	$(CC) $(CFLAGS) -o $@ $<

test: csrc/test_libtinygiant.c $(SRC)
	$(CC) -O3 -mavx2 -mf16c -mfma -o test_libtinygiant csrc/test_libtinygiant.c -lm
	./test_libtinygiant

clean:
	rm -f tinygiant/libtinygiant.dylib tinygiant/libtinygiant.so test_libtinygiant
