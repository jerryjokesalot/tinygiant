UNAME := $(shell uname -s)
ARCH := $(shell uname -m)

ifeq ($(UNAME),Darwin)
  CC = clang
  CFLAGS = -shared -O3 -mcpu=apple-m1
  LIB = tinygiant/libtinygiant.dylib
else
  CC = gcc
  CFLAGS = -shared -fPIC -O3 -march=armv8.2-a+dotprod
  LIB = tinygiant/libtinygiant.so
endif

SRC = csrc/libtinygiant.c

.PHONY: all clean

all: $(LIB)

$(LIB): $(SRC)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f tinygiant/libtinygiant.dylib tinygiant/libtinygiant.so
