UNAME := $(shell uname -s)
ARCH := $(shell uname -m)

ifeq ($(UNAME),Darwin)
  CC = clang
  CXX = clang++
  CFLAGS = -O3
  CXXFLAGS = -O3 -fno-objc-arc -std=c++17
  SHARED_FLAGS = -dynamiclib
  LIB = tinygiant/libtinygiant.dylib
  METAL_SRC = csrc/metal_attn.mm
  METAL_OBJ = csrc/metal_attn.o
  EXTRA_LIBS = -framework Metal -framework Foundation -framework CoreFoundation
  ifeq ($(ARCH),arm64)
    CFLAGS += -mcpu=apple-m1
    CXXFLAGS += -mcpu=apple-m1
  endif
  ifeq ($(ARCH),x86_64)
    CFLAGS += -march=native
    CXXFLAGS += -march=native
  endif
else
  CC = gcc
  CXX = g++
  CFLAGS = -O3
  CXXFLAGS = -O3
  SHARED_FLAGS = -shared -fPIC
  CFLAGS += -fPIC
  ifneq (,$(filter $(ARCH),arm64 aarch64))
    CFLAGS += -march=armv8.2-a+dotprod
  endif
  ifeq ($(ARCH),x86_64)
    CFLAGS += -march=native
  endif
  LIB = tinygiant/libtinygiant.so
  METAL_SRC = csrc/metal_attn_stub.c
  METAL_OBJ = csrc/metal_attn_stub.o
  EXTRA_LIBS =
endif

SRC = csrc/libtinygiant.c
OBJ = csrc/libtinygiant.o

.PHONY: all clean test test-metal help

all: $(LIB)

$(OBJ): $(SRC)
	$(CC) $(CFLAGS) -c -o $@ $<

ifeq ($(UNAME),Darwin)
$(METAL_OBJ): $(METAL_SRC) csrc/metal_attn.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<
else
$(METAL_OBJ): $(METAL_SRC) csrc/metal_attn.h
	$(CC) $(CFLAGS) -c -o $@ $<
endif

$(LIB): $(OBJ) $(METAL_OBJ)
ifeq ($(UNAME),Darwin)
	$(CXX) $(SHARED_FLAGS) $(CFLAGS) -o $@ $(OBJ) $(METAL_OBJ) $(EXTRA_LIBS)
else
	$(CC) $(SHARED_FLAGS) $(CFLAGS) -o $@ $(OBJ) $(METAL_OBJ) $(EXTRA_LIBS)
endif

test: csrc/test_libtinygiant.c $(SRC)
	$(CC) $(CFLAGS) -o test_libtinygiant csrc/test_libtinygiant.c -lm
	./test_libtinygiant

# Darwin-only: numerical agreement Metal vs CPU attention (synthetic, 1 layer)
test-metal: $(LIB) csrc/test_metal_attn.c
ifeq ($(UNAME),Darwin)
	$(CC) $(CFLAGS) -o test_metal_attn csrc/test_metal_attn.c \
		$(LIB) $(EXTRA_LIBS) -lm
	./test_metal_attn
else
	@echo "test-metal skipped (not Darwin)"
endif

nws_q8dot_bench: tools/nws_q8dot_bench.c
	$(CC) $(CFLAGS) -o $@ $< -lm
nws_attention_bench: tools/nws_attention_bench.c $(SRC)
	$(CC) $(CFLAGS) -o $@ $< -lm
clean:
	rm -f tinygiant/libtinygiant.dylib tinygiant/libtinygiant.so \
		test_libtinygiant test_metal_attn nws_attention_bench \
		$(OBJ) $(METAL_OBJ) csrc/metal_attn_stub.o csrc/metal_attn.o

help:
	@echo "TinyGiant build targets:"
	@echo "  make all                 Build libtinygiant (dylib/so into tinygiant/)"
	@echo "  make test                Build and run csrc/test_libtinygiant"
	@echo "  make test-metal          Darwin: Metal vs CPU attn numerical check"
	@echo "  make nws_q8dot_bench     Warm Q8+vdot microbenchmark binary"
	@echo "  make nws_attention_bench Attention microbenchmark binary"
	@echo "  make clean               Remove build artifacts"
	@echo ""
	@echo "After: pip install -e '.[tokenizer]' && make all"
	@echo "  tinygiant-setup --model ~/models/Qwen3-30B-A3B-Q4_K_M.gguf --cache ~/models/qwen3-30b-nws-cache --format q4"
	@echo "  tinygiant --model ... --cache ... --pin 48"
	@echo "  tinygiant-server --model ... --cache ... --port 8000"
