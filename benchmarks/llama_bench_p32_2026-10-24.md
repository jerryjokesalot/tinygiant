| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | BLAS       |       8 |            pp32 |          2.23 ± 0.18 |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | BLAS       |       8 |           tg128 |          1.04 ± 0.11 |

build: 73adf89 (1)
