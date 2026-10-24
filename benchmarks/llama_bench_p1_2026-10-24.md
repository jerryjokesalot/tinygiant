| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | BLAS       |       8 |             pp1 |          8.11 ± 2.71 |
| qwen3moe 30B.A3B Q4_K - Medium |  17.28 GiB |    30.53 B | BLAS       |       8 |           tg128 |          1.06 ± 0.05 |

build: 73adf89 (1)
