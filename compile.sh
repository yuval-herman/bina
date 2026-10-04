#!/usr/bin/env bash
set -euo pipefail

mode="${1:-debug}"

common_flags=(
  -std=gnu11
  -Werror
  -Wall
  -Wextra
  -Wconversion
  -Wdouble-promotion
  -Wpointer-arith
)

case "$mode" in
  debug)
    gcc "${common_flags[@]}" \
      -g -O0 \
      -fsanitize=address,undefined \
      -o main main.c -lm
    ;;

  production|prod|release)
    gcc "${common_flags[@]}" \
      -O2 -DNDEBUG \
      -o main main.c -lm
    ;;

  *)
    echo "Usage: $0 [debug|prod]"
    exit 1
    ;;
esac
