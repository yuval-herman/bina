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

build_debug() {
  gcc "${common_flags[@]}" \
    -DNN_MAIN \
    -g -O0 \
    -fsanitize=address,undefined \
    -o nn nn.c -lm
}

build_production() {
  gcc "${common_flags[@]}" \
    -DNN_MAIN \
    -O3 -DNDEBUG \
    -o nn nn.c -lm
}

build_emnist() {
  gcc "${common_flags[@]}" \
    -O3 -DNDEBUG \
    -fsanitize=address,undefined \
    -o emnist emnist.c -lm
}

run_tests() {
  echo "Building debug..."
  build_debug

  echo "Running debug executable..."
  if ./nn; then
    echo "Debug executable exited successfully."
  else
    status=$?
    echo "Debug executable failed with exit status $status."
    exit "$status"
  fi

  echo "Building production..."
  build_production

  echo "Production build succeeded."
}

case "$mode" in
  debug)
    build_debug
    ;;

  production|prod|release)
    build_production
    ;;

  emnist)
    build_emnist
    ;;

  test)
    run_tests
    ;;

  *)
    echo "Usage: $0 [debug|prod|emnist|test]"
    exit 1
    ;;
esac
