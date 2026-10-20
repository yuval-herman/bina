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
    -DNN_TEST \
    -g -O0 \
    -fsanitize=address,undefined \
    -o nn nn.c -lm
}

build_production() {
  gcc "${common_flags[@]}" \
    -DNN_BENCH \
    -O3 -DNDEBUG \
    -o nn nn.c -lm
}

build_experiment() {
  gcc "${common_flags[@]}" \
    -Wno-error=unused-function \
    -O2 \
    -o experiment experiment.c plot.c -lm
}

build_plotview() {
  gcc "${common_flags[@]}" \
    -O2 \
    -o plotview plotview.c plot.c -lm
}

build_plot_demo() {
  gcc "${common_flags[@]}" \
    -O2 \
    -o plot-demo plot_demo.c plot.c -lm
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

  experiment)
    build_experiment
    ;;

  plotview)
    build_plotview
    ;;

  plot-demo)
    build_plot_demo
    ;;

  test)
    run_tests
    ;;

  *)
    echo "Usage: $0 [debug|prod|experiment|plotview|plot-demo|test]"
    exit 1
    ;;
esac
