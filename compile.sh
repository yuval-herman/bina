#!/usr/bin/env bash
gcc -std=c11 -Wall -Wextra -Wconversion -Wdouble-promotion -Wpointer-arith -g -O0 -fsanitize=address,undefined -lm -o main main.c
