#!/usr/bin/env bash
gcc -std=c11 -Wall -Wextra -Wconversion -Wdouble-promotion -Wpedantic -g -O0 -fsanitize=address,undefined -o main main.c
