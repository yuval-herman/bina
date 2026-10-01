#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define ARR_LENGTH(arr) (sizeof(arr) / sizeof(arr[0]))

// Our float type, can be changed to double or whatever
typedef float Ntype;

// Check if two float are "close enough" to be considered equal
bool float_equals(Ntype a, Ntype b) { return fabs(a - b) < 1e-6; }

// ReLU
static inline Ntype activation(Ntype value) {
  return isfinite(value) ? value > 0 ? value : 0 : NAN;
}

Ntype feed_neuron(unsigned input_length, const Ntype *inputs,
                  const Ntype *weights, Ntype bias) {
  assert(input_length > 0 && "input length of neuron can't be zero");
  assert(inputs != NULL);
  assert(weights != NULL);

  Ntype output = 0;
  for (unsigned i = 0; i < input_length; i++) {
    output += inputs[i] * weights[i];
  }

  output = activation(output + bias);

  return output;
}

#define TEST_FLOAT(got, expected)                                              \
  if ((isnan(got) && isnan(expected)) || float_equals(got, expected))          \
    printf("Got %g which is correct\n", got);                                  \
  else                                                                         \
    printf("Got incorrect value! expected %g but got %g\n", expected, got);

int main(void) {
  Ntype inputs[] = {1, 1};
  Ntype weights[] = {1, 1};
  Ntype bias = 1;

  Ntype ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias);
  TEST_FLOAT(ret, 3.0f);
  inputs[0] = 2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias);
  TEST_FLOAT(ret, 4.0f);
  inputs[1] = 2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias);
  TEST_FLOAT(ret, 5.0f);
  weights[0] = 2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias);
  TEST_FLOAT(ret, 7.0f);
  weights[1] = 2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias);
  TEST_FLOAT(ret, 9.0f);
  weights[1] = -2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias);
  TEST_FLOAT(ret, 1.0f);
  weights[0] = -2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias);
  TEST_FLOAT(ret, 0.0f);

  bias = NAN;
  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias);
  TEST_FLOAT(ret, NAN);
  // should be 0 but just because of activation
  // function. Otherwise we had 2*-2+2*-2+1 = -7

  return 0;
}
