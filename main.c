#include <assert.h>
#include <stdio.h>
#include <math.h>

// Our float type, can be changed to double or whatever
typedef float Ntype;

// ReLU
static inline Ntype activation(Ntype value) { return value > 0 ? value : 0; }

Ntype feed_neuron(unsigned input_length, const Ntype *inputs, const Ntype *weights,
                  Ntype bias) {
  assert(input_length > 0 && "input length of neuron can't be zero");
  assert(inputs != NULL);
  assert(weights != NULL);

  Ntype output = 0;
  for (unsigned i = 0; i < input_length; i++) {
    output += inputs[i] * weights[i];
  }

  // Should this realy be an assert? What if we infinity while learning? Can
  // this really be preveneted always? Maybe returning NaN on such errors is the
  // correct way, and the caller should handle that.
  assert(isfinite(output) && "output is invalid");

  return activation(output + bias);
}

#define ARR_LENGTH(arr) (sizeof(arr) / sizeof(arr[0]))

int main(void) {
  Ntype inputs[] = {1,1};
  Ntype weights[] = {1,1};
  Ntype bias = 1;

  Ntype ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights,bias);
  printf("Got: %g\n", ret); // should be 3

  inputs[0] = 2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias);
  printf("Got: %g\n", ret); // should be 4

  inputs[1] = 2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias);
  printf("Got: %g\n", ret); // should be 5

  weights[0] = 2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias);
  printf("Got: %g\n", ret); // should be 7

  weights[1] = 2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias);
  printf("Got: %g\n", ret); // should be 9

  return 0;
}
