#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define ANSI_RED "\033[31m"
#define ANSI_GREEN "\033[32m"
#define ANSI_RESET "\033[0m"

#define ARR_LENGTH(arr) (sizeof(arr) / sizeof(arr[0]))

// Our float type, can be changed to double or whatever
typedef float NNtype;

typedef struct {
  NNtype *weights;
  unsigned input_length; // weights length are input * output
  NNtype *biases;
  NNtype *outputs;
  unsigned output_length; // biases and outputs have the same length
} NNLayer;

static inline NNtype *get_weight(NNLayer *layer, unsigned output_idx,
                                 unsigned weight_idx) {
  return layer->weights + output_idx * (layer->output_length - 1) + weight_idx;
}

bool init_layer(NNLayer *layer, unsigned input_length, unsigned output_length) {
  layer->input_length = input_length;
  layer->output_length = output_length;

  size_t weights_length = input_length * output_length;
  // biases == output => biases + output = output*2
  NNtype *memory =
      malloc((weights_length + output_length * 2) * sizeof(NNtype));

  if (memory == NULL) {
    return false;
  }

  layer->weights = memory;
  memory += weights_length;

  layer->biases = memory;
  memory += output_length;

  layer->outputs = memory;
  memory += output_length;

  return true;
}

void free_layer(NNLayer *layer) {
  // Only one free since we use only one malloc
  free(layer->weights);
  layer->weights = NULL;
  layer->biases = NULL;
  layer->outputs = NULL;
}

// Check if two float are "close enough" to be considered equal
bool float_equals(NNtype a, NNtype b) { return fabs(a - b) < 1e-6; }

// ReLU
static inline NNtype activation(NNtype value) {
  // The check is made in such a way that NaN values would progagte through here
  // and not turn into zeros.
  return value <= 0 ? 0 : value;
}

NNtype feed_neuron(unsigned input_length, const NNtype *inputs,
                   const NNtype *weights, NNtype bias) {
  assert(input_length > 0 && "input length of neuron can't be zero");
  assert(inputs != NULL);
  assert(weights != NULL);

  NNtype output = 0;
  for (unsigned i = 0; i < input_length; i++) {
    output += inputs[i] * weights[i];
  }

  output = activation(output + bias);

  return output;
}

void feed_layer(NNLayer *layer, const NNtype *inputs) {
  assert(layer);
  assert(inputs);
  for (unsigned output_idx = 0; output_idx < layer->output_length;
       output_idx++) {
    layer->outputs[output_idx] = feed_neuron(layer->input_length, inputs,
                                             get_weight(layer, output_idx, 0),
                                             layer->biases[output_idx]);
  }
}

#define TEST_FLOAT(got, expected)                                              \
  do {                                                                         \
    if ((isnan(got) && isnan(expected)) || float_equals(got, expected))        \
      printf(ANSI_GREEN "Got %g which is correct\n" ANSI_RESET, got);          \
    else                                                                       \
      printf(ANSI_RED "Got incorrect value! (line %d) expected %g but got "    \
                      "%g\n" ANSI_RESET,                                       \
             __LINE__, expected, got);                                         \
  } while (0)

void test_neuron() {
  NNtype inputs[] = {1, 1};
  NNtype weights[] = {1, 1};
  NNtype bias = 1;

  NNtype ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias);
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
}

int main(void) {
  test_neuron();

  NNLayer layer = {0};
  init_layer(&layer, 2, 3);

  for (unsigned i = 0; i < layer.output_length * layer.input_length; i++) {
    layer.weights[i] = 1;
  }

  for (unsigned i = 0; i < layer.output_length; i++) {
    layer.biases[i] = 1;
  }

  NNtype inputs[2] = {1, 1};

  feed_layer(&layer, inputs);
  TEST_FLOAT(layer.outputs[0], 3.0f);
  TEST_FLOAT(layer.outputs[1], 3.0f);
  TEST_FLOAT(layer.outputs[2], 3.0f);

  for (unsigned i = 0; i < layer.output_length * layer.input_length; i++) {
    layer.weights[i] = 2;
  }

  feed_layer(&layer, inputs);
  TEST_FLOAT(layer.outputs[0], 5.0f);
  TEST_FLOAT(layer.outputs[1], 5.0f);
  TEST_FLOAT(layer.outputs[2], 5.0f);

  for (unsigned i = 0; i < layer.output_length; i++) {
    layer.biases[i] = 2;
  }

  feed_layer(&layer, inputs);
  TEST_FLOAT(layer.outputs[0], 6.0f);
  TEST_FLOAT(layer.outputs[1], 6.0f);
  TEST_FLOAT(layer.outputs[2], 6.0f);

  for (unsigned i = 0; i < layer.output_length * layer.input_length; i++) {
    layer.weights[i] = (float)i;
  }

  for (unsigned i = 0; i < layer.output_length; i++) {
    layer.biases[i] = (float)layer.output_length - (float)i;
  }

  for (unsigned i = 0; i < ARR_LENGTH(inputs); i++) {
    inputs[i] = (float)i;
  }

  feed_layer(&layer, inputs);
  TEST_FLOAT(layer.outputs[0], 4.0f);
  TEST_FLOAT(layer.outputs[1], 5.0f);
  TEST_FLOAT(layer.outputs[2], 6.0f);

  free_layer(&layer);
}
