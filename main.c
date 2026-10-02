#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

typedef struct {
  // Points to the begining of the NN weights, biases, and output memory array.
  // Saved here as a conveniece for freeing and in case we ever change the
  // memory layout.
  void *memory_start;
  // A convenience pointer and length for the output of the last layer.
  NNtype *final_output;
  unsigned final_output_length;
  NNLayer *layers;
  unsigned layers_length;
} NNNetwork;

typedef struct {
  void *memory_start;
  NNNetwork *network;
  NNtype *slopes;
  unsigned slopes_length;
} NNTrainer;

static inline NNtype *get_weight(NNLayer *layer, unsigned output_idx,
                                 unsigned weight_idx) {
  return layer->weights + output_idx * layer->input_length + weight_idx;
}

// layers_neurons is a count of how many outputs a layer has. The first items in
// that array is the amount of inputs the network may accept, then the rest of
// the numbers are the outputs for every sequential layer.
bool init_network(NNNetwork *network, unsigned layers_length,
                  unsigned *layers_neurons) {
  assert(layers_length >= 2 && "A network must be at least one layer long");

  size_t memory_length = 0;
  for (unsigned i = 0; i < layers_length - 1; i++) {
    unsigned input_length = layers_neurons[i];
    unsigned output_length = layers_neurons[i + 1];
    size_t weights_length = input_length * output_length;

    // biases == output => biases + output = output*2
    memory_length += (weights_length + output_length * 2) * sizeof(NNtype);
  }

  NNtype *memory = malloc(memory_length);
  if (memory == NULL) {
    return false;
  }
  NNLayer *layers = malloc((layers_length - 1) * sizeof(NNLayer));
  if (layers == NULL) {
    free(memory);
    return false;
  }

  network->layers = layers;
  network->layers_length = layers_length - 1;
  network->memory_start = memory;

  for (unsigned i = 0; i < layers_length - 1; i++) {
    NNLayer *layer = layers + i;
    unsigned input_length = layers_neurons[i];
    unsigned output_length = layers_neurons[i + 1];
    size_t weights_length = input_length * output_length;

    layer->input_length = input_length;
    layer->output_length = output_length;

    layer->weights = memory;
    memory += weights_length;

    layer->biases = memory;
    memory += output_length;

    layer->outputs = memory;
    memory += output_length;
  }

  network->final_output = network->layers[network->layers_length - 1].outputs;
  network->final_output_length =
      network->layers[network->layers_length - 1].output_length;
  return true;
}

void free_network(NNNetwork *network) {
  free(network->layers);
  free(network->memory_start);
  *network = (NNNetwork){0};
}

bool init_trainer(NNTrainer *trainer, NNNetwork *network) {
  assert(trainer);
  assert(network);
  assert(network->layers_length && network->memory_start &&
         "Network must be initialized before training starts");

  size_t memory_length = 0;
  unsigned slopes_length = 0;
  for (unsigned i = 0; i < network->layers_length; i++) {
    NNLayer l = network->layers[i];
    assert(l.input_length);
    assert(l.output_length);
    unsigned weights_length = l.input_length * l.output_length;
    unsigned layer_size = weights_length + l.output_length;
    memory_length += layer_size * sizeof(NNtype);
    slopes_length += layer_size;
  }

  NNtype *memory = malloc(memory_length);
  if (memory == NULL) {
    return false;
  }

  trainer->network = network;
  trainer->memory_start = memory;
  trainer->slopes = memory;
  trainer->slopes_length = slopes_length;

  return true;
}

void free_trainer(NNTrainer *trainer) {
  free(trainer->memory_start);
  *trainer = (NNTrainer){0};
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

void feed_network(NNNetwork *network, const NNtype *inputs) {
  assert(network);
  assert(inputs);
  feed_layer(network->layers + 0, inputs);

  for (unsigned layer_idx = 1; layer_idx < network->layers_length;
       layer_idx++) {
    assert(network->layers[layer_idx].input_length ==
           network->layers[layer_idx - 1].output_length);
    feed_layer(network->layers + layer_idx,
               network->layers[layer_idx - 1].outputs);
  }
}

NNtype calculate_loss(unsigned inputs_length, const NNtype *values,
                      const NNtype *expected) {
  assert(inputs_length);
  assert(values);
  assert(expected);

  NNtype loss = 0;
  // If the loss is huge, we might to infinity in this loop. This is actually
  // fine, loss should go down, not up, so going to infinity means something is
  // wrong, and its a good place to catch that.
  for (unsigned i = 0; i < inputs_length; i++) {
    const NNtype input_loss = values[i] - expected[i];
    loss += input_loss * input_loss;
  }

  return loss / (NNtype)inputs_length;
}

#define TEST_FLOAT(got, expected)                                              \
  do {                                                                         \
    if ((isnan(got) && isnan(expected)) || float_equals(got, expected))        \
      printf(ANSI_GREEN "Got %g which is correct\n" ANSI_RESET, (double)got);  \
    else                                                                       \
      printf(ANSI_RED "Got incorrect value! (line %d) expected %g but got "    \
                      "%g\n" ANSI_RESET,                                       \
             __LINE__, (double)expected, (double)got);                         \
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
  // should be 0 but just because of activation
  // function. Otherwise we had 2*-2+2*-2+1 = -7
  TEST_FLOAT(ret, NAN);
}

void test_layer() {
  {
    NNNetwork network = {0};
    init_network(&network, 2, (unsigned[]){2, 3});
    NNLayer layer = network.layers[0];

    for (unsigned i = 0; i < layer.output_length * layer.input_length; i++) {
      layer.weights[i] = 1;
    }

    for (unsigned i = 0; i < layer.output_length; i++) {
      layer.biases[i] = 1;
    }

    NNtype inputs[] = {1, 1};
    assert(ARR_LENGTH(inputs) == layer.input_length);

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

    free_network(&network);
  }
  {
    NNNetwork network = {0};
    init_network(&network, 2, (unsigned[]){3, 3});
    NNLayer layer = network.layers[0];

    for (unsigned i = 0; i < layer.output_length * layer.input_length; i++) {
      layer.weights[i] = 1;
    }

    for (unsigned i = 0; i < layer.output_length; i++) {
      layer.biases[i] = 1;
    }

    NNtype inputs[] = {1, 1, 1};
    assert(ARR_LENGTH(inputs) == layer.input_length);

    feed_layer(&layer, inputs);
    TEST_FLOAT(layer.outputs[0], 4.0f);
    TEST_FLOAT(layer.outputs[1], 4.0f);
    TEST_FLOAT(layer.outputs[2], 4.0f);

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
    TEST_FLOAT(layer.outputs[0], 8.0f);
    TEST_FLOAT(layer.outputs[1], 16.0f);
    TEST_FLOAT(layer.outputs[2], 24.0f);

    free_network(&network);
  }
  {
    NNNetwork network = {0};
    init_network(&network, 2, (unsigned[]){3, 2});
    NNLayer layer = network.layers[0];

    for (unsigned i = 0; i < layer.output_length * layer.input_length; i++) {
      layer.weights[i] = 1;
    }

    for (unsigned i = 0; i < layer.output_length; i++) {
      layer.biases[i] = 1;
    }

    NNtype inputs[] = {1, 1, 1};
    assert(ARR_LENGTH(inputs) == layer.input_length);

    feed_layer(&layer, inputs);
    TEST_FLOAT(layer.outputs[0], 4.0f);
    TEST_FLOAT(layer.outputs[1], 4.0f);

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
    TEST_FLOAT(layer.outputs[0], 7.0f);
    TEST_FLOAT(layer.outputs[1], 15.0f);

    free_network(&network);
  }
  {
    NNNetwork network = {0};
    init_network(&network, 2, (unsigned[]){3, 2});
    NNLayer layer = network.layers[0];

    for (unsigned i = 0; i < layer.output_length * layer.input_length; i++) {
      layer.weights[i] = -1;
    }

    for (unsigned i = 0; i < layer.output_length; i++) {
      layer.biases[i] = 1;
    }

    NNtype inputs[] = {1, 1, 1};
    assert(ARR_LENGTH(inputs) == layer.input_length);

    feed_layer(&layer, inputs);
    TEST_FLOAT(layer.outputs[0], 0.0f);
    TEST_FLOAT(layer.outputs[1], 0.0f);

    for (unsigned i = 0; i < layer.output_length * layer.input_length; i++) {
      layer.weights[i] = (float)i;
    }

    for (unsigned i = 0; i < layer.output_length; i++) {
      layer.biases[i] = (float)layer.output_length - (float)i;
    }

    for (unsigned i = 0; i < ARR_LENGTH(inputs); i++) {
      inputs[i] = (float)i - 1.5f;
    }

    feed_layer(&layer, inputs);
    TEST_FLOAT(layer.outputs[0], 2.5f);
    TEST_FLOAT(layer.outputs[1], 0.0f);

    free_network(&network);
  }
}

void test_network() {
  {
    NNNetwork network = {0};
    init_network(&network, 3, (unsigned[]){2, 3, 1});

    NNtype inputs[2] = {-0.5, 2};

    memcpy(network.layers[0].biases, (NNtype[3]){1, 2, 3}, 3 * sizeof(NNtype));
    memcpy(network.layers[1].biases, (NNtype[1]){4}, 1 * sizeof(NNtype));

    memcpy(network.layers[0].weights, (NNtype[2 * 3]){6, 5, 4, 3, 2, 1},
           2 * 3 * sizeof(NNtype));
    memcpy(network.layers[1].weights, (NNtype[3 * 1]){0.5, 0.5, 0.5},
           3 * 1 * sizeof(NNtype));

    feed_network(&network, inputs);

    TEST_FLOAT(network.final_output[0], 13.0f);

    free_network(&network);
  }
  {
    NNNetwork network = {0};
    init_network(&network, 3, (unsigned[]){2, 2, 1});

    NNtype xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    NNtype xor_expected[4] = {0, 1, 1, 0};

    memcpy(network.layers[0].biases, (NNtype[2]){0, 0}, 2 * sizeof(NNtype));
    memcpy(network.layers[1].biases, (NNtype[1]){0}, 1 * sizeof(NNtype));

    memcpy(network.layers[0].weights, (NNtype[2 * 2]){-1, 1, 1, -1},
           2 * 2 * sizeof(NNtype));
    memcpy(network.layers[1].weights, (NNtype[2 * 1]){1, 1},
           2 * 1 * sizeof(NNtype));

    for (unsigned i = 0; i < ARR_LENGTH(xor_inputs); i++) {
      feed_network(&network, xor_inputs[i]);

      TEST_FLOAT(network.final_output[0], xor_expected[i]);

      NNtype loss = calculate_loss(1, network.final_output, &xor_expected[i]);

      TEST_FLOAT(loss, 0.0f);
    }

    network.layers[0].weights[0] = 1;

    NNtype expected_loss[4] = {0, 0, 1, 4};
    for (unsigned i = 0; i < ARR_LENGTH(xor_inputs); i++) {
      feed_network(&network, xor_inputs[i]);

      NNtype loss = calculate_loss(1, network.final_output, &xor_expected[i]);

      TEST_FLOAT(loss, expected_loss[i]);
    }

    free_network(&network);
  }
}

int main(void) {
  test_neuron();
  test_layer();
  test_network();
}
