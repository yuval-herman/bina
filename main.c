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
typedef NNtype (*NNactivation)(NNtype);

typedef enum {
  NN_NO_ACTIVATION = 0,
  NN_RELU,
} NNActivationType;

typedef struct {
  NNtype *weights;
  unsigned input_length; // weights length are input * output
  NNtype *biases;
  NNtype *outputs;
  unsigned output_length;           // biases and outputs have the same length
  NNActivationType activation_type; // Pass NULL here to have no activation
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
  NNNetwork *network;
  NNtype *slopes;
  unsigned slopes_length;
  NNtype *deltas;
  unsigned deltas_length;
  // This array containes pointers to the start of each layer's slope values.
  NNtype **layer_slopes;
  // This array containes pointers to the start of each layer's delta values.
  NNtype **layer_deltas;
} NNTrainer;

static inline NNtype *get_weight(NNLayer *layer, unsigned output_idx,
                                 unsigned weight_idx) {
  return layer->weights + output_idx * layer->input_length + weight_idx;
}

static inline NNtype random_NNtype(NNtype min, NNtype max) {
  return min +
         ((NNtype)rand() / ((NNtype)RAND_MAX + (NNtype)1.0)) * (max - min);
}

static inline NNtype noop_activation(NNtype value) { return value; }

static inline NNtype relu_activation(NNtype value) {
  // The check is made in such a way that NaN values would progagte through here
  // and not turn into zeros.
  return value <= 0 ? 0 : value;
}

static inline NNtype relu_derivative(NNtype value) {
  return value <= 0 ? 0 : 1;
}

const struct {
  NNactivation activation;
  NNactivation derivative;
} activation_functions[] = {
    [NN_NO_ACTIVATION] = {.activation = noop_activation,
                          .derivative = noop_activation},
    [NN_RELU] = {.activation = relu_activation, .derivative = relu_derivative},
};

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

    layer->activation_type = i < layers_length - 2 ? NN_RELU : NN_NO_ACTIVATION;
  }

  network->final_output = network->layers[network->layers_length - 1].outputs;
  network->final_output_length =
      network->layers[network->layers_length - 1].output_length;
  return true;
}

void print_network(NNNetwork network) {
  for (unsigned layer_idx = 0; layer_idx < network.layers_length; layer_idx++) {
    const NNLayer layer = network.layers[layer_idx];
    const unsigned weights_length = layer.input_length * layer.output_length;
    printf("layer %u\n", layer_idx);
    for (unsigned weight_idx = 0; weight_idx < weights_length; weight_idx++) {
      printf("weight[%u] = %g\n", weight_idx,
             (double)layer.weights[weight_idx]);
    }
    for (unsigned bias_idx = 0; bias_idx < layer.output_length; bias_idx++) {
      printf("bias[%u] = %g\n", bias_idx, (double)layer.biases[bias_idx]);
    }
  }
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

  unsigned slopes_length = 0;
  unsigned deltas_length = 0;
  for (unsigned i = 0; i < network->layers_length; i++) {
    const NNLayer l = network->layers[i];
    assert(l.input_length);
    assert(l.output_length);
    const unsigned weights_length = l.input_length * l.output_length;
    const unsigned layer_size = weights_length + l.output_length;
    slopes_length += layer_size;
    deltas_length += l.output_length;
  }

  NNtype *slopes_memory = malloc(slopes_length * sizeof(NNtype));
  if (slopes_memory == NULL) {
    return false;
  }

  NNtype *deltas_memory = malloc(deltas_length * sizeof(NNtype));
  if (deltas_memory == NULL) {
    free(slopes_memory);
    return false;
  }

  trainer->layer_slopes =
      malloc(network->layers_length * sizeof(*trainer->layer_slopes));

  if (trainer->layer_slopes == NULL) {
    free(slopes_memory);
    free(deltas_memory);
    return false;
  }

  trainer->layer_deltas =
      malloc(network->layers_length * sizeof(*trainer->layer_deltas));

  if (trainer->layer_deltas == NULL) {
    free(slopes_memory);
    free(deltas_memory);
    free(trainer->layer_slopes);
    return false;
  }

  trainer->network = network;
  trainer->slopes = slopes_memory;
  trainer->deltas = deltas_memory;
  trainer->slopes_length = slopes_length;
  trainer->deltas_length = deltas_length;

  trainer->layer_slopes[0] = trainer->slopes;
  trainer->layer_deltas[0] = trainer->deltas;

  for (unsigned i = 0; i < network->layers_length - 1; i++) {
    const NNLayer l = network->layers[i];
    const unsigned weights_length = l.input_length * l.output_length;
    const unsigned layer_size = weights_length + l.output_length;

    trainer->layer_slopes[i + 1] = trainer->layer_slopes[i] + layer_size;
    trainer->layer_deltas[i + 1] = trainer->layer_deltas[i] + l.output_length;
  }

  return true;
}

void free_trainer(NNTrainer *trainer) {
  free(trainer->slopes);
  free(trainer->deltas);
  free(trainer->layer_slopes);
  free(trainer->layer_deltas);
  *trainer = (NNTrainer){0};
}

// Check if two float are "close enough" to be considered equal with a dynamic
// tolerance level
static inline bool float_equals_tolerance(NNtype a, NNtype b,
                                          NNtype tolerance) {
  return fabsf(a - b) < fmaxf(1e-6f, tolerance * fmaxf(fabsf(a), fabsf(b)));
}

NNtype feed_neuron(unsigned input_length, const NNtype *inputs,
                   const NNtype *weights, NNtype bias,
                   NNactivation activation) {
  assert(input_length > 0 && "input length of neuron can't be zero");
  assert(inputs != NULL);
  assert(weights != NULL);

  NNtype output = 0;
  for (unsigned i = 0; i < input_length; i++) {
    output += inputs[i] * weights[i];
  }

  // TODO save value vefore activation as well for derivative calculation
  if (activation)
    output = activation(output + bias);
  else
    output += bias;

  return output;
}

void feed_layer(NNLayer *layer, const NNtype *inputs) {
  assert(layer);
  assert(inputs);
  for (unsigned output_idx = 0; output_idx < layer->output_length;
       output_idx++) {
    layer->outputs[output_idx] =
        feed_neuron(layer->input_length, inputs,
                    get_weight(layer, output_idx, 0), layer->biases[output_idx],
                    activation_functions[layer->activation_type].activation);
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

// `inputs` and `expected` are both 2d arrays that are expected to be in
// contiguous memory.
NNtype calculate_batch_network_loss(NNNetwork *network, unsigned inputs_length,
                                    const NNtype *inputs,
                                    const NNtype *expected) {
  double avg_loss = 0;
  for (unsigned i = 0; i < inputs_length; i++) {
    feed_network(network, &inputs[i * network->layers[0].input_length]);
    avg_loss += (double)calculate_loss(
        network->final_output_length, network->final_output,
        &expected[i * network->final_output_length]);
  }
  return (NNtype)avg_loss / (NNtype)inputs_length;
}

// How should this function handle NaN and infinity cropping up? Abort? This is
// something that should be thought about.
// `inputs` and `expected` are both 2d arrays that are expected to be in
// contiguous memory.
void calculate_batch_slope_nudge_version(NNTrainer *trainer,
                                         unsigned inputs_length,
                                         const NNtype *inputs,
                                         const NNtype *expected) {
  assert(inputs_length && inputs);
  assert(expected);
  assert(trainer);
  NNNetwork *network = trainer->network;
  assert(network && "trainer must be initialised before training");
  const NNtype nudge_value = 1e-4f;

  double avg_loss_before =
      calculate_batch_network_loss(network, inputs_length, inputs, expected);

  for (unsigned layer_idx = 0; layer_idx < network->layers_length;
       layer_idx++) {
    const NNLayer layer = network->layers[layer_idx];
    const unsigned weights_length = layer.input_length * layer.output_length;

    for (unsigned weight_idx = 0; weight_idx < weights_length; weight_idx++) {
      NNtype previous_nn_value = layer.weights[weight_idx];
      layer.weights[weight_idx] += nudge_value;

      double avg_loss_after = calculate_batch_network_loss(
          network, inputs_length, inputs, expected);

      trainer->layer_slopes[layer_idx][weight_idx] =
          (NNtype)(avg_loss_after - avg_loss_before) / nudge_value;

      layer.weights[weight_idx] = previous_nn_value;
    }
    for (unsigned bias_idx = 0; bias_idx < layer.output_length; bias_idx++) {
      NNtype previous_nn_value = layer.biases[bias_idx];
      layer.biases[bias_idx] += nudge_value;

      double avg_loss_after = calculate_batch_network_loss(
          network, inputs_length, inputs, expected);

      trainer->layer_slopes[layer_idx][weights_length + bias_idx] =
          (NNtype)(avg_loss_after - avg_loss_before) / nudge_value;

      layer.biases[bias_idx] = previous_nn_value;
    }
  }
}

// Calculates and applies a single backpropagation step for a batch of inputs.
void batch_backpropagation_step(NNTrainer *trainer, unsigned inputs_length,
                                const NNtype *inputs, const NNtype *expected) {
  assert(inputs_length && inputs);
  assert(expected);
  assert(trainer);
  NNNetwork *network = trainer->network;
  assert(network && "trainer must be initialised before training");

  memset(trainer->slopes, 0, trainer->slopes_length * sizeof(NNtype));

  for (unsigned input_batch_idx = 0; input_batch_idx < inputs_length;
       input_batch_idx++) {
    const NNtype *input_batch =
        inputs + network->layers[0].input_length * input_batch_idx;

    memset(trainer->deltas, 0, trainer->deltas_length * sizeof(NNtype));
    feed_network(network, input_batch);

    // delta for the output layer, the last layer
    for (unsigned neuron_idx = 0; neuron_idx < network->final_output_length;
         neuron_idx++) {
      const NNtype neuron_expected =
          expected[network->final_output_length * input_batch_idx + neuron_idx];
      const NNtype neuron_output = network->final_output[neuron_idx];
      NNtype *layer_deltas = trainer->layer_deltas[network->layers_length - 1];

      layer_deltas[neuron_idx] = (neuron_output - neuron_expected) * 2 /
                                 (NNtype)network->final_output_length;
    }

    // deltas for the rest of the layers
    for (unsigned layer_counter = 0; layer_counter < network->layers_length - 1;
         layer_counter++) {
      const unsigned layer_idx = network->layers_length - layer_counter - 2;
      const NNLayer *layer = &network->layers[layer_idx];
      NNtype *layer_deltas = trainer->layer_deltas[layer_idx];
      NNLayer *last_visited_layer = &network->layers[layer_idx + 1];
      NNtype *last_visited_layer_deltas = trainer->layer_deltas[layer_idx + 1];

      for (unsigned neuron_idx = 0; neuron_idx < layer->output_length;
           neuron_idx++) {

        for (unsigned last_layer_neuron_idx = 0;
             last_layer_neuron_idx < last_visited_layer->output_length;
             last_layer_neuron_idx++) {

          layer_deltas[neuron_idx] +=
              *get_weight(last_visited_layer, last_layer_neuron_idx,
                          neuron_idx) *
              last_visited_layer_deltas[last_layer_neuron_idx];
        }
        layer_deltas[neuron_idx] *=
            activation_functions[layer->activation_type].derivative(
                // TODO use the output before activation to get the correct
                // result for functions other then ReLU
                layer->outputs[neuron_idx]);
      }
    }

    // first layer needs special handling because we don't have the input saved
    // within the layers

    {
      const unsigned layer_idx = 0;
      const NNLayer layer = network->layers[layer_idx];
      NNtype *layer_slopes = trainer->layer_slopes[layer_idx];
      const unsigned weights_length = layer.input_length * layer.output_length;

      for (unsigned bias_idx = 0;
           bias_idx < network->layers[layer_idx].output_length; bias_idx++) {
        trainer->layer_slopes[layer_idx][weights_length + bias_idx] +=
            trainer->layer_deltas[layer_idx][bias_idx];
      }

      for (unsigned input_idx = 0; input_idx < layer.input_length;
           input_idx++) {
        for (unsigned neuron_idx = 0; neuron_idx < layer.output_length;
             neuron_idx++) {
          layer_slopes[input_idx + layer.input_length * neuron_idx] +=
              input_batch[input_idx] *
              trainer->layer_deltas[layer_idx][neuron_idx];
        }
      }
    }

    for (unsigned layer_idx = 1; layer_idx < network->layers_length;
         layer_idx++) {
      const NNLayer layer = network->layers[layer_idx];
      const NNLayer preceding_layer = network->layers[layer_idx - 1];
      NNtype *layer_slopes = trainer->layer_slopes[layer_idx];
      const unsigned weights_length = layer.input_length * layer.output_length;

      for (unsigned bias_idx = 0;
           bias_idx < network->layers[layer_idx].output_length; bias_idx++) {
        trainer->layer_slopes[layer_idx][weights_length + bias_idx] +=
            trainer->layer_deltas[layer_idx][bias_idx];
      }

      for (unsigned input_idx = 0; input_idx < layer.input_length;
           input_idx++) {
        for (unsigned neuron_idx = 0; neuron_idx < layer.output_length;
             neuron_idx++) {
          layer_slopes[input_idx + layer.input_length * neuron_idx] +=
              preceding_layer.outputs[input_idx] *
              trainer->layer_deltas[layer_idx][neuron_idx];
        }
      }
    }
  }

  for (unsigned slope_idx = 0; slope_idx < trainer->slopes_length;
       slope_idx++) {
    trainer->slopes[slope_idx] /= (NNtype)inputs_length;
  }
}

void apply_slopes(NNTrainer *trainer, NNtype learning_rate) {
  assert(trainer);
  NNNetwork *network = trainer->network;
  assert(network && "trainer must be initialised before training");

  for (unsigned layer_idx = 0; layer_idx < network->layers_length;
       layer_idx++) {
    const NNLayer layer = network->layers[layer_idx];
    const unsigned weights_length = layer.input_length * layer.output_length;

    for (unsigned weight_idx = 0; weight_idx < weights_length; weight_idx++) {
      layer.weights[weight_idx] -=
          trainer->layer_slopes[layer_idx][weight_idx] * learning_rate;
    }
    for (unsigned bias_idx = 0; bias_idx < layer.output_length; bias_idx++) {
      layer.biases[bias_idx] -=
          trainer->layer_slopes[layer_idx][weights_length + bias_idx] *
          learning_rate;
    }
  }
}

#define BOOL_STRING(value) ((value) ? "true" : "false")
#define TEST_BOOL(got, expected)                                               \
  do {                                                                         \
    if ((got) == (expected))                                                   \
      printf(ANSI_GREEN "Got %s which is correct\n" ANSI_RESET,                \
             BOOL_STRING(got));                                                \
    else                                                                       \
      printf(ANSI_RED "Got incorrect value! (line %d) expected %s but got "    \
                      "%s\n" ANSI_RESET,                                       \
             __LINE__, BOOL_STRING(expected), BOOL_STRING(got));               \
  } while (0)

#define TEST_FLOAT(got, expected) TEST_FLOAT_TOL(got, expected, 1e-6f)
#define TEST_FLOAT_TOL(got, expected, tolerance)                               \
  do {                                                                         \
    if ((isnan(got) && isnan(expected)) ||                                     \
        float_equals_tolerance(got, expected, tolerance))                      \
      printf(ANSI_GREEN "Got %g which is correct\n" ANSI_RESET,                \
             (double)(got));                                                   \
    else                                                                       \
      printf(ANSI_RED "Got incorrect value! (line %d) expected %g but got "    \
                      "%g\n" ANSI_RESET,                                       \
             __LINE__, (double)(expected), (double)(got));                     \
  } while (0)

void test_neuron() {
  NNtype inputs[] = {1, 1};
  NNtype weights[] = {1, 1};
  NNtype bias = 1;

  NNtype ret =
      feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
  TEST_FLOAT(ret, 3.0f);
  inputs[0] = 2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
  TEST_FLOAT(ret, 4.0f);
  inputs[1] = 2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
  TEST_FLOAT(ret, 5.0f);
  weights[0] = 2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
  TEST_FLOAT(ret, 7.0f);
  weights[1] = 2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
  TEST_FLOAT(ret, 9.0f);
  weights[1] = -2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
  TEST_FLOAT(ret, 1.0f);
  weights[0] = -2;

  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
  TEST_FLOAT(ret, 0.0f);

  bias = NAN;
  ret = feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
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
    layer.activation_type = NN_RELU;

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

bool init_xor_network(NNNetwork *network) {
  if (!init_network(network, 3, (unsigned[]){2, 2, 1}))
    return false;

  memcpy(network->layers[0].biases, (NNtype[2]){0, 0}, 2 * sizeof(NNtype));
  memcpy(network->layers[1].biases, (NNtype[1]){0}, 1 * sizeof(NNtype));

  memcpy(network->layers[0].weights, (NNtype[2 * 2]){-1, 1, 1, -1},
         2 * 2 * sizeof(NNtype));
  memcpy(network->layers[1].weights, (NNtype[2 * 1]){1, 1},
         2 * 1 * sizeof(NNtype));

  return true;
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

    init_xor_network(&network);

    NNtype xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    NNtype xor_expected[4] = {0, 1, 1, 0};

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

void test_trainer() {
  {
    NNNetwork network = {0};
    init_xor_network(&network);
    NNTrainer trainer = {0};
    init_trainer(&trainer, &network);

    NNtype xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    NNtype xor_expected[4][1] = {{0}, {1}, {1}, {0}};

    // Set one weight wrong
    network.layers[0].weights[0] = 1;

    calculate_batch_slope_nudge_version(&trainer, 4, &xor_inputs[0][0],
                                        &xor_expected[0][0]);

    TEST_FLOAT_TOL(trainer.layer_slopes[1][0], 2.5f, 1e-2f);
    TEST_FLOAT_TOL(trainer.layer_slopes[1][1], 0.5f, 1e-2f);
    TEST_FLOAT_TOL(trainer.layer_slopes[1][2], 1.5f, 1e-2f);

    free_network(&network);
    free_trainer(&trainer);
  }
  {
    NNNetwork network = {0};
    init_xor_network(&network);
    NNTrainer nudge_trainer = {0};
    init_trainer(&nudge_trainer, &network);
    NNTrainer trainer = {0};
    init_trainer(&trainer, &network);

    NNtype xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    NNtype xor_expected[4][1] = {{0}, {1}, {1}, {0}};

    // Set one weight wrong
    network.layers[0].weights[0] = 1;

    NNtype prev_batch_loss = calculate_batch_network_loss(
        trainer.network, 4, &xor_inputs[0][0], &xor_expected[0][0]);

    for (unsigned i = 0; i < 10; i++) {
      calculate_batch_slope_nudge_version(&nudge_trainer, 4, &xor_inputs[0][0],
                                          &xor_expected[0][0]);
      batch_backpropagation_step(&trainer, 4, &xor_inputs[0][0],
                                 &xor_expected[0][0]);

      for (unsigned slope_idx = 0; slope_idx < trainer.slopes_length;
           slope_idx++) {
        TEST_FLOAT_TOL(nudge_trainer.slopes[slope_idx],
                       trainer.slopes[slope_idx], 0.1f);
      }
      apply_slopes(&trainer, 0.1f);

      NNtype batch_loss = calculate_batch_network_loss(
          trainer.network, 4, &xor_inputs[0][0], &xor_expected[0][0]);

      TEST_BOOL(batch_loss < prev_batch_loss, true);
      prev_batch_loss = batch_loss;
    }

    free_network(&network);
    free_trainer(&nudge_trainer);
    free_trainer(&trainer);
  }
}

int main(void) {
  test_neuron();
  test_layer();
  test_network();
  test_trainer();

  // NNNetwork network = {0};
  // init_network(&network, 3, (unsigned[]){2, 4, 1});
  // NNTrainer trainer = {0};
  // init_trainer(&trainer, &network);

  // NNtype xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  // NNtype xor_expected[4][1] = {{0}, {1}, {1}, {0}};

  // for (unsigned test = 0; test < 40; test++) {
  //   srand(test);

  //   NNtype *network_value = network.memory_start;
  //   while (network_value < network.final_output +
  //   network.final_output_length) {
  //     *network_value = random_NNtype(-1, 1);

  //     network_value++;
  //   }

  //   for (unsigned i = 0; i < 200; i++) {
  //     calculate_batch_slope(&trainer, 4, &xor_inputs[0][0],
  //                           &xor_expected[0][0]);
  //     apply_slopes(&trainer, 0.1f);
  //   }

  //   printf("test %u final loss: %g\n", test,
  //          (double)calculate_batch_network_loss(&network, 4,
  //          &xor_inputs[0][0],
  //                                               &xor_expected[0][0]));
  // }

  // free_network(&network);
  // free_trainer(&trainer);
}
