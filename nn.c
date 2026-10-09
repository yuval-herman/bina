#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ANSI_RED "\033[31m"
#define ANSI_GREEN "\033[32m"
#define ANSI_YELLOW "\033[33m"
#define ANSI_MAGENTA "\033[95m"
#define ANSI_RESET "\033[0m"

#define ARR_LENGTH(arr) (sizeof(arr) / sizeof(arr[0]))

// Our float type, can be changed to double or whatever
typedef float NNtype;
typedef NNtype (*NNactivation)(NNtype);

typedef enum {
  NN_NO_ACTIVATION = 0,
  NN_RELU_ACTIVATION,
  NN_SIGMOID_ACTIVATION,
  NN_TANH_ACTIVATION,
  NN_ACTIVATION_COUNT__,
} NNActivationType;

typedef enum {
  NN_NO_INIT = 0,
  NN_UNIFORM_INIT,
  NN_HE_INIT,
  NN_XAVIER_INIT,
  NN_INIT_COUNT__,
} NNInitType;

typedef enum {
  NN_MSE_LOSS = 0,
  NN_CROSS_ENTROPY_LOSS,
  NN_LOSS_COUNT__,
} NNLossType;

typedef enum {
  NN_SDG_OPTIMIZER = 0,
  NN_MOMENTUM_OPTIMIZER,
  NN_ADAM_OPTIMIZER,
  NN_OPTIMIZER_COUNT__,
} NNOptimizerType;

typedef struct {
  NNtype *weights;
  unsigned input_length; // weights length are input * output
  NNtype *biases;
  NNtype *outputs;
  unsigned output_length; // biases and outputs have the same length
  NNActivationType activation_type;
  NNInitType init_type;
  NNtype init_gain;
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
  NNtype *velocities;
  unsigned velocities_length;
  NNtype *network_loss;
  unsigned network_loss_length;
  // This array containes pointers to the start of each layer's slope values.
  NNtype **layer_slopes;
  // This array containes pointers to the start of each layer's delta values.
  NNtype **layer_deltas;
  unsigned steps;
  NNLossType loss_type;
  NNOptimizerType optimizer_type;
  NNtype learning_rate;
  NNtype momentum_damp;
  NNtype adam_damp;
  NNtype p;  // used in adam as running power
  NNtype p2; // used in adam as running power
} NNTrainer;

// ================ PCG ================
// *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

typedef struct {
  uint64_t state;
  uint64_t inc;
} pcg32_random_t;

static inline uint32_t pcg32_random_r(pcg32_random_t *rng) {
  uint64_t oldstate = rng->state;
  // Advance internal state
  rng->state = oldstate * 6364136223846793005ULL + (rng->inc | 1);
  // Calculate output function (XSH RR), uses old state for max ILP
  uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
  uint32_t rot = (uint32_t)(oldstate >> 59u);
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

static inline void pcg32_srandom_r(pcg32_random_t *rng, uint64_t initstate,
                                   uint64_t initseq) {
  rng->state = 0U;
  rng->inc = (initseq << 1u) | 1u;
  pcg32_random_r(rng);
  rng->state += initstate;
  pcg32_random_r(rng);
}

static inline float pcg32_uniform_float(pcg32_random_t *rng) {
  return (float)(pcg32_random_r(rng) >> 8) * 0x1.0p-24f;
}

// Fast uniform float in [min, max)
static inline float pcg32_float_range(pcg32_random_t *rng, float min,
                                      float max) {
  float u = pcg32_uniform_float(rng);
  return min + u * (max - min);
}

static inline float pcg32_normal_float(pcg32_random_t *rng) {
  // Generate two uniform floats in (0, 1] to avoid logf(0)
  float u1 = 1.0f - pcg32_uniform_float(rng);
  float u2 = 1.0f - pcg32_uniform_float(rng);

  // Box-Muller transform
  float radius = sqrtf(-2.0f * logf(u1));
  float theta = 2.0f * (float)M_PI * u2;

  return radius * cosf(theta);
}

static inline unsigned pcg32_categorical_sample(pcg32_random_t *rng,
                                                unsigned length,
                                                const NNtype *probabilities) {
  float threshold = pcg32_uniform_float(rng);
  float cumulative_prob = 0.0f;

  for (unsigned i = 0; i < length; i++) {
    cumulative_prob += probabilities[i];
    if (threshold < cumulative_prob) {
      return i;
    }
  }

  // Fallback in case floating point imprecision causes the sum to be slightly
  // less than 1.0
  return length - 1;
}

// Generates a number between [0, range)
static inline uint32_t pcg32_bounded_rand(pcg32_random_t *rng, uint32_t range) {
  uint32_t x = pcg32_random_r(rng);
  uint64_t m = (uint64_t)x * (uint64_t)range;
  uint32_t l = (uint32_t)m;
  if (l < range) {
    uint32_t t = -range;
    if (t >= range) {
      t -= range;
      if (t >= range)
        t %= range;
    }
    while (l < t) {
      x = pcg32_random_r(rng);
      m = (uint64_t)x * (uint64_t)range;
      l = (uint32_t)m;
    }
  }
  return (uint32_t)(m >> 32);
}

// Generates a number between [min, max)
static inline uint32_t pcg32_int_range(pcg32_random_t *rng, uint32_t min,
                                       uint32_t max) {
  return pcg32_bounded_rand(rng, max - min) + min;
}

static inline void pcg32_shuffle_index_array(pcg32_random_t *rng,
                                             unsigned length,
                                             unsigned *index_array) {
  if (length < 2)
    return;

  for (unsigned idx = length; idx > 1; --idx) {
    unsigned swap_idx = pcg32_int_range(rng, 0, idx);
    unsigned temp = index_array[idx - 1];
    index_array[idx - 1] = index_array[swap_idx];
    index_array[swap_idx] = temp;
  }
}
// ================ PCG ================

static inline NNtype *get_weight(NNLayer *layer, unsigned output_idx,
                                 unsigned weight_idx) {
  return layer->weights + output_idx * layer->input_length + weight_idx;
}

typedef void (*NNInitLayer)(NNLayer *layer, pcg32_random_t *rng, NNtype gain);
typedef NNtype (*NNComputeLoss)(unsigned input_length, const NNtype *values,
                                const NNtype *expected);
// Loss derivative functions overwrite the `values` array
typedef void (*NNComputeLossDerivative)(unsigned input_length, NNtype *values,
                                        const NNtype *expected);
typedef void (*NNOptimizer)(NNTrainer *trainer, NNtype learning_rate,
                            unsigned batch_length);

static inline uint64_t get_time(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);

  return 1000 * 1000 * 1000 * (uint64_t)ts.tv_sec + (uint64_t)ts.tv_nsec;
}

static inline NNtype noop_activation(NNtype value) { return value; }

static inline NNtype noop_derivative(NNtype value) {
  (void)value;
  return 1;
}

static inline NNtype relu_activation(NNtype value) {
  // The check is made in such a way that NaN values would progagte through here
  // and not turn into zeros.
  return value <= 0 ? 0 : value;
}

static inline NNtype relu_derivative(NNtype value) {
  return value <= 0 ? 0 : 1;
}

static inline NNtype sigmoid_activation(NNtype value) {
  return 1.0f / (1.0f + expf(-value));
}

static inline NNtype sigmoid_derivative(NNtype value) {
  return value * (1.0f - value);
}

static inline NNtype tanh_activation(NNtype value) { return tanhf(value); }

static inline NNtype tanh_derivative(NNtype value) {
  return 1.0f - (value * value);
}

static char *activation_function_name(NNActivationType activation_type) {
  switch (activation_type) {
  default:
    assert(!"impossible activation type");
    return "impossible activation type";

  case NN_NO_ACTIVATION:
    return "no activation";
  case NN_RELU_ACTIVATION:
    return "relu";
  case NN_SIGMOID_ACTIVATION:
    return "sigmoid";
  case NN_TANH_ACTIVATION:
    return "tanh";
  }
}
static const struct {
  NNactivation activation;
  NNactivation derivative;
} activation_functions[] = {
    [NN_NO_ACTIVATION] = {.activation = noop_activation,
                          .derivative = noop_derivative},
    [NN_RELU_ACTIVATION] = {.activation = relu_activation,
                            .derivative = relu_derivative},
    [NN_SIGMOID_ACTIVATION] = {.activation = sigmoid_activation,
                               .derivative = sigmoid_derivative},
    [NN_TANH_ACTIVATION] = {.activation = tanh_activation,
                            .derivative = tanh_derivative},
};

static inline void noop_init(NNLayer *layer, pcg32_random_t *rng, NNtype gain) {
  (void)layer;
  (void)rng;
  (void)gain;
}

static inline void uniform_init(NNLayer *layer, pcg32_random_t *rng,
                                NNtype gain) {
  for (unsigned weight_idx = 0;
       weight_idx < layer->output_length * layer->input_length; weight_idx++) {
    layer->weights[weight_idx] = gain * pcg32_float_range(rng, -1, 1);
  }
  for (unsigned bias_idx = 0; bias_idx < layer->output_length; bias_idx++) {
    layer->biases[bias_idx] = gain * pcg32_float_range(rng, -1, 1);
  }
}

static inline void he_init(NNLayer *layer, pcg32_random_t *rng, NNtype gain) {
  const NNtype boundary = sqrtf(6.0f / (NNtype)layer->input_length);
  for (unsigned weight_idx = 0;
       weight_idx < layer->output_length * layer->input_length; weight_idx++) {
    layer->weights[weight_idx] =
        gain * pcg32_float_range(rng, -boundary, boundary);
  }
  for (unsigned bias_idx = 0; bias_idx < layer->output_length; bias_idx++) {
    layer->biases[bias_idx] = 0;
  }
}

static inline void xavier_init(NNLayer *layer, pcg32_random_t *rng,
                               NNtype gain) {
  const NNtype boundary = sqrtf(
      6.0f / ((NNtype)layer->input_length + (NNtype)layer->output_length));
  for (unsigned weight_idx = 0;
       weight_idx < layer->output_length * layer->input_length; weight_idx++) {
    layer->weights[weight_idx] =
        gain * pcg32_float_range(rng, -boundary, boundary);
  }
  for (unsigned bias_idx = 0; bias_idx < layer->output_length; bias_idx++) {
    layer->biases[bias_idx] = 0;
  }
}

static char *init_function_name(NNInitType init_type) {
  switch (init_type) {
  default:
    assert(!"impossible init type");
    return "impossible init type";

  case NN_NO_INIT:
    return "no init";
  case NN_UNIFORM_INIT:
    return "uniform";
  case NN_HE_INIT:
    return "he";
  case NN_XAVIER_INIT:
    return "xavier";
  }
}

static const NNInitLayer init_functions[] = {
    [NN_NO_INIT] = noop_init,
    [NN_UNIFORM_INIT] = uniform_init,
    [NN_HE_INIT] = he_init,
    [NN_XAVIER_INIT] = xavier_init,
};

// `values` is modified in place. If you need the values unaltered pass a copy
// here.
static inline void softmax(unsigned inputs_length, NNtype *values) {
  NNtype max = -INFINITY;
  for (unsigned i = 0; i < inputs_length; i++) {
    if (values[i] > max)
      max = values[i];
  }

  for (unsigned i = 0; i < inputs_length; i++) {
    values[i] = expf(values[i] - max);
  }

  NNtype sum = 0;
  for (unsigned i = 0; i < inputs_length; i++) {
    sum += values[i];
  }

  for (unsigned i = 0; i < inputs_length; i++) {
    values[i] = values[i] / sum;
  }
}

static inline NNtype mse_loss(unsigned inputs_length, const NNtype *values,
                              const NNtype *expected) {
  assert(inputs_length);
  assert(values);
  assert(expected);

  NNtype loss = 0;
  for (unsigned i = 0; i < inputs_length; i++) {
    const NNtype input_loss = values[i] - expected[i];
    loss += input_loss * input_loss;
  }

  return loss / (NNtype)inputs_length;
}

static inline void mse_loss_derivative(unsigned inputs_length, NNtype *values,
                                       const NNtype *expected) {
  assert(inputs_length);
  assert(values);
  assert(expected);

  for (unsigned i = 0; i < inputs_length; i++) {
    values[i] = (values[i] - expected[i]) * 2 / (NNtype)inputs_length;
  }
}

static inline NNtype cross_entropy_loss(unsigned inputs_length,
                                        const NNtype *values,
                                        const NNtype *expected) {
  assert(inputs_length);
  assert(values);
  assert(expected);

  // combine results with softmax
  NNtype max = -INFINITY;
  for (unsigned i = 0; i < inputs_length; i++) {
    if (values[i] > max)
      max = values[i];
  }

  NNtype sum = 0;
  for (unsigned i = 0; i < inputs_length; i++) {
    sum += expf(values[i] - max);
  }

  for (unsigned i = 0; i < inputs_length; i++) {
    if (expected[i] == 1) {
      return -(values[i] - max) + logf(sum);
    }
  }
  assert(
      !"cross entropy loss expects one element in the expected array to be 1");
  return NAN;
}

static inline void cross_entropy_loss_derivative(unsigned inputs_length,
                                                 NNtype *values,
                                                 const NNtype *expected) {
  assert(inputs_length);
  assert(values);
  assert(expected);

  softmax(inputs_length, values);

  for (unsigned i = 0; i < inputs_length; i++) {
    values[i] = values[i] - expected[i];
  }
}

static char *loss_function_name(NNLossType loss_type) {
  switch (loss_type) {
  default:
    assert(!"impossible loss type");
    return "impossible loss type";

  case NN_MSE_LOSS:
    return "mse";
  case NN_CROSS_ENTROPY_LOSS:
    return "cross entropy";
  }
}

static const struct {
  NNComputeLoss loss;
  NNComputeLossDerivative derivative;
} loss_functions[] = {
    [NN_MSE_LOSS] =
        {
            .loss = mse_loss,
            .derivative = mse_loss_derivative,
        },
    [NN_CROSS_ENTROPY_LOSS] =
        {
            .loss = cross_entropy_loss,
            .derivative = cross_entropy_loss_derivative,
        },
};

void SDG_optimizer(NNTrainer *trainer, NNtype learning_rate,
                   unsigned batch_length) {
  assert(batch_length);
  assert(trainer);
  assert(isfinite(learning_rate) && learning_rate != 0);
  NNNetwork *network = trainer->network;
  assert(network && "trainer must be initialised before training");

  learning_rate /= (NNtype)batch_length;

  for (unsigned layer_idx = 0; layer_idx < network->layers_length;
       layer_idx++) {
    const NNLayer layer = network->layers[layer_idx];
    const unsigned weights_length = layer.input_length * layer.output_length;

#define apply_slope(param, slope) (param -= slope * learning_rate)
    for (unsigned weight_idx = 0; weight_idx < weights_length; weight_idx++) {
      apply_slope(layer.weights[weight_idx],
                  trainer->layer_slopes[layer_idx][weight_idx]);
    }
    for (unsigned bias_idx = 0; bias_idx < layer.output_length; bias_idx++) {
      apply_slope(layer.biases[bias_idx],
                  trainer->layer_slopes[layer_idx][weights_length + bias_idx]);
    }
#undef apply_slope
  }
}

void momentum_optimizer(NNTrainer *trainer, NNtype learning_rate,
                        unsigned batch_length) {
  assert(batch_length);
  assert(trainer);
  assert(isfinite(learning_rate) && learning_rate != 0);
  NNNetwork *network = trainer->network;
  assert(network && "trainer must be initialised before training");

  learning_rate /= (NNtype)batch_length;

#define apply_slope(param, slope) (param -= slope * learning_rate)
#define update_momentum(momentum, slope)                                       \
  (momentum = trainer->momentum_damp * momentum + slope)

  for (unsigned layer_idx = 0; layer_idx < network->layers_length;
       layer_idx++) {
    const NNLayer layer = network->layers[layer_idx];
    const unsigned weights_length = layer.input_length * layer.output_length;

    for (unsigned weight_idx = 0; weight_idx < weights_length; weight_idx++) {
      const unsigned velocity_idx =
          (unsigned)(&trainer->layer_slopes[layer_idx][weight_idx] -
                     trainer->slopes);
      update_momentum(trainer->velocities[velocity_idx],
                      trainer->layer_slopes[layer_idx][weight_idx]);

      apply_slope(layer.weights[weight_idx], trainer->velocities[velocity_idx]);
    }

    for (unsigned bias_idx = 0; bias_idx < layer.output_length; bias_idx++) {
      const unsigned velocity_idx =
          (unsigned)(&trainer
                          ->layer_slopes[layer_idx][weights_length + bias_idx] -
                     trainer->slopes);
      update_momentum(
          trainer->velocities[velocity_idx],
          trainer->layer_slopes[layer_idx][weights_length + bias_idx]);

      apply_slope(layer.biases[bias_idx], trainer->velocities[velocity_idx]);
    }
  }
#undef apply_slope
#undef update_momentum
}

void adam_optimizer(NNTrainer *trainer, NNtype learning_rate,
                    unsigned batch_length) {
  (void)batch_length;

  assert(trainer);
  assert(isfinite(learning_rate) && learning_rate != 0);
  NNNetwork *network = trainer->network;
  assert(network && "trainer must be initialised before training");

#define apply_slope(param, momentum, velocity)                                 \
  (param -=                                                                    \
   learning_rate * (momentum / (p_div * sqrtf(velocity / p2_div) + 1e-8f)))
#define update_velocities(momentum, velocity, slope)                           \
  do {                                                                         \
    momentum = trainer->momentum_damp * (momentum - slope) + slope;            \
    velocity = trainer->adam_damp * velocity +                                 \
               (1 - trainer->adam_damp) * slope * slope;                       \
  } while (0)

  trainer->p *= trainer->momentum_damp;
  const NNtype p_div = 1 - trainer->p;
  trainer->p2 *= trainer->adam_damp;
  const NNtype p2_div = 1 - trainer->p2;

  for (unsigned layer_idx = 0; layer_idx < network->layers_length;
       layer_idx++) {
    const NNLayer layer = network->layers[layer_idx];
    const unsigned weights_length = layer.input_length * layer.output_length;

    for (unsigned weight_idx = 0; weight_idx < weights_length; weight_idx++) {
      const unsigned velocity_idx =
          (unsigned)(&trainer->layer_slopes[layer_idx][weight_idx] -
                     trainer->slopes);
      const unsigned adam_idx = trainer->velocities_length - velocity_idx - 1;
      update_velocities(trainer->velocities[velocity_idx],
                        trainer->velocities[adam_idx],
                        trainer->layer_slopes[layer_idx][weight_idx]);

      apply_slope(layer.weights[weight_idx], trainer->velocities[velocity_idx],
                  trainer->velocities[adam_idx]);
    }

    for (unsigned bias_idx = 0; bias_idx < layer.output_length; bias_idx++) {
      const unsigned velocity_idx =
          (unsigned)(&trainer
                          ->layer_slopes[layer_idx][weights_length + bias_idx] -
                     trainer->slopes);
      const unsigned adam_idx = trainer->velocities_length - velocity_idx - 1;
      update_velocities(
          trainer->velocities[velocity_idx], trainer->velocities[adam_idx],
          trainer->layer_slopes[layer_idx][weights_length + bias_idx]);

      apply_slope(layer.biases[bias_idx], trainer->velocities[velocity_idx],
                  trainer->velocities[adam_idx]);
    }
  }
#undef apply_slope
#undef update_momentum
}

static char *optimizer_function_name(NNOptimizerType optimizer_type) {
  switch (optimizer_type) {
  default:
    assert(!"impossible optimizer type");
    return "impossible optimizer type";

  case NN_SDG_OPTIMIZER:
    return "SDG";
  case NN_MOMENTUM_OPTIMIZER:
    return "momentum";
  case NN_ADAM_OPTIMIZER:
    return "adam";
  }
}

static const NNOptimizer optimizer_functions[] = {
    [NN_SDG_OPTIMIZER] = SDG_optimizer,
    [NN_MOMENTUM_OPTIMIZER] = momentum_optimizer,
    [NN_ADAM_OPTIMIZER] = adam_optimizer,
};

// layers_neurons is a count of how many outputs a layer has. The first items
// in that array is the amount of inputs the network may accept, then the rest
// of the numbers are the outputs for every sequential layer.
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

    layer->activation_type =
        i < layers_length - 2 ? NN_RELU_ACTIVATION : NN_NO_ACTIVATION;

    layer->init_type = NN_UNIFORM_INIT;
    layer->init_gain = 1.0f;
  }

  network->final_output = network->layers[network->layers_length - 1].outputs;
  network->final_output_length =
      network->layers[network->layers_length - 1].output_length;
  return true;
}

void init_layer_weights(NNLayer *layer, pcg32_random_t *rng) {
  init_functions[layer->init_type](layer, rng, layer->init_gain);
}

void init_network_weights(NNNetwork *network, pcg32_random_t *rng) {
  for (unsigned layer_idx = 0; layer_idx < network->layers_length;
       layer_idx++) {
    init_layer_weights(&network->layers[layer_idx], rng);
  }
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

bool init_trainer(NNTrainer *trainer, NNNetwork *network,
                  NNOptimizerType optimizer_type) {
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

  unsigned velocities_length = 0;
  switch (optimizer_type) {
  case NN_SDG_OPTIMIZER:
    break;
  case NN_MOMENTUM_OPTIMIZER:
    velocities_length = slopes_length;
    break;
  case NN_ADAM_OPTIMIZER:
    velocities_length = slopes_length * 2;
    break;

  case NN_OPTIMIZER_COUNT__:
  default:
    assert(!"Invalid optimizer type");
  }

  // calloc because we need to zero out velocities at first, and usually the OS
  // gives zeroed out memory anyways so its more efficient to use calloc for the
  // whole thing then memset just the velocities part.
  NNtype *slopes_memory =
      calloc(slopes_length + velocities_length + deltas_length, sizeof(NNtype));
  if (slopes_memory == NULL) {
    return false;
  }

  trainer->layer_slopes =
      malloc(network->layers_length * sizeof(*trainer->layer_slopes));

  if (trainer->layer_slopes == NULL) {
    free(slopes_memory);
    return false;
  }

  trainer->layer_deltas =
      malloc(network->layers_length * sizeof(*trainer->layer_deltas));

  if (trainer->layer_deltas == NULL) {
    free(slopes_memory);
    free(trainer->layer_slopes);
    return false;
  }

  trainer->network_loss =
      malloc(network->final_output_length * sizeof(*trainer->network_loss));

  if (trainer->network_loss == NULL) {
    free(slopes_memory);
    free(trainer->layer_slopes);
    free(trainer->layer_deltas);
    return false;
  }

  trainer->slopes = slopes_memory;
  trainer->deltas = trainer->slopes + slopes_length;
  trainer->velocities = optimizer_type == NN_SDG_OPTIMIZER
                            ? NULL
                            : trainer->deltas + deltas_length;

  trainer->slopes_length = slopes_length;
  trainer->deltas_length = deltas_length;
  trainer->velocities_length = velocities_length;

  trainer->network = network;
  trainer->network_loss_length = network->final_output_length;
  trainer->learning_rate = 0.01f;
  trainer->momentum_damp = 0.9f;
  trainer->adam_damp = 0.999f;

  trainer->layer_slopes[0] = trainer->slopes;
  trainer->layer_deltas[0] = trainer->deltas;

  for (unsigned i = 0; i < network->layers_length - 1; i++) {
    const NNLayer l = network->layers[i];
    const unsigned weights_length = l.input_length * l.output_length;
    const unsigned layer_size = weights_length + l.output_length;

    trainer->layer_slopes[i + 1] = trainer->layer_slopes[i] + layer_size;
    trainer->layer_deltas[i + 1] = trainer->layer_deltas[i] + l.output_length;
  }

  trainer->steps = 0;
  trainer->p = 1;
  trainer->p2 = 1;
  trainer->loss_type = NN_MSE_LOSS;
  trainer->optimizer_type = optimizer_type;

  return true;
}

void free_trainer(NNTrainer *trainer) {
  free(trainer->slopes);
  free(trainer->layer_slopes);
  free(trainer->layer_deltas);
  free(trainer->network_loss);
  *trainer = (NNTrainer){0};
}

// Check if two float are "close enough" to be considered equal with a dynamic
// tolerance level
static inline bool float_equals_tolerance(NNtype a, NNtype b,
                                          NNtype absolute_tolerance,
                                          NNtype relative_tolerance) {
  return fabsf(a - b) < fmaxf(absolute_tolerance,
                              relative_tolerance * fmaxf(fabsf(a), fabsf(b)));
}

NNtype feed_neuron(unsigned input_length, const NNtype *inputs,
                   const NNtype *weights, NNtype bias,
                   NNactivation activation) {
  assert(input_length > 0 && "input length of neuron can't be zero");
  assert(inputs != NULL);
  assert(weights != NULL);

  // The calculation are split and recombined to take advantage of auto
  // vectorization
#define DRAIN_COUNT 4
  NNtype output_drains[DRAIN_COUNT] = {0};

  const unsigned full_row_inputs = input_length - input_length % DRAIN_COUNT;
  for (unsigned i = 0; i < full_row_inputs;) {
    for (unsigned drain_idx = 0; drain_idx < DRAIN_COUNT; drain_idx++, i++) {
      output_drains[drain_idx] += inputs[i] * weights[i];
    }
  }

  {
    unsigned i = full_row_inputs;
    for (unsigned drain_idx = 0; i < input_length; drain_idx++, i++) {
      output_drains[drain_idx] += inputs[i] * weights[i];
    }
  }

  NNtype output = 0;
  for (unsigned i = 0; i < DRAIN_COUNT; i++) {
    output += output_drains[i];
  }
#undef DRAIN_COUNT

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

// `inputs` and `expected` are both 2d arrays that are expected to be in
// contiguous memory.
NNtype calculate_batch_network_loss(NNNetwork *network, NNLossType loss_type,
                                    unsigned inputs_length,
                                    const NNtype *inputs,
                                    const NNtype *expected) {
  double avg_loss = 0;
  for (unsigned i = 0; i < inputs_length; i++) {
    feed_network(network, &inputs[i * network->layers[0].input_length]);
    avg_loss += (double)loss_functions[loss_type].loss(
        network->final_output_length, network->final_output,
        &expected[i * network->final_output_length]);
  }
  return (NNtype)avg_loss / (NNtype)inputs_length;
}

// `inputs` and `expected` are both 2d arrays that are expected to be in
// contiguous memory.
void calculate_batch_slope_numeric(NNTrainer *trainer, unsigned inputs_length,
                                   const NNtype *inputs,
                                   const NNtype *expected) {
  assert(inputs_length && inputs);
  assert(expected);
  assert(trainer);
  NNNetwork *network = trainer->network;
  assert(network && "trainer must be initialised before training");
  const NNtype nudge_value = 1e-4f;

  double avg_loss_before = calculate_batch_network_loss(
      network, trainer->loss_type, inputs_length, inputs, expected);

  for (unsigned layer_idx = 0; layer_idx < network->layers_length;
       layer_idx++) {
    const NNLayer layer = network->layers[layer_idx];
    const unsigned weights_length = layer.input_length * layer.output_length;

    for (unsigned weight_idx = 0; weight_idx < weights_length; weight_idx++) {
      NNtype previous_nn_value = layer.weights[weight_idx];
      layer.weights[weight_idx] += nudge_value;

      double avg_loss_after = calculate_batch_network_loss(
          network, trainer->loss_type, inputs_length, inputs, expected);

      trainer->layer_slopes[layer_idx][weight_idx] =
          (NNtype)(avg_loss_after - avg_loss_before) / nudge_value;

      layer.weights[weight_idx] = previous_nn_value;
    }
    for (unsigned bias_idx = 0; bias_idx < layer.output_length; bias_idx++) {
      NNtype previous_nn_value = layer.biases[bias_idx];
      layer.biases[bias_idx] += nudge_value;

      double avg_loss_after = calculate_batch_network_loss(
          network, trainer->loss_type, inputs_length, inputs, expected);

      trainer->layer_slopes[layer_idx][weights_length + bias_idx] =
          (NNtype)(avg_loss_after - avg_loss_before) / nudge_value;

      layer.biases[bias_idx] = previous_nn_value;
    }
  }
}

const NNtype *calculate_output_gradient(NNTrainer *trainer,
                                        const NNtype *inputs,
                                        const NNtype *expected) {
  assert(inputs);
  assert(expected);
  assert(trainer);
  NNNetwork *network = trainer->network;
  assert(network && "trainer must be initialised before training");

  feed_network(network, inputs);

  assert(trainer->network_loss_length == network->final_output_length);
  memcpy(trainer->network_loss, network->final_output,
         trainer->network_loss_length * sizeof(*trainer->network_loss));

  loss_functions[trainer->loss_type].derivative(
      trainer->network_loss_length, trainer->network_loss, expected);

  NNtype *layer_deltas = trainer->layer_deltas[network->layers_length - 1];

  for (unsigned neuron_idx = 0; neuron_idx < network->final_output_length;
       neuron_idx++) {

    layer_deltas[neuron_idx] = trainer->network_loss[neuron_idx];
  }

  return layer_deltas;
}

// `output_delta` can be set to NULL if this function is called after
// `calculate_output_gradient`.
void calculate_layers_delta(NNTrainer *trainer, const NNtype *output_gradient) {
  assert(trainer);
  NNNetwork *network = trainer->network;
  assert(network && "trainer must be initialised before training");

  {
    if (output_gradient == NULL)
      output_gradient = trainer->layer_deltas[network->layers_length - 1];
    NNtype *layer_deltas = trainer->layer_deltas[network->layers_length - 1];
    const NNLayer *layer = &network->layers[network->layers_length - 1];

    for (unsigned neuron_idx = 0; neuron_idx < network->final_output_length;
         neuron_idx++) {
      layer_deltas[neuron_idx] =
          output_gradient[neuron_idx] *
          activation_functions[layer->activation_type].derivative(
              layer->outputs[neuron_idx]);
    }
  }
  for (NNtype *delta_p = trainer->deltas;
       delta_p < trainer->layer_deltas[network->layers_length - 1]; delta_p++) {
    *delta_p = 0;
  }

  for (unsigned layer_counter = 0; layer_counter < network->layers_length - 1;
       layer_counter++) {
    const unsigned layer_idx = network->layers_length - layer_counter - 2;
    const NNLayer *layer = &network->layers[layer_idx];
    NNtype *layer_deltas = trainer->layer_deltas[layer_idx];
    NNLayer *last_visited_layer = &network->layers[layer_idx + 1];
    NNtype *last_visited_layer_deltas = trainer->layer_deltas[layer_idx + 1];

    for (unsigned last_layer_neuron_idx = 0;
         last_layer_neuron_idx < last_visited_layer->output_length;
         last_layer_neuron_idx++) {
      for (unsigned neuron_idx = 0; neuron_idx < layer->output_length;
           neuron_idx++) {
        layer_deltas[neuron_idx] +=
            *get_weight(last_visited_layer, last_layer_neuron_idx, neuron_idx) *
            last_visited_layer_deltas[last_layer_neuron_idx];
      }
    }
    for (unsigned neuron_idx = 0; neuron_idx < layer->output_length;
         neuron_idx++) {
      layer_deltas[neuron_idx] *=
          activation_functions[layer->activation_type].derivative(
              layer->outputs[neuron_idx]);
    }
  }
}

// Adds to the network slopes, requires clearing before call for each batch.
void calculate_network_slopes(NNTrainer *trainer, const NNtype *input) {
  assert(input);
  assert(trainer);
  NNNetwork *network = trainer->network;
  assert(network && "trainer must be initialised before training");

  // first layer needs special handling because we don't have the input
  // saved within the layers

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

    for (unsigned neuron_idx = 0; neuron_idx < layer.output_length;
         neuron_idx++) {
      NNtype *weight_slopes = layer_slopes + layer.input_length * neuron_idx;
      for (unsigned input_idx = 0; input_idx < layer.input_length;
           input_idx++) {
        weight_slopes[input_idx] +=
            input[input_idx] * trainer->layer_deltas[layer_idx][neuron_idx];
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

    for (unsigned neuron_idx = 0; neuron_idx < layer.output_length;
         neuron_idx++) {
      NNtype *weight_slopes = layer_slopes + layer.input_length * neuron_idx;
      for (unsigned input_idx = 0; input_idx < layer.input_length;
           input_idx++) {
        weight_slopes[input_idx] +=
            preceding_layer.outputs[input_idx] *
            trainer->layer_deltas[layer_idx][neuron_idx];
      }
    }
  }
}

void apply_slopes(NNTrainer *trainer, NNtype learning_rate,
                  unsigned batch_length) {
  assert(trainer);
  assert(batch_length);
  assert(isfinite(learning_rate) && learning_rate != 0);

  trainer->steps++;

  optimizer_functions[trainer->optimizer_type](trainer, learning_rate,
                                               batch_length);
}

// Calculates and applies a single training step for a batch of inputs.
// `inputs` and `expected` are both 2d arrays that are expected to be in
// contiguous memory.
void train_step(NNTrainer *trainer, unsigned batch_length,
                unsigned *batch_indices, const NNtype *inputs,
                const NNtype *expected) {
  assert(batch_length && inputs);
  assert(expected);
  assert(trainer);
  NNNetwork *network = trainer->network;
  assert(network && "trainer must be initialised before training");

  memset(trainer->slopes, 0, trainer->slopes_length * sizeof(NNtype));

  for (unsigned input_batch_idx = 0; input_batch_idx < batch_length;
       input_batch_idx++) {
    const NNtype *input_batch = inputs + network->layers[0].input_length *
                                             batch_indices[input_batch_idx];

    calculate_output_gradient(trainer, input_batch,
                              &expected[network->final_output_length *
                                        batch_indices[input_batch_idx]]);

    calculate_layers_delta(trainer, NULL);

    calculate_network_slopes(trainer, input_batch);
  }

  apply_slopes(trainer, trainer->learning_rate, batch_length);
}

void train_epoch(NNTrainer *trainer, pcg32_random_t *rng,
                 unsigned indices_length, unsigned *indices,
                 unsigned batch_length, const NNtype *inputs,
                 const NNtype *expected) {
  assert(rng);
  assert(indices);
  assert(batch_length);
  assert(inputs);
  assert(expected);
  assert(trainer);

  pcg32_shuffle_index_array(rng, indices_length, indices);

  // We skip leftover examples if there are any
  for (unsigned batch_idx = 0; batch_idx < batch_length; batch_idx++) {
    train_step(trainer, batch_length, indices + batch_length * batch_idx,
               inputs, expected);
  }
}

static unsigned test_checks_passing = 0;
static unsigned test_check_count = 0;
static unsigned test_passing = 0;
static unsigned test_count = 0;

static bool test_block_latch_;
static unsigned test_block_failing_;

#define TEST_BLOCK(name, ...)                                                  \
  for (test_block_latch_ = (test_block_failing_ = 0, test_count++, true);      \
       test_block_latch_;                                                      \
       test_block_latch_ = false,                                              \
      test_block_failing_ == 0                                                 \
           ? (void)test_passing++                                              \
           : printf(ANSI_MAGENTA name                                          \
                    " test failed! (line %d)\n%u checks are failing"           \
                    "\n" ANSI_RESET,                                           \
                    ##__VA_ARGS__, __LINE__, test_block_failing_))

static bool test_step_latch_;
static unsigned test_step_failing_;

#define TEST_STEP(step_name, ...)                                              \
  for (test_step_latch_ = (test_step_failing_ = 0, true); test_step_latch_;    \
       test_step_latch_ = false,                                               \
      test_step_failing_ == 0                                                  \
           ? 0                                                                 \
           : printf(ANSI_YELLOW step_name                                      \
                    " test step failed! (line %d)\n%u checks are failing"      \
                    "\n" ANSI_RESET,                                           \
                    ##__VA_ARGS__, __LINE__, test_step_failing_))

#define CHECK_REGISTER(got, expected, pass, format)                            \
  do {                                                                         \
    test_check_count++;                                                        \
    if (pass)                                                                  \
      test_checks_passing++;                                                   \
    else {                                                                     \
      if (test_block_latch_)                                                   \
        test_block_failing_++;                                                 \
      if (test_step_latch_)                                                    \
        test_step_failing_++;                                                  \
      printf(ANSI_RED "Got incorrect value! (line %d) expected " format        \
                      " but got " format "\n" ANSI_RESET,                      \
             __LINE__, (expected), (got));                                     \
    }                                                                          \
  } while (0)

#define BOOL_STRING(value) ((value) ? "true" : "false")
#define CHECK_BOOL(got, expected)                                              \
  CHECK_REGISTER(BOOL_STRING(got), BOOL_STRING(expected), (got) == (expected), \
                 "%s")

#define CHECK_FLOAT(got, expected) CHECK_FLOAT_TOL(got, expected, 1e-6f, 1e-6f)
#define CHECK_FLOAT_TOL(got, expected, a_tolerance, r_tolerance)               \
  CHECK_REGISTER(                                                              \
      (double)got, (double)expected,                                           \
      (isnan(got) && isnan(expected)) ||                                       \
          float_equals_tolerance(got, expected, a_tolerance, r_tolerance),     \
      "%g")

#define CHECK_UINT(got, expected)                                              \
  CHECK_REGISTER((got), (expected), (got) == (expected), "%u")

void test_neuron() {
  NNtype inputs[] = {1, 1};
  NNtype weights[] = {1, 1};
  NNtype bias = 1;
  NNtype ret;

  TEST_BLOCK("single neuron") {
    ret =
        feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
    CHECK_FLOAT(ret, 3.0f);
    inputs[0] = 2;

    ret =
        feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
    CHECK_FLOAT(ret, 4.0f);
    inputs[1] = 2;

    ret =
        feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
    CHECK_FLOAT(ret, 5.0f);
    weights[0] = 2;

    ret =
        feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
    CHECK_FLOAT(ret, 7.0f);
    weights[1] = 2;

    ret =
        feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
    CHECK_FLOAT(ret, 9.0f);
    weights[1] = -2;

    ret =
        feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
    CHECK_FLOAT(ret, 1.0f);
    weights[0] = -2;

    ret =
        feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
    CHECK_FLOAT(ret, 0.0f);
  }

  TEST_BLOCK("neuron NaN") {
    bias = NAN;
    ret =
        feed_neuron(ARR_LENGTH(inputs), inputs, weights, bias, relu_activation);
    // should be 0 but just because of activation
    // function. Otherwise we had 2*-2+2*-2+1 = -7
    CHECK_FLOAT(ret, NAN);
  }
}

void test_layer() {
  TEST_BLOCK("2x3 layer with 1 weights") {
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
    CHECK_FLOAT(layer.outputs[0], 3.0f);
    CHECK_FLOAT(layer.outputs[1], 3.0f);
    CHECK_FLOAT(layer.outputs[2], 3.0f);

    for (unsigned i = 0; i < layer.output_length * layer.input_length; i++) {
      layer.weights[i] = 2;
    }

    feed_layer(&layer, inputs);
    CHECK_FLOAT(layer.outputs[0], 5.0f);
    CHECK_FLOAT(layer.outputs[1], 5.0f);
    CHECK_FLOAT(layer.outputs[2], 5.0f);

    for (unsigned i = 0; i < layer.output_length; i++) {
      layer.biases[i] = 2;
    }

    feed_layer(&layer, inputs);
    CHECK_FLOAT(layer.outputs[0], 6.0f);
    CHECK_FLOAT(layer.outputs[1], 6.0f);
    CHECK_FLOAT(layer.outputs[2], 6.0f);

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
    CHECK_FLOAT(layer.outputs[0], 4.0f);
    CHECK_FLOAT(layer.outputs[1], 5.0f);
    CHECK_FLOAT(layer.outputs[2], 6.0f);

    free_network(&network);
  }
  TEST_BLOCK("3x3 layer with 1 weights") {
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
    CHECK_FLOAT(layer.outputs[0], 4.0f);
    CHECK_FLOAT(layer.outputs[1], 4.0f);
    CHECK_FLOAT(layer.outputs[2], 4.0f);

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
    CHECK_FLOAT(layer.outputs[0], 8.0f);
    CHECK_FLOAT(layer.outputs[1], 16.0f);
    CHECK_FLOAT(layer.outputs[2], 24.0f);

    free_network(&network);
  }
  TEST_BLOCK("3x2 layer with 1 weights") {
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
    CHECK_FLOAT(layer.outputs[0], 4.0f);
    CHECK_FLOAT(layer.outputs[1], 4.0f);

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
    CHECK_FLOAT(layer.outputs[0], 7.0f);
    CHECK_FLOAT(layer.outputs[1], 15.0f);

    free_network(&network);
  }
  TEST_BLOCK("3x2 layer with 1 weights") {
    NNNetwork network = {0};
    init_network(&network, 2, (unsigned[]){3, 2});
    NNLayer layer = network.layers[0];
    layer.activation_type = NN_RELU_ACTIVATION;

    for (unsigned i = 0; i < layer.output_length * layer.input_length; i++) {
      layer.weights[i] = -1;
    }

    for (unsigned i = 0; i < layer.output_length; i++) {
      layer.biases[i] = 1;
    }

    NNtype inputs[] = {1, 1, 1};
    assert(ARR_LENGTH(inputs) == layer.input_length);

    feed_layer(&layer, inputs);
    CHECK_FLOAT(layer.outputs[0], 0.0f);
    CHECK_FLOAT(layer.outputs[1], 0.0f);

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
    CHECK_FLOAT(layer.outputs[0], 2.5f);
    CHECK_FLOAT(layer.outputs[1], 0.0f);

    free_network(&network);
  }
  TEST_BLOCK("random layer sizes") {
    for (unsigned input_size = 1; input_size < 50; input_size++) {
      for (unsigned output_size = 1; output_size < 50; output_size++) {
        TEST_STEP("input = %u output = %u", input_size, output_size) {
          NNNetwork network = {0};
          init_network(&network, 2, (unsigned[]){input_size, output_size});
          NNLayer layer = network.layers[0];

          pcg32_random_t rng;
          pcg32_srandom_r(&rng, 0, 0);

          init_network_weights(&network, &rng);

          NNtype *inputs = malloc(sizeof inputs[0] * input_size);

          for (unsigned i = 0; i < input_size; i++) {
            inputs[i] = pcg32_float_range(&rng, -1, 1);
          }

          feed_layer(&layer, inputs);

          // Simple vector multiplication for test

          for (unsigned output_idx = 0; output_idx < output_size;
               output_idx++) {
            // Using double for higer prescision while testing
            double output = 0;
            for (unsigned input_idx = 0; input_idx < input_size; input_idx++) {
              output +=
                  (double)inputs[input_idx] *
                  (double)layer.weights[input_idx + input_size * output_idx];
            }
            output =
                (double)activation_functions[layer.activation_type].activation(
                    (NNtype)output + layer.biases[output_idx]);

            CHECK_FLOAT_TOL(layer.outputs[output_idx], (NNtype)output, 1e-5f,
                            1e-4f);
          }

          free_network(&network);
          free(inputs);
        }
      }
    }
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
  TEST_BLOCK("simple network") {
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

    CHECK_FLOAT(network.final_output[0], 13.0f);

    free_network(&network);
  }
  TEST_BLOCK("xor network") {
    NNNetwork network = {0};

    init_xor_network(&network);

    NNtype xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    NNtype xor_expected[4] = {0, 1, 1, 0};

    for (unsigned i = 0; i < ARR_LENGTH(xor_inputs); i++) {
      feed_network(&network, xor_inputs[i]);

      CHECK_FLOAT(network.final_output[0], xor_expected[i]);

      NNtype loss = mse_loss(1, network.final_output, &xor_expected[i]);

      CHECK_FLOAT(loss, 0.0f);
    }

    network.layers[0].weights[0] = 1;

    NNtype expected_loss[4] = {0, 0, 1, 4};
    for (unsigned i = 0; i < ARR_LENGTH(xor_inputs); i++) {
      feed_network(&network, xor_inputs[i]);

      NNtype loss = mse_loss(1, network.final_output, &xor_expected[i]);

      CHECK_FLOAT(loss, expected_loss[i]);
    }

    free_network(&network);
  }
}

void compare_numeric_backprop_slopes(NNNetwork *network, NNLossType loss_type,
                                     NNOptimizerType optimizer_type) {
  NNTrainer numeric_trainer = {0};
  init_trainer(&numeric_trainer, network, optimizer_type);
  NNTrainer trainer = {0};
  init_trainer(&trainer, network, optimizer_type);

  numeric_trainer.loss_type = loss_type;
  trainer.loss_type = loss_type;

  NNtype inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  NNtype expected[4][2] = {{1, 0}, {0, 1}, {0, 1}, {0, 1}};

  calculate_batch_slope_numeric(&numeric_trainer, 4, &inputs[0][0],
                                &expected[0][0]);
  train_step(&trainer, 4, (unsigned[]){0, 1, 2, 3}, &inputs[0][0],
             &expected[0][0]);

  for (unsigned slope_idx = 0; slope_idx < trainer.slopes_length; slope_idx++) {
    // backpropagation only divides slopes by batch length when applying them,
    // so we do it here instead
    CHECK_FLOAT_TOL(trainer.slopes[slope_idx] / 4,
                    numeric_trainer.slopes[slope_idx], 0.003f, 1e-2f);
  }

  for (unsigned output_idx = 0; output_idx < network->final_output_length;
       output_idx++) {
    CHECK_BOOL(isfinite(network->final_output[output_idx]), true);
  }

  free_trainer(&trainer);
  free_trainer(&numeric_trainer);
}

void test_trainer() {
  TEST_BLOCK("xor numeric trainer") {
    NNNetwork network = {0};
    init_xor_network(&network);
    NNTrainer trainer = {0};
    init_trainer(&trainer, &network, NN_MOMENTUM_OPTIMIZER);

    NNtype xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    NNtype xor_expected[4][1] = {{0}, {1}, {1}, {0}};

    // Set one weight wrong
    network.layers[0].weights[0] = 1;

    calculate_batch_slope_numeric(&trainer, 4, &xor_inputs[0][0],
                                  &xor_expected[0][0]);

    CHECK_FLOAT_TOL(trainer.layer_slopes[1][0], 2.5f, 1e-6f, 1e-2f);
    CHECK_FLOAT_TOL(trainer.layer_slopes[1][1], 0.5f, 1e-6f, 1e-2f);
    CHECK_FLOAT_TOL(trainer.layer_slopes[1][2], 1.5f, 1e-6f, 1e-2f);

    free_network(&network);
    free_trainer(&trainer);
  }
  TEST_BLOCK("xor train manually") {
    NNNetwork network = {0};
    init_xor_network(&network);
    NNTrainer trainer = {0};
    init_trainer(&trainer, &network, NN_MOMENTUM_OPTIMIZER);

    NNtype xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    NNtype xor_expected[4][1] = {{0}, {1}, {1}, {0}};
    NNtype last_layer_deltas[4] = {0, 0, 2, 4};

    // Set one weight wrong
    network.layers[0].weights[0] = 1;

    memset(trainer.slopes, 0, trainer.slopes_length * sizeof(NNtype));

    for (unsigned input_idx = 0; input_idx < ARR_LENGTH(xor_inputs);
         input_idx++) {
      calculate_output_gradient(&trainer, xor_inputs[input_idx],
                                xor_expected[input_idx]);

      calculate_layers_delta(&trainer, &last_layer_deltas[input_idx]);

      calculate_network_slopes(&trainer, xor_inputs[input_idx]);
    }

    CHECK_FLOAT_TOL(trainer.layer_slopes[0][0], 1.5f * 4, 1e-6f, 1e-2f);
    CHECK_FLOAT_TOL(trainer.layer_slopes[0][1], 1.0f * 4, 1e-6f, 1e-2f);
    CHECK_FLOAT_TOL(trainer.layer_slopes[0][2], 0.5f * 4, 1e-6f, 1e-2f);

    CHECK_FLOAT_TOL(trainer.layer_slopes[0][3], 0.0f * 4, 1e-6f, 1e-2f);
    CHECK_FLOAT_TOL(trainer.layer_slopes[0][4], 1.5f * 4, 1e-6f, 1e-2f);
    CHECK_FLOAT_TOL(trainer.layer_slopes[0][5], 0.5f * 4, 1e-6f, 1e-2f);

    CHECK_FLOAT_TOL(trainer.layer_slopes[1][0], 2.5f * 4, 1e-6f, 1e-2f);
    CHECK_FLOAT_TOL(trainer.layer_slopes[1][1], 0.5f * 4, 1e-6f, 1e-2f);
    CHECK_FLOAT_TOL(trainer.layer_slopes[1][2], 1.5f * 4, 1e-6f, 1e-2f);

    free_network(&network);
    free_trainer(&trainer);
  }
  TEST_BLOCK(
      "random activation, init, loss, and optimizer functions in trainer") {
    NNNetwork network = {0};
    init_network(&network, 4, (unsigned[]){2, 2, 2, 2});

    for (NNInitType init_type = NN_NO_INIT + 1; init_type < NN_INIT_COUNT__;
         init_type++) {
      for (NNActivationType activation_type = NN_NO_ACTIVATION;
           activation_type < NN_ACTIVATION_COUNT__; activation_type++) {
        for (NNLossType loss_type = NN_MSE_LOSS; loss_type < NN_LOSS_COUNT__;
             loss_type++) {
          for (NNOptimizerType optimizer_type = NN_SDG_OPTIMIZER;
               optimizer_type < NN_OPTIMIZER_COUNT__; optimizer_type++) {
            TEST_STEP("activation = %s init = %s loss = %s optimizer = %s",
                      activation_function_name(activation_type),
                      init_function_name(init_type),
                      loss_function_name(loss_type),
                      optimizer_function_name(optimizer_type)) {

              for (unsigned layer_idx = 0; layer_idx < network.layers_length;
                   layer_idx++) {
                network.layers[layer_idx].init_type = init_type;
                network.layers[layer_idx].activation_type = activation_type;
              }

              pcg32_random_t rng;
              pcg32_srandom_r(&rng, 0, 0);

              init_network_weights(&network, &rng);

              // initializing the biases seperately is required because he init
              // zeros them out which causes the numeric approach to give
              // different results with a (0,0) input.
              for (unsigned layer_idx = 0; layer_idx < network.layers_length;
                   layer_idx++) {
                for (unsigned bias_idx = 0;
                     bias_idx < network.layers[layer_idx].output_length;
                     bias_idx++) {
                  network.layers[layer_idx].biases[bias_idx] =
                      pcg32_float_range(&rng, 0.01f, 0.1f);
                }
              }

              compare_numeric_backprop_slopes(&network, loss_type,
                                              optimizer_type);
            }
          }
        }
      }
    }

    free_network(&network);
  }
  TEST_BLOCK("numeric oracle trainer") {
    NNNetwork network = {0};
    init_network(&network, 3, (unsigned[]){2, 2, 1});
    for (unsigned layer_idx = 0; layer_idx < network.layers_length - 1;
         layer_idx++) {
      network.layers[layer_idx].init_type = NN_HE_INIT;
    }

    pcg32_random_t rng;
    pcg32_srandom_r(&rng, 0, 0);

    init_network_weights(&network, &rng);

    // initializing the biases seperately is required because he init zeros
    // them out which causes the numeric approach to give different results
    // with a (0,0) input.
    for (unsigned layer_idx = 0; layer_idx < network.layers_length;
         layer_idx++) {
      for (unsigned bias_idx = 0;
           bias_idx < network.layers[layer_idx].output_length; bias_idx++) {
        network.layers[layer_idx].biases[bias_idx] =
            pcg32_float_range(&rng, 0.01f, 0.1f);
      }
    }

    NNTrainer numeric_trainer = {0};
    init_trainer(&numeric_trainer, &network, NN_MOMENTUM_OPTIMIZER);
    NNTrainer trainer = {0};
    init_trainer(&trainer, &network, NN_MOMENTUM_OPTIMIZER);

    NNtype xor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    NNtype xor_expected[4][1] = {{0}, {1}, {1}, {0}};

    NNtype prev_batch_loss =
        calculate_batch_network_loss(trainer.network, trainer.loss_type, 4,
                                     &xor_inputs[0][0], &xor_expected[0][0]);

    for (unsigned i = 0; i < 10; i++) {
      calculate_batch_slope_numeric(&numeric_trainer, 4, &xor_inputs[0][0],
                                    &xor_expected[0][0]);
      train_step(&trainer, 4, (unsigned[]){0, 1, 2, 3}, &xor_inputs[0][0],
                 &xor_expected[0][0]);

      for (unsigned slope_idx = 0; slope_idx < trainer.slopes_length;
           slope_idx++) {
        CHECK_FLOAT_TOL(numeric_trainer.slopes[slope_idx],
                        trainer.slopes[slope_idx] / 4, 1e-3f, 0.01f);
      }

      NNtype batch_loss =
          calculate_batch_network_loss(trainer.network, trainer.loss_type, 4,
                                       &xor_inputs[0][0], &xor_expected[0][0]);

      CHECK_BOOL(batch_loss < prev_batch_loss, true);
      prev_batch_loss = batch_loss;
    }

    free_network(&network);
    free_trainer(&numeric_trainer);
    free_trainer(&trainer);
  }
  {
    NNNetwork network = {0};
    init_network(&network, 3, (unsigned[]){1, 1, 1});
    NNTrainer trainer = {0};
    init_trainer(&trainer, &network, NN_ADAM_OPTIMIZER);

#define INIT_PARAMS()                                                          \
  for (unsigned layer_idx = 0; layer_idx < network.layers_length;              \
       layer_idx++) {                                                          \
    NNLayer layer = network.layers[layer_idx];                                 \
    for (unsigned weight_idx = 0;                                              \
         weight_idx < layer.output_length * layer.input_length;                \
         weight_idx++) {                                                       \
      layer.weights[weight_idx] = 1;                                           \
    }                                                                          \
    for (unsigned bias_idx = 0; bias_idx < layer.output_length; bias_idx++) {  \
      layer.biases[bias_idx] = 0;                                              \
    }                                                                          \
  }                                                                            \
  for (unsigned velocity_idx = 0; velocity_idx < trainer.velocities_length;    \
       velocity_idx++) {                                                       \
    trainer.velocities[velocity_idx] = 0.0f;                                   \
  }                                                                            \
  trainer.p = 1.0f;                                                            \
  trainer.p2 = 1.0f;

    TEST_STEP("same slope every step (0.5f)") {
      INIT_PARAMS();
      for (unsigned slope_idx = 0; slope_idx < trainer.slopes_length;
           slope_idx++) {
        trainer.slopes[slope_idx] = 0.5f;
      }

      adam_optimizer(&trainer, 0.01f, 1);
      CHECK_FLOAT(network.layers[0].weights[0], 0.99f);
      adam_optimizer(&trainer, 0.01f, 1);
      CHECK_FLOAT(network.layers[0].weights[0], 0.98f);
      adam_optimizer(&trainer, 0.01f, 1);
      CHECK_FLOAT(network.layers[0].weights[0], 0.97f);
    }

    TEST_STEP("slope 0, twice") {
      INIT_PARAMS();
      for (unsigned slope_idx = 0; slope_idx < trainer.slopes_length;
           slope_idx++) {
        trainer.slopes[slope_idx] = 0.0f;
      }
      for (unsigned velocity_idx = 0; velocity_idx < trainer.velocities_length;
           velocity_idx++) {
        trainer.velocities[velocity_idx] = 0.0f;
      }

      adam_optimizer(&trainer, 0.01f, 1);
      CHECK_FLOAT(network.layers[0].weights[0], 1.0f);
      adam_optimizer(&trainer, 0.01f, 1);
      CHECK_FLOAT(network.layers[0].weights[0], 1.0f);
    }

    TEST_STEP("slope 1, then 0") {
      INIT_PARAMS();
      for (unsigned slope_idx = 0; slope_idx < trainer.slopes_length;
           slope_idx++) {
        trainer.slopes[slope_idx] = 1.0f;
      }

      adam_optimizer(&trainer, 0.01f, 1);
      CHECK_FLOAT(network.layers[0].weights[0], 0.99f);

      for (unsigned slope_idx = 0; slope_idx < trainer.slopes_length;
           slope_idx++) {
        trainer.slopes[slope_idx] = 0.0f;
      }

      adam_optimizer(&trainer, 0.01f, 1);
      CHECK_FLOAT(network.layers[0].weights[0], 0.98329942f);
    }
    free_network(&network);
    free_trainer(&trainer);
#undef INIT_PARAMS
  }
}

void test_loss() {
  TEST_BLOCK("softmax") {
    NNtype inputs[3];

    memcpy(inputs, (NNtype[ARR_LENGTH(inputs)]){1, 2, 3},
           sizeof(*inputs) * ARR_LENGTH(inputs));

    softmax(ARR_LENGTH(inputs), inputs);
    CHECK_FLOAT(inputs[0], 0.0900306f);
    CHECK_FLOAT(inputs[1], 0.244728f);
    CHECK_FLOAT(inputs[2], 0.665241f);

    memcpy(inputs, (NNtype[ARR_LENGTH(inputs)]){1001, 1002, 1003},
           sizeof(*inputs) * ARR_LENGTH(inputs));

    softmax(ARR_LENGTH(inputs), inputs);
    CHECK_FLOAT(inputs[0], 0.0900306f);
    CHECK_FLOAT(inputs[1], 0.244728f);
    CHECK_FLOAT(inputs[2], 0.665241f);

    memcpy(inputs, (NNtype[ARR_LENGTH(inputs)]){1e4, 0, -1e4},
           sizeof(*inputs) * ARR_LENGTH(inputs));

    softmax(ARR_LENGTH(inputs), inputs);
    CHECK_FLOAT(inputs[0], 1.0f);
    CHECK_FLOAT(inputs[1], 0.0f);
    CHECK_FLOAT(inputs[2], 0.0f);
  }
  // TODO
  // TEST_BLOCK("cross entropy") {}
}

bool test_summary() {
  if (test_check_count > test_checks_passing) {
    const unsigned failing_tests = test_count - test_passing;
    printf("\n\nran %u tests, with %u (%.3g%%) failing.\n", test_count,
           failing_tests, ((double)failing_tests / test_count) * 100);
    const unsigned failing_test_checks = test_check_count - test_checks_passing;
    printf("ran %u checks, with %u (%.3g%%) failing.\n", test_check_count,
           failing_test_checks,
           ((double)failing_test_checks / test_check_count) * 100);
    return false;
  } else {
    printf("ran %u tests, with no fails!\n", test_count);
    return true;
  }
}

void bench_and_print() {
// count of all batches
#define BATCHS_COUNT 10
// length of a single batch
#define BATCH_LENGTH 10
// length of a single input
#define INPUT_LENGTH 32
#define EXPECTED_LENGTH 10

  NNtype inputs[BATCHS_COUNT * BATCH_LENGTH][INPUT_LENGTH];
  NNtype expected[BATCHS_COUNT * BATCH_LENGTH][EXPECTED_LENGTH];
  unsigned indices[BATCHS_COUNT * BATCH_LENGTH];

  for (unsigned i = 0; i < BATCHS_COUNT * BATCH_LENGTH; i++) {
    indices[i] = i;
  }

  pcg32_random_t rng;
  pcg32_srandom_r(&rng, 0, 0);

  for (unsigned batch_idx = 0; batch_idx < BATCHS_COUNT; batch_idx++) {
    for (unsigned input_batch_idx = 0; input_batch_idx < BATCH_LENGTH;
         input_batch_idx++) {
      for (unsigned input_idx = 0; input_idx < INPUT_LENGTH; input_idx++) {
        inputs[batch_idx * input_batch_idx][input_idx] =
            pcg32_float_range(&rng, -1, 1);
      }
      for (unsigned expected_idx = 0; expected_idx < EXPECTED_LENGTH;
           expected_idx++) {
        expected[batch_idx * input_batch_idx][expected_idx] =
            pcg32_float_range(&rng, -1, 1);
      }
    }
  }

  NNNetwork network = {0};
  unsigned network_layers[] = {INPUT_LENGTH, 32, 32, EXPECTED_LENGTH};
  init_network(&network, ARR_LENGTH(network_layers), network_layers);
  NNTrainer trainer = {0};
  init_trainer(&trainer, &network, NN_ADAM_OPTIMIZER);
  trainer.learning_rate = 1e-4f;

  init_network_weights(&network, &rng);

  uint64_t train_step_min_time = UINT64_MAX;
  uint64_t feed_net_min_time = UINT64_MAX;

  double train_step_average_time = 0;
  double feed_net_average_time = 0;

  unsigned warmup_runs = 1000;
  const unsigned runs = 10000;

  for (unsigned run = 0; warmup_runs || run < runs;) {
    uint64_t train_step_time = get_time();
    train_epoch(&trainer, &rng, BATCHS_COUNT * BATCH_LENGTH, indices,
                BATCH_LENGTH, &inputs[0][0], &expected[0][0]);
    train_step_time = get_time() - train_step_time;

    uint64_t feed_net_time = get_time();
    feed_network(&network, &inputs[0][0]);
    feed_net_time = get_time() - feed_net_time;

    if (train_step_min_time > train_step_time)
      train_step_min_time = train_step_time;
    if (feed_net_min_time > feed_net_time)
      feed_net_min_time = feed_net_time;

    if (!warmup_runs) {
      train_step_average_time += (double)train_step_time;
      feed_net_average_time += (double)feed_net_time;
    }
    if (warmup_runs)
      warmup_runs--;
    else
      run++;
  }

  train_step_average_time /= runs;
  feed_net_average_time /= runs;

  printf("train_step => min: %lu avg %g\n", train_step_min_time,
         train_step_average_time);
  printf("feed_net => min: %lu avg %g\n", feed_net_min_time,
         feed_net_average_time);
  printf("end loss => %g\n",
         (double)calculate_batch_network_loss(&network, trainer.loss_type,
                                              BATCH_LENGTH * BATCHS_COUNT,
                                              &inputs[0][0], &expected[0][0]));

  free_network(&network);
  free_trainer(&trainer);
#undef BATCH_LENGTH
#undef INPUT_LENGTH
}

void test_helpers() {
  TEST_BLOCK("shuffle") {
#define INDICES_LENGTH 5
    unsigned indices[INDICES_LENGTH];
    unsigned indices_appearance[INDICES_LENGTH] = {0};
    pcg32_random_t rng;
    pcg32_srandom_r(&rng, 0, 0);

    TEST_STEP("indices appear once") {
      for (unsigned i = 0; i < INDICES_LENGTH; i++) {
        indices[i] = i;
      }

      pcg32_shuffle_index_array(&rng, INDICES_LENGTH, indices);

      for (unsigned i = 0; i < INDICES_LENGTH; i++) {
        indices_appearance[indices[i]]++;
      }
      for (unsigned i = 0; i < INDICES_LENGTH; i++) {
        CHECK_UINT(indices_appearance[i], 1);
      }
    }

    TEST_STEP("indices changed order") {
      const unsigned step_count = 100;
      float unchange_avg = 0;
      for (unsigned step = 0; step < step_count; step++) {
        unsigned unchanged_count = INDICES_LENGTH;
        for (unsigned i = 0; i < INDICES_LENGTH; i++) {
          indices[i] = i;
        }

        pcg32_shuffle_index_array(&rng, INDICES_LENGTH, indices);
        for (unsigned i = 0; i < INDICES_LENGTH; i++) {
          if (indices[i] != i)
            unchanged_count--;
        }
        unchange_avg += (float)unchanged_count;
      }
      unchange_avg /= (float)step_count;
      CHECK_FLOAT_TOL(unchange_avg, 1.0f, 5 / sqrtf((float)(step_count)), 0);
    }
#undef INDICES_LENGTH
  }
}

#ifdef NN_MAIN
int main(void) {
#ifndef NDEBUG
  test_neuron();
  test_layer();
  test_network();
  test_loss();
  test_trainer();
  test_helpers();
  if (!test_summary())
    return 1;
#else
  bench_and_print();
#endif
}
#endif
