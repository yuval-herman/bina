#include "nn.c"

NNtype word_reward(pcg32_random_t *rng, NNtype action) {
  assert(action == 0 || action == 1);
  float n = pcg32_uniform_float(rng);
  if (action == 0) {
    return n < 0.3f ? 1 : -1;
  } else {
    return n < 0.7f ? 1 : -1;
  }
}

int main(void) {
  pcg32_random_t rng;
  pcg32_srandom_r(&rng, 0, 0);

  NNNetwork network = {0};
  const unsigned layers[] = {1, 2};
  init_network(&network, ARR_LENGTH(layers), layers);
  network.layers[0].activation_type = NN_NO_ACTIVATION;
  init_network_weights(&network, &rng);

  NNTrainer trainer = {0};
  init_trainer(&trainer, &network, NN_ADAM_OPTIMIZER);
  trainer.learning_rate = 0.01f;
  trainer.loss_type = NN_CROSS_ENTROPY_LOSS;

  NNtype inputs[] = {1};
  assert(ARR_LENGTH(inputs) == network.layers[0].input_length);

  unsigned steps = 5000;
  for (unsigned try = 0; try < steps; try++) {
    // feed
    feed_network(&network, (const NNtype[]){1});

    // sample
    softmax(network.final_output_length, network.final_output);
    memcpy(trainer.network_loss, network.final_output,
           trainer.network_loss_length * sizeof(*trainer.network_loss));

    unsigned idx = pcg32_categorical_sample(&rng, network.final_output_length,
                                            network.final_output);

    // reward
    NNtype reward = word_reward(&rng, (NNtype)idx);

    // gradient for the action taken
    const NNtype action_grad = reward * (network.final_output[idx] - 1);

    // gradient
    for (unsigned i = 0; i < network.final_output_length; i++) {
      network.final_output[i] = reward * network.final_output[i];
    }

    network.final_output[idx] = action_grad;

    memset(trainer.slopes, 0, trainer.slopes_length * sizeof(NNtype));

    calculate_layers_delta(&trainer, network.final_output);

    calculate_network_slopes(&trainer, inputs);

    apply_slopes(&trainer, trainer.learning_rate, 1);

    if (try % 500 == 0) {
      printf("p1 is %g\n\n", (double)trainer.network_loss[1]);
    }
  }

  free_trainer(&trainer);
  free_network(&network);
  return 0;
}
