#include "nn.c"

static void calculate_returns(NNtype discount, unsigned length,
                              const NNtype *rewards, NNtype *returns) {
  assert(length);
  assert(rewards);
  assert(returns);

  NNtype running_sum = 0;
  for (unsigned i = length; i > 0; i--) {
    running_sum *= discount;
    returns[i - 1] = running_sum += rewards[i - 1];
  }
}

// Clears and fills a one-hot array with 1 at the action index.
// For convenience, this function returns the pointer it received.
static inline NNtype *onehot(unsigned action, unsigned length,
                             NNtype *onehot_arr) {
  assert(length && onehot_arr);
  assert(action < length);

  memset(onehot_arr, 0, sizeof *onehot_arr * length);
  onehot_arr[action] = 1;

  return onehot_arr;
}

static inline void calculate_probabilities(const NNtype *inputs,
                                           NNNetwork *network) {
  feed_network(network, inputs);

  softmax(network->final_output_length, network->final_output);
}

static void reinforce_train_step(NNTrainer *trainer, unsigned episode_length,
                                 const NNtype *episode_inputs,
                                 const NNtype *episode_advantage,
                                 const unsigned *episode_actions) {
  assert(trainer);
  assert(episode_inputs);
  assert(episode_advantage);
  assert(episode_actions);
  NNNetwork *network = trainer->network;
  assert(network);

  memset(trainer->slopes, 0, trainer->slopes_length * sizeof(NNtype));

  for (unsigned replay_idx = 0; replay_idx < episode_length; replay_idx++) {
    const NNtype *episode_positions =
        episode_inputs + network->layers[0].input_length * replay_idx;
    calculate_probabilities(episode_positions, network);

    // gradient for the action taken
    const NNtype action_grad =
        episode_advantage[replay_idx] *
        (network->final_output[episode_actions[replay_idx]] - 1);

    // gradient
    for (unsigned i = 0; i < network->final_output_length; i++) {
      network->final_output[i] =
          episode_advantage[replay_idx] * network->final_output[i];
    }

    network->final_output[episode_actions[replay_idx]] = action_grad;

    calculate_layers_delta(trainer, network->final_output);

    calculate_network_slopes(trainer, episode_positions);
  }

  apply_slopes(trainer, episode_length);
}

static void value_network_train_step(NNTrainer *trainer,
                                     unsigned episode_length,
                                     const NNtype *episode_inputs,
                                     const NNtype *episode_returns,
                                     NNtype *episode_advantage) {
  assert(trainer);
  assert(episode_length);
  assert(episode_inputs);
  assert(episode_returns);
  assert(episode_advantage);
  NNNetwork *network = trainer->network;
  assert(network);

  memset(trainer->slopes, 0, trainer->slopes_length * sizeof(NNtype));

  for (unsigned replay_idx = 0; replay_idx < episode_length; replay_idx++) {
    const NNtype *episode_positions =
        episode_inputs + network->layers[0].input_length * replay_idx;

    // Has feed network in it, so we can safely use the network output as
    // value after calling it.
    calculate_output_gradient(trainer, episode_positions,
                              &episode_returns[replay_idx]);

    episode_advantage[replay_idx] =
        episode_returns[replay_idx] - network->final_output[0];

    calculate_layers_delta(trainer, NULL);

    calculate_network_slopes(trainer, episode_positions);
  }

  apply_slopes(trainer, episode_length);
}

typedef struct {
  NNtype x;
  NNtype x_velocity;
  NNtype pole_angle;
  NNtype pole_angular_velocity;
} WorldState;

// return true if cart is within bounds, false otherwise
static bool world_step(WorldState *world, unsigned agent_action) {
  assert(agent_action == 0 || agent_action == 1);
  assert(world);

  const NNtype gravity = 9.8f;
  const NNtype cart_mass = 1.0f;
  const NNtype pole_mass = 0.1f;
  const NNtype total_mass = cart_mass + pole_mass;
  const NNtype half_pole_length = 0.5f;
  const NNtype push_force =
      10.0f * (agent_action ? 1 : -1); // (sign = push direction)
  const NNtype step_duration = 0.02f;  // in seconds
  const NNtype rod_constant = 4.0f / 3.0f;

  const NNtype temp =
      (push_force +
       pole_mass * half_pole_length * world->pole_angular_velocity *
           world->pole_angular_velocity * sinf(world->pole_angle)) /
      total_mass;
  const NNtype pole_angle_cos = cosf(world->pole_angle);
  const NNtype pole_angle_acc =
      (gravity * sinf(world->pole_angle) - pole_angle_cos * temp) /
      (0.5f * (rod_constant -
               pole_mass * pole_angle_cos * pole_angle_cos / total_mass));

  const NNtype x_acc = temp - pole_mass * half_pole_length * pole_angle_acc *
                                  cosf(world->pole_angle) / total_mass;

  world->x += step_duration * world->x_velocity;
  world->x_velocity += step_duration * x_acc;

  world->pole_angle += step_duration * world->pole_angular_velocity;
  world->pole_angular_velocity += step_duration * pole_angle_acc;

  return fabsf(world->pole_angle) < 0.2095f && fabsf(world->x) < 2.4f;
}

int main(void) {
#if 0
#define print_returns(returns, length)                                         \
  putchar('[');                                                                \
  for (unsigned i = 0; i < (length); i++) {                                    \
    printf("%g", (double)(returns)[i]);                                        \
    if (i < (length) - 1)                                                      \
      printf(", ");                                                            \
  }                                                                            \
  printf("]\n");

  NNtype rewards[3];
  NNtype returns[3];

  rewards[0] = 0;
  rewards[1] = 0;
  rewards[2] = 1;

  calculate_returns(0.9f, ARR_LENGTH(returns), rewards, returns);

  print_returns(returns, ARR_LENGTH(returns));

  rewards[0] = 1;
  rewards[1] = 0;
  rewards[2] = 1;

  calculate_returns(0.5f, ARR_LENGTH(returns), rewards, returns);

  print_returns(returns, ARR_LENGTH(returns));

  NNtype wander_rewards[6] = {0, 0, 0, 0, 0, 1};
  NNtype wander_returns[6];

  calculate_returns(0.9f, 6, wander_rewards, wander_returns);

  print_returns(wander_returns, ARR_LENGTH(wander_returns));

#else
  pcg32_random_t rng;
  pcg32_srandom_r(&rng, 0, 0);

  NNNetwork policy_network = {0};
  const unsigned policy_layers[] = {4, 8, 2};
  init_network(&policy_network, ARR_LENGTH(policy_layers), policy_layers);
  for (unsigned i = 0; i < policy_network.layers_length; i++) {
    policy_network.layers[i].init_type = NN_HE_INIT;
  }
  init_network_weights(&policy_network, &rng);

  NNTrainer policy_trainer = {0};
  init_trainer(&policy_trainer, &policy_network, NN_ADAM_OPTIMIZER);
  policy_trainer.learning_rate = 0.015f;
  policy_trainer.learning_rate_decay = 0.997f;
  policy_trainer.learning_rate_min = 3e-4f;
  policy_trainer.loss_type = NN_CROSS_ENTROPY_LOSS;

  NNNetwork value_network = {0};
  const unsigned value_layers[] = {4, 8, 1};
  init_network(&value_network, ARR_LENGTH(value_layers), value_layers);
  for (unsigned i = 0; i < value_network.layers_length; i++) {
    value_network.layers[i].init_type = NN_HE_INIT;
  }
  init_network_weights(&value_network, &rng);

  NNTrainer value_trainer = {0};
  init_trainer(&value_trainer, &value_network, NN_SGD_OPTIMIZER);
  value_trainer.learning_rate = 0.01f;
  value_trainer.loss_type = NN_MSE_LOSS;

  printf("policy layers = {");
  for (unsigned i = 0; i < ARR_LENGTH(policy_layers); i++) {
    printf("%u", policy_layers[i]);
    if (i < ARR_LENGTH(policy_layers) - 1)
      putchar(',');
  }
  printf("}\n");
  printf("value layers  = {");
  for (unsigned i = 0; i < ARR_LENGTH(value_layers); i++) {
    printf("%u", value_layers[i]);
    if (i < ARR_LENGTH(value_layers) - 1)
      putchar(',');
  }
  printf("}\n\n");
#define EPISODE_AMOUNT 1000
#define EPISODE_LENGTH 200
  NNtype episode_inputs[EPISODE_LENGTH][4];
  unsigned episode_actions[EPISODE_LENGTH];
  NNtype episode_rewards[EPISODE_LENGTH];
  NNtype episode_returns[EPISODE_LENGTH];
  NNtype episode_advantage[EPISODE_LENGTH];

  unsigned steps_took = 0;

  for (unsigned episode = 0; episode < EPISODE_AMOUNT; episode++) {
    memset(episode_inputs, 0, sizeof **episode_inputs * EPISODE_LENGTH * 4);
    memset(episode_actions, 0, sizeof *episode_actions * EPISODE_LENGTH);
    memset(episode_rewards, 0, sizeof *episode_rewards * EPISODE_LENGTH);
    memset(episode_returns, 0, sizeof *episode_returns * EPISODE_LENGTH);

    unsigned step = 0;
    WorldState world = {
        pcg32_float_range(&rng, -0.05f, 0.05f),
        pcg32_float_range(&rng, -0.05f, 0.05f),
        pcg32_float_range(&rng, -0.05f, 0.05f),
        pcg32_float_range(&rng, -0.05f, 0.05f),
    };

    for (; step < EPISODE_LENGTH; step++) {
      episode_inputs[step][0] = world.x;
      episode_inputs[step][1] = world.x_velocity;
      episode_inputs[step][2] = world.pole_angle;
      episode_inputs[step][3] = world.pole_angular_velocity;

      calculate_probabilities(episode_inputs[step], &policy_network);

      unsigned idx =
          pcg32_categorical_sample(&rng, policy_network.final_output_length,
                                   policy_network.final_output);

      episode_actions[step] = idx;

      if (!world_step(&world, idx)) {
        step += 1;
        break;
      }

      const NNtype reward = 1.0f;

      episode_rewards[step] = (NNtype)reward;
    }
    steps_took += step;

    calculate_returns(0.9f, step, episode_rewards, episode_returns);

    value_network_train_step(&value_trainer, step, &episode_inputs[0][0],
                             episode_returns, episode_advantage);

    reinforce_train_step(&policy_trainer, step, &episode_inputs[0][0],
                         episode_advantage, episode_actions);

    if (episode && episode % 200 == 0) {
      printf("Average episode length: %u, current episode (%u)\n", steps_took / 200, episode);
      steps_took = 0;
    }
  }
  printf("\nAverage episode length: %u\n", steps_took / 199);

  free_trainer(&policy_trainer);
  free_network(&policy_network);

  free_trainer(&value_trainer);
  free_network(&value_network);
#endif
  return 0;
}
