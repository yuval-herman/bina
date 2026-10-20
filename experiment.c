#include "nn.c"

static void calculate_returns(NNtype discount, unsigned length,
                              const NNtype *rewards, NNtype *returns) {
  assert(0 <= discount && discount <= 1);
  assert(length);
  assert(rewards);
  assert(returns);

  NNtype running_sum = 0;
  for (unsigned i = length; i > 0; i--) {
    running_sum *= discount;
    returns[i - 1] = running_sum += rewards[i - 1];
  }
}

static void calculate_gae(NNtype discount, NNtype gae_lambda, unsigned length,
                          const NNtype *rewards, NNtype *values,
                          NNtype *advantages) {
  assert(0 <= discount && discount <= 1);
  assert(0 <= gae_lambda && gae_lambda <= 1);
  assert(length > 0);
  assert(rewards);
  assert(values);
  assert(advantages);

  const NNtype gae_mul = discount * gae_lambda;

#define surprise(idx) (rewards[idx] + discount * values[idx + 1] - values[idx])
  advantages[length - 1] = surprise(length - 1);

  for (unsigned i = length - 1; i > 0; i--) {
    advantages[i - 1] = surprise(i - 1) + gae_mul * advantages[i];
  }
#undef surprise
}

// Calculates log probability of a Gaussian distribution
static NNtype calculate_log_probability(NNtype action, NNtype mean,
                                        NNtype stdev) {
  assert(stdev > 0);
  const NNtype ln_2_pi = logf(2 * (NNtype)M_PI) / 2;
  const NNtype z = (action - mean) / stdev;

  return -(z * z) / 2 - logf(stdev) - ln_2_pi;
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

static inline NNtype get_agent_action(pcg32_random_t *rng, const NNtype *inputs,
                                      NNNetwork *network) {
  feed_network(network, inputs);

  const NNtype mean = network->final_output[0];
  const NNtype stdev = expf(network->final_output[1]);

  return mean + stdev * pcg32_normal_float(rng);
}

static void reinforce_train_step_continous(NNTrainer *trainer,
                                           unsigned episode_length,
                                           const NNtype *episode_inputs,
                                           const NNtype *episode_advantage,
                                           const NNtype *episode_actions) {
  assert(trainer);
  assert(episode_inputs);
  assert(episode_advantage);
  assert(episode_actions);
  NNNetwork *network = trainer->network;
  assert(network);

  memset(trainer->slopes, 0, trainer->slopes_length * sizeof(NNtype));

  for (unsigned replay_idx = 0; replay_idx < episode_length; replay_idx++) {
    const NNtype *inputs =
        episode_inputs + network->layers[0].input_length * replay_idx;
    feed_network(network, inputs);

    const NNtype mean = network->final_output[0];
    const NNtype stdev_recip = 1 / expf(network->final_output[1]);

    const NNtype z = (episode_actions[replay_idx] - mean) * stdev_recip;
    const NNtype advantage = episode_advantage[replay_idx];

    network->final_output[0] = -advantage * z * stdev_recip;
    network->final_output[1] = -advantage * (z * z - 1);

    calculate_layers_delta(trainer, network->final_output);

    calculate_network_slopes(trainer, inputs);
  }

  apply_slopes(trainer, episode_length);
}

static void reinforce_train_step(NNTrainer *trainer, unsigned episode_length,
                                 const NNtype *episode_inputs,
                                 const NNtype *episode_advantage,
                                 const NNtype *episode_actions) {
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
    feed_network(network, episode_positions);

    // gradient for the action taken
    const NNtype action_grad =
        episode_advantage[replay_idx] * (network->final_output[0] - 1);

    // gradient
    for (unsigned i = 0; i < network->final_output_length; i++) {
      network->final_output[i] =
          episode_advantage[replay_idx] * network->final_output[i];
    }

    network->final_output[0] = action_grad;

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

    calculate_layers_delta(trainer, NULL);

    calculate_network_slopes(trainer, episode_positions);
  }

  apply_slopes(trainer, episode_length);
}

typedef struct {
  NNtype angle;
  NNtype angular_velocity;
} WorldState;

static inline NNtype clamp(NNtype value, NNtype min, NNtype max) {
  return (value < min) ? min : ((value > max) ? max : value);
}

static inline NNtype wrap(NNtype value, NNtype min, NNtype max) {
  const NNtype range = max - min;
  const NNtype inv_range = 1.0f / range;

  const NNtype t = (value - min) * inv_range;
  const NNtype wrapped_t = t - floorf(t);

  return wrapped_t * range + min;
}

// Returns the reward. Agent action is the torque.
static NNtype world_step(WorldState *world, NNtype agent_action) {
  assert(world);

  const NNtype step_duration = 0.05f; // in seconds

  const NNtype gravity = 10.0f;
  const NNtype mass = 1.0f;
  const NNtype length = 1.0f;
  const NNtype max_torque = 2.0f;
  const NNtype max_speed = 8.0f;

  const NNtype t = wrap(world->angle, -(NNtype)M_PI, (NNtype)M_PI);
  const NNtype td = world->angular_velocity;
  const NNtype u = clamp(agent_action, -max_torque, max_torque);

  const NNtype reward = -(t * t + 0.1f * td * td + 0.001f * u * u);

  world->angular_velocity =
      clamp(td + (((3 * gravity) / (2 * length)) * sinf(t) +
                  (3 / (mass * length * length)) * u) *
                     step_duration,
            -max_speed, max_speed);
  world->angle += world->angular_velocity * 0.05f;

  return reward;
}

void init_networks_trainers(pcg32_random_t *rng, NNNetwork *policy_network,
                            NNNetwork *value_network, NNTrainer *policy_trainer,
                            NNTrainer *value_trainer) {
  const unsigned policy_layers[] = {3, 8, 2};
  init_network(policy_network, ARR_LENGTH(policy_layers), policy_layers);
  for (unsigned i = 0; i < policy_network->layers_length; i++) {
    policy_network->layers[i].init_type = NN_HE_INIT;
  }
  for (unsigned i = 0; i < policy_network->layers_length - 1; i++) {
    policy_network->layers[i].activation_type = NN_TANH_ACTIVATION;
  }
  init_network_weights(policy_network, rng);
  {
    const NNLayer layer =
        policy_network->layers[policy_network->layers_length - 1];
    const unsigned parameter_count =
        layer.input_length * layer.output_length + layer.output_length;
    NNtype *parameters = layer.weights;
    for (unsigned parameter_idx = 0; parameter_idx < parameter_count;
         parameter_idx++) {
      parameters[parameter_idx] *= 0.01f;
    }
  }
  init_trainer(policy_trainer, policy_network, NN_ADAM_OPTIMIZER);
  policy_trainer->learning_rate = 0.01f;
  policy_trainer->clip_limit = 2.0f;
  policy_trainer->learning_rate_decay = 0.9999f;
  policy_trainer->learning_rate_min = 0.001f;
  policy_trainer->adam_damp = 0.999f;
  policy_trainer->momentum_damp = 0.899f;

  const unsigned value_layers[] = {3, 8, 1};
  init_network(value_network, ARR_LENGTH(value_layers), value_layers);
  for (unsigned i = 0; i < value_network->layers_length; i++) {
    value_network->layers[i].init_type = NN_HE_INIT;
  }
  for (unsigned i = 0; i < value_network->layers_length; i++) {
    value_network->layers[i].activation_type = NN_TANH_ACTIVATION;
  }
  init_network_weights(value_network, rng);

  init_trainer(value_trainer, value_network, NN_SGD_OPTIMIZER);
  value_trainer->learning_rate = 0.04f;
  value_trainer->clip_limit = 3.0f;
  value_trainer->learning_rate_decay = 0.996f;
  value_trainer->learning_rate_min = 0.001f;
  value_trainer->loss_type = NN_MSE_LOSS;

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
}

int main(void) {
#if 0
#define print_array(arr, length)                                               \
  putchar('[');                                                                \
  for (unsigned i = 0; i < (length); i++) {                                    \
    printf("%g", (double)(arr)[i]);                                            \
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

  print_array(returns, ARR_LENGTH(returns));

  rewards[0] = 1;
  rewards[1] = 0;
  rewards[2] = 1;

  calculate_returns(0.5f, ARR_LENGTH(returns), rewards, returns);

  print_array(returns, ARR_LENGTH(returns));

  NNtype wander_rewards[6] = {0, 0, 0, 0, 0, 1};
  NNtype wander_returns[6];

  calculate_returns(0.9f, 6, wander_rewards, wander_returns);

  print_array(wander_returns, ARR_LENGTH(wander_returns));

  NNtype gae_advantage[3];
  NNtype values[4];

  rewards[0] = 0;
  rewards[1] = 0;
  rewards[2] = 1;

  values[0] = 0.5f;
  values[1] = 0.7f;
  values[2] = 0.9f;
  values[3] = 0.0f;
  calculate_gae(0.9f, 0, 3, rewards, values, gae_advantage);

  print_array(gae_advantage, ARR_LENGTH(gae_advantage));
#else
  pcg32_random_t rng;
  double last_per_seed[10] = {0};

  for (unsigned seed = 0; seed < ARR_LENGTH(last_per_seed); seed++) {
    pcg32_srandom_r(&rng, seed, 0);

    NNNetwork policy_network = {0};
    NNNetwork value_network = {0};
    NNTrainer policy_trainer = {0};
    NNTrainer value_trainer = {0};

    init_networks_trainers(&rng, &policy_network, &value_network,
                           &policy_trainer, &value_trainer);

#define EPISODE_AMOUNT 1000
#define EPISODE_LENGTH 200
    NNtype episode_inputs[EPISODE_LENGTH + 1][3];
    NNtype episode_values[EPISODE_LENGTH + 1];
    NNtype episode_actions[EPISODE_LENGTH];
    NNtype episode_rewards[EPISODE_LENGTH];
    NNtype episode_returns[EPISODE_LENGTH];
    NNtype episode_advantage[EPISODE_LENGTH];

    double avg_reward = 0;

    for (unsigned episode = 0; episode < EPISODE_AMOUNT; episode++) {
      memset(episode_inputs, 0, sizeof **episode_inputs * EPISODE_LENGTH * 3);
      memset(episode_actions, 0, sizeof *episode_actions * EPISODE_LENGTH);
      memset(episode_rewards, 0, sizeof *episode_rewards * EPISODE_LENGTH);
      memset(episode_returns, 0, sizeof *episode_returns * EPISODE_LENGTH);

      unsigned step = 0;
      WorldState world = {
          pcg32_float_range(&rng, -(NNtype)M_PI, (NNtype)M_PI),
          pcg32_float_range(&rng, -1.0f, 1.0f),
      };

      for (; step < EPISODE_LENGTH; step++) {
        episode_inputs[step][0] = sinf(world.angle);
        episode_inputs[step][1] = cosf(world.angle);
        episode_inputs[step][2] = world.angular_velocity;

        NNtype action =
            get_agent_action(&rng, episode_inputs[step], &policy_network);

        episode_actions[step] = action;

        feed_network(&value_network, episode_inputs[step]);
        episode_values[step] = value_network.final_output[0];

        const NNtype reward = world_step(&world, action);

        avg_reward += (double)reward;

        episode_rewards[step] = (NNtype)reward;
      }

      episode_inputs[step][0] = sinf(world.angle);
      episode_inputs[step][1] = cosf(world.angle);
      episode_inputs[step][2] = world.angular_velocity;
      feed_network(&value_network, episode_inputs[step]);
      episode_values[step] = value_network.final_output[0];

      calculate_returns(0.9f, step, episode_rewards, episode_returns);
      calculate_gae(0.9f, 0.5f, step, episode_rewards, episode_values,
                    episode_advantage);

      value_network_train_step(&value_trainer, step, &episode_inputs[0][0],
                               episode_returns, episode_advantage);

      reinforce_train_step_continous(&policy_trainer, step,
                                     &episode_inputs[0][0], episode_advantage,
                                     episode_actions);

      if (episode % 100 == 99) {
        avg_reward /= 100;
        last_per_seed[seed] = avg_reward;
        printf("The average reward is %g\n", avg_reward);
        avg_reward = 0;
      }
    }

    free_trainer(&policy_trainer);
    free_network(&policy_network);

    free_trainer(&value_trainer);
    free_network(&value_network);
  }
  double overall_avg = 0;
  printf("\n\nLast 100 episode averages per seed\nseed average\n");
  for (unsigned seed = 0; seed < ARR_LENGTH(last_per_seed); seed++) {
    printf("%-4u %g\n", seed, last_per_seed[seed]);
    overall_avg += last_per_seed[seed];
  }
  printf("\nOverall average over %lu seed: %g\n", ARR_LENGTH(last_per_seed),
         overall_avg / 10);
#endif
  return 0;
}
