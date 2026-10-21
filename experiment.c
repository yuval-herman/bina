#include "nn.c"
#include "plot.h"

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

typedef struct {
  NNtype score;
  NNtype bias;
} ExplainedVariance;

static ExplainedVariance explained_variance(unsigned length, NNtype *targets,
                                            NNtype *predictions) {
  assert(length);

  NNtype t_mean = 0;
  NNtype tp_mean = 0;
  NNtype t_mean_square = 0;
  NNtype tp_mean_square = 0;

  for (unsigned i = 0; i < length; i++) {
    t_mean += targets[i];
    tp_mean += targets[i] - predictions[i];
  }
  t_mean /= (NNtype)length;
  tp_mean /= (NNtype)length;

  for (unsigned i = 0; i < length; i++) {
    const NNtype t = targets[i] - t_mean;
    t_mean_square += t * t;
    const NNtype tp = targets[i] - predictions[i] - tp_mean;
    tp_mean_square += tp * tp;
  }

  return (ExplainedVariance){.score = 1 - tp_mean_square / t_mean_square,
                             .bias = tp_mean};
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
  const unsigned policy_layers[] = {3, 4, 4, 2};
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
  policy_trainer->learning_rate = 1e-4f;
  // policy_trainer->clip_limit = 0.1f;
  // policy_trainer->learning_rate_decay = 0.999f;
  // policy_trainer->learning_rate_min = 1e-5f;
  // policy_trainer->adam_damp = 0.999f;
  // policy_trainer->momentum_damp = 0.899f;

  const unsigned value_layers[] = {3, 8, 1};
  init_network(value_network, ARR_LENGTH(value_layers), value_layers);
  for (unsigned i = 0; i < value_network->layers_length; i++) {
    value_network->layers[i].init_type = NN_HE_INIT;
  }
  for (unsigned i = 0; i < value_network->layers_length - 1; i++) {
    value_network->layers[i].activation_type = NN_TANH_ACTIVATION;
  }
  init_network_weights(value_network, rng);

  init_trainer(value_trainer, value_network, NN_SGD_OPTIMIZER);
  value_trainer->learning_rate = 1e-4f;
  // value_trainer->clip_limit = 3.0f;
  // value_trainer->learning_rate_decay = 0.9999f;
  // value_trainer->learning_rate_min = 1e-7f;
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

double evaluate_policy(NNNetwork *network) {
  const unsigned episode_length = 200;
  const unsigned episode_count = 20;

  double avg_reward = 0;
  for (unsigned i = 0; i < episode_count; i++) {
    WorldState world = {
        .angle = (NNtype)(-M_PI + 2 * M_PI * (i + 0.5) / episode_count),
        .angular_velocity = 0,
    };

    for (unsigned step = 0; step < episode_length; step++) {
      NNtype inputs[3];
      inputs[0] = sinf(world.angle);
      inputs[1] = cosf(world.angle);
      inputs[2] = world.angular_velocity;

      feed_network(network, inputs);
      NNtype action = network->final_output[0];

      const NNtype reward = world_step(&world, action);

      avg_reward += (double)reward;
    }
  }

  return avg_reward / episode_count;
}

// x stay the same and does not get averaged. series count is the amount of
// series data we are averaging here.
void add_value_to_mean_series(Plot *plot, PlotSeries *series,
                              unsigned series_count, size_t index, double x,
                              double y) {
  double prev_y = plot_series_get_y(plot, series, index);
  if (isnan(prev_y)) {
    plot_series_add(plot, series, x, y);
  } else {
    prev_y = (prev_y * (series_count) + y) / (series_count + 1);
    plot_series_set(plot, series, index, x, prev_y);
  }
}

int main(int argc, char **argv) {
  bool live_plot = 0;
  const char *plot_save_path = NULL;
  for (int arg = 1; arg < argc; arg++) {
    if (strcmp(argv[arg], "--plot") == 0) {
      live_plot = 1;
    } else if (strcmp(argv[arg], "--save-plot") == 0 && arg + 1 < argc) {
      plot_save_path = argv[++arg];
    } else if (strcmp(argv[arg], "--help") == 0) {
      printf("Usage: %s [--plot] [--save-plot FILE]\n", argv[0]);
      printf("  --plot           Open/update a live gnuplot window\n");
      printf("  --save-plot FILE Save all seed reward curves after training\n");
      return 0;
    } else {
      fprintf(stderr, "Unknown or incomplete option: %s\n", argv[arg]);
      fprintf(stderr, "Run %s --help for usage.\n", argv[0]);
      return 2;
    }
  }
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
  double last_per_seed[6] = {0};

  PlotMulti *plots = plot_multi_create("Pendulum training session", 2, 2);

  Plot *policy_plot = plot_multi_add(plots, 0, 0, "Policy reward", "Episode",
                                     "Mean episode reward");

  Plot *value_score_plot = plot_multi_add(plots, 1, 0, "Value score", "Episode",
                                          "Mean 100 episode scores");

  Plot *value_bias_plot = plot_multi_add(plots, 1, 1, "Value bias", "Episode",
                                         "Mean 100 episode biases");

  PlotSeries *avg_policy_series =
      plot_add_series(policy_plot, "Average policy score", PLOT_LINESPOINTS);

  PlotSeries *avg_value_series = plot_add_series(
      value_score_plot, "Average value score", PLOT_LINESPOINTS);
  PlotSeries *avg_value_bias_series =
      plot_add_series(value_bias_plot, "Average value bias", PLOT_LINESPOINTS);

  struct {
    PlotSeries *policy;
    PlotSeries *value;
    PlotSeries *value_bias;
  } seed_series[ARR_LENGTH(last_per_seed)] = {0};

  for (unsigned seed = 0; seed < ARR_LENGTH(seed_series); seed++) {
    seed_series[seed].policy =
        plot_add_seriesf(policy_plot, PLOT_LINESPOINTS, "Seed %u", seed);
  }
  for (unsigned seed = 0; seed < ARR_LENGTH(seed_series); seed++) {
    seed_series[seed].value = plot_add_seriesf(
        value_score_plot, PLOT_LINESPOINTS, "Seed %u score", seed);
  }
  for (unsigned seed = 0; seed < ARR_LENGTH(seed_series); seed++) {
    seed_series[seed].value_bias = plot_add_seriesf(
        value_bias_plot, PLOT_LINESPOINTS, "Seed %u bias", seed);
  }

  if (live_plot && plot_multi_show(plots) != 0) {
    fprintf(stderr, "Continuing training without a live plot.\n");
    live_plot = 0;
  }

  for (unsigned seed = 0; seed < ARR_LENGTH(last_per_seed); seed++) {
    pcg32_srandom_r(&rng, seed, 0);

    NNNetwork policy_network = {0};
    NNNetwork value_network = {0};
    NNTrainer policy_trainer = {0};
    NNTrainer value_trainer = {0};

    init_networks_trainers(&rng, &policy_network, &value_network,
                           &policy_trainer, &value_trainer);

    plot_set_annotationf(policy_plot, PLOT_ANNOTATION_RIGHT_BOTTOM,
                         "LR: %g\nLR decay %g\nMin LR %g",
                         (double)policy_trainer.learning_rate,
                         (double)policy_trainer.learning_rate_decay,
                         (double)policy_trainer.learning_rate_min);

#define EPISODE_AMOUNT 6000
#define EPISODE_LENGTH 200
    NNtype episode_inputs[EPISODE_LENGTH + 1][3];
    NNtype episode_values[EPISODE_LENGTH + 1];
    NNtype episode_actions[EPISODE_LENGTH];
    NNtype episode_rewards[EPISODE_LENGTH];
    NNtype episode_returns[EPISODE_LENGTH];
    NNtype episode_advantage[EPISODE_LENGTH];

    ExplainedVariance ev_avg = {0};

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

        episode_rewards[step] = (NNtype)reward;
      }

      episode_inputs[step][0] = sinf(world.angle);
      episode_inputs[step][1] = cosf(world.angle);
      episode_inputs[step][2] = world.angular_velocity;
      feed_network(&value_network, episode_inputs[step]);
      episode_values[step] = value_network.final_output[0];

      float discount = 0.95f;
      calculate_returns(discount, step, episode_rewards, episode_returns);

      ExplainedVariance ev =
          explained_variance(100, episode_returns, episode_values);

      ev_avg.bias += ev.bias;
      ev_avg.score += ev.score;

      calculate_gae(discount, 0.4f, step, episode_rewards, episode_values,
                    episode_advantage);

      value_network_train_step(&value_trainer, step, &episode_inputs[0][0],
                               episode_returns, episode_advantage);

      reinforce_train_step_continous(&policy_trainer, step,
                                     &episode_inputs[0][0], episode_advantage,
                                     episode_actions);

      if (episode % 100 == 99) {
        const double avg_reward = evaluate_policy(&policy_network);
        last_per_seed[seed] = avg_reward;

        plot_series_add(policy_plot, seed_series[seed].policy,
                        (double)(episode + 1), avg_reward);
        add_value_to_mean_series(policy_plot, avg_policy_series, seed + 1,
                                 episode / 100, episode + 1, avg_reward);

        ev_avg.bias /= 100;
        ev_avg.score /= 100;

        plot_series_add(value_score_plot, seed_series[seed].value,
                        (double)(episode + 1), ev_avg.score);
        add_value_to_mean_series(value_score_plot, avg_value_series, seed + 1,
                                 episode / 100, episode + 1, ev_avg.score);
        plot_series_add(value_bias_plot, seed_series[seed].value_bias,
                        (double)(episode + 1), ev_avg.bias);
        add_value_to_mean_series(value_bias_plot, avg_value_bias_series,
                                 seed + 1, episode / 100, episode + 1,
                                 ev_avg.bias);

        ev_avg = (ExplainedVariance){0};
        if (live_plot && plot_multi_refresh(plots) != 0) {
          fprintf(stderr,
                  "Live plotting failed; continuing without updates.\n");
          live_plot = 0;
        }
      }
    }

    free_trainer(&policy_trainer);
    free_network(&policy_network);

    free_trainer(&value_trainer);
    free_network(&value_network);
  }

  double overall_avg = 0;
  if (plot_save_path != NULL && plot_save_path[0] != '\0') {
    if (plot_multi_save(plots, plot_save_path) != 0) {
      fprintf(stderr, "Failed to save plot state to '%s'.\n", plot_save_path);
      plot_multi_destroy(plots);
      return 1;
    }
    printf("Saved training plot to %s\n", plot_save_path);
  }
  plot_multi_destroy(plots);

  printf("\n\nLast 100 episode averages per seed\nseed average\n");
  for (unsigned seed = 0; seed < ARR_LENGTH(last_per_seed); seed++) {
    printf("%-4u %g\n", seed, last_per_seed[seed]);
    overall_avg += last_per_seed[seed];
  }
  overall_avg /= (double)ARR_LENGTH(last_per_seed);
  printf("\nOverall average over %lu seeds: %g\n", ARR_LENGTH(last_per_seed),
         overall_avg);
#endif
  return 0;
}
