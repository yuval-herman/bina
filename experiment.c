#include "nn.c"

void calculate_returns(NNtype gamma, unsigned length, const NNtype *rewards,
                       NNtype *returns) {
  assert(length);
  assert(rewards);
  assert(returns);

  NNtype running_sum = 0;
  for (unsigned i = length; i > 0; i--) {
    running_sum *= gamma;
    returns[i - 1] = running_sum += rewards[i - 1];
  }
}

unsigned world_reward(unsigned *mouse_pos, unsigned mouse_step) {
  assert(mouse_step == 0 || mouse_step == 1);
  assert(mouse_pos);

  unsigned cheese_pos = 4;

  if (mouse_step == 1) {
    (*mouse_pos)++;
  } else if (*mouse_pos != 0) {
    (*mouse_pos)--;
  }

  return (NNtype)(*mouse_pos == cheese_pos);
}

unsigned agent_action(pcg32_random_t *rng) {
  return pcg32_bounded_rand(rng, 2);
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
  pcg32_srandom_r(&rng, 2, 0);

  NNNetwork policy_network = {0};
  const unsigned policy_layers[] = {5, 2};
  init_network(&policy_network, ARR_LENGTH(policy_layers), policy_layers);
  policy_network.layers[0].activation_type = NN_NO_ACTIVATION;
  init_network_weights(&policy_network, &rng);

  NNTrainer policy_trainer = {0};
  init_trainer(&policy_trainer, &policy_network, NN_ADAM_OPTIMIZER);
  policy_trainer.learning_rate = 0.01f;
  policy_trainer.loss_type = NN_CROSS_ENTROPY_LOSS;

  NNNetwork value_network = {0};
  const unsigned value_layers[] = {5, 1};
  init_network(&value_network, ARR_LENGTH(value_layers), value_layers);
  value_network.layers[0].activation_type = NN_NO_ACTIVATION;
  init_network_weights(&value_network, &rng);

  NNTrainer value_trainer = {0};
  init_trainer(&value_trainer, &value_network, NN_ADAM_OPTIMIZER);
  value_trainer.learning_rate = 0.01f;
  value_trainer.loss_type = NN_MSE_LOSS;

#define EPISODE_LENGTH 20
  unsigned episode_positions[EPISODE_LENGTH];
  unsigned episode_actions[EPISODE_LENGTH];
  NNtype episode_rewards[EPISODE_LENGTH];
  NNtype episode_returns[EPISODE_LENGTH];
  NNtype episode_advantage[EPISODE_LENGTH];
  NNtype net_inputs[5];

  unsigned goal_reached = 0;
  unsigned average_steps = 0;

  for (unsigned episode = 0; episode < 2000; episode++) {
    memset(episode_positions, 0, sizeof *episode_positions * EPISODE_LENGTH);
    memset(episode_actions, 0, sizeof *episode_actions * EPISODE_LENGTH);
    memset(episode_rewards, 0, sizeof *episode_rewards * EPISODE_LENGTH);
    memset(episode_returns, 0, sizeof *episode_returns * EPISODE_LENGTH);

    unsigned mouse = 1;
    unsigned step = 0;
    for (; step < EPISODE_LENGTH; step++) {
      episode_positions[step] = mouse;

      // feed
      memset(net_inputs, 0, sizeof *net_inputs * 5);
      net_inputs[mouse] = 1;
      feed_network(&policy_network, net_inputs);

      // sample
      softmax(policy_network.final_output_length, policy_network.final_output);
      memcpy(policy_trainer.network_loss, policy_network.final_output,
             policy_trainer.network_loss_length *
                 sizeof(*policy_trainer.network_loss));

      unsigned idx =
          pcg32_categorical_sample(&rng, policy_network.final_output_length,
                                   policy_network.final_output);

      episode_actions[step] = idx;

      // reward
      unsigned reward = world_reward(&mouse, idx);

      episode_rewards[step] = (NNtype)reward;

      if (reward == 1) {
        step += 1;
        goal_reached++;
        average_steps += step;
        break;
      }
    }

    calculate_returns(0.9f, step, episode_rewards, episode_returns);

    memset(value_trainer.slopes, 0,
           value_trainer.slopes_length * sizeof(NNtype));

    for (unsigned replay_idx = 0; replay_idx < step; replay_idx++) {
      memset(net_inputs, 0, sizeof *net_inputs * 5);
      net_inputs[episode_positions[replay_idx]] = 1;

      // Has feed network in it, so we can safely use the network output as
      // value after calling it.
      calculate_output_gradient(&value_trainer, net_inputs,
                                &episode_returns[replay_idx]);

      episode_advantage[replay_idx] =
          episode_returns[replay_idx] - value_network.final_output[0];

      calculate_layers_delta(&value_trainer, NULL);

      calculate_network_slopes(&value_trainer, net_inputs);
    }

    apply_slopes(&value_trainer, value_trainer.learning_rate, step);

    memset(policy_trainer.slopes, 0,
           policy_trainer.slopes_length * sizeof(NNtype));

    for (unsigned replay_idx = 0; replay_idx < step; replay_idx++) {
      // feed
      memset(net_inputs, 0, sizeof *net_inputs * 5);
      net_inputs[episode_positions[replay_idx]] = 1;
      feed_network(&policy_network, net_inputs);

      // sample
      softmax(policy_network.final_output_length, policy_network.final_output);
      memcpy(policy_trainer.network_loss, policy_network.final_output,
             policy_trainer.network_loss_length *
                 sizeof(*policy_trainer.network_loss));

      const unsigned idx = episode_actions[replay_idx];

      // gradient for the action taken
      const NNtype action_grad =
          episode_advantage[replay_idx] * (policy_network.final_output[idx] - 1);

      // gradient
      for (unsigned i = 0; i < policy_network.final_output_length; i++) {
        policy_network.final_output[i] =
            episode_advantage[replay_idx] * policy_network.final_output[i];
      }

      policy_network.final_output[idx] = action_grad;

      calculate_layers_delta(&policy_trainer, policy_network.final_output);

      calculate_network_slopes(&policy_trainer, net_inputs);
    }

    apply_slopes(&policy_trainer, policy_trainer.learning_rate, step);

    if (episode && episode % 200 == 0) {
      printf("Goal reached %u times\nAverage steps to reach goal were %g\n\n",
             goal_reached, (double)average_steps / goal_reached);
      goal_reached = 0;
      average_steps = 0;

      printf("value network output:\n");
      printf("position: ");
      for (unsigned i = 0; i < 5; i++) {
        printf("%-7u", i);
      }
      printf("\nvalue:    ");
      for (unsigned i = 0; i < 5; i++) {
        memset(net_inputs, 0, sizeof *net_inputs * 5);
        net_inputs[i] = 1;
        feed_network(&value_network, net_inputs);

        printf("%-7.3f", (double)value_network.final_output[0]);
      }
      printf("\n\n");
    }
  }

  // unsigned steps_took = 0;
  // unsigned reached_goal = 0;

  // for (unsigned episode = 0; episode < 1000; episode++) {
  //   unsigned mouse = 2;
  //   unsigned step = 0;
  //   for (; step < EPISODE_LENGTH; step++) {
  //     unsigned reward = world_reward(&mouse, agent_action(&rng));
  //     if (reward == 1) {
  //       reached_goal++;
  //       break;
  //     }
  //   }
  //   steps_took += step+1;
  // }

  // printf("Episodes that reached the goal: %u\n", reached_goal);
  // printf("Average steps to reach goal: %u\n", steps_took / 1000);

  free_trainer(&policy_trainer);
  free_network(&policy_network);

  free_trainer(&value_trainer);
  free_network(&value_network);
#endif
  return 0;
}
