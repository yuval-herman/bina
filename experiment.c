#include "nn.c"

NNtype word_reward(pcg32_random_t *rng, unsigned action) {
  assert(action == 0 || action == 1);
  float n = pcg32_uniform_float(rng);
  if (action == 0) {
    return n < 0.3f ? 1 : 0;
  } else {
    return n < 0.7f ? 1 : 0;
  }
}

int main(void) {
  pcg32_random_t rng;
  pcg32_srandom_r(&rng, 0, 0);

  NNtype avg = 0;
  unsigned tries = 10000;
  for (unsigned try = 0; try < tries; try++) {
    avg += word_reward(&rng, pcg32_bounded_rand(&rng, 2));
  }

  printf("Average reward: %g\n", (double)(avg /= (NNtype)tries));

  return 0;
}
