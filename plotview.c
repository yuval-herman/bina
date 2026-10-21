#include "plot.h"

#include <stdio.h>
#include <string.h>

static void print_usage(const char *program) {
  fprintf(stderr, "Usage: %s STATE.cplot\n", program);
  fprintf(stderr, "       %s --help\n", program);
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "--help") == 0) {
    print_usage(argv[0]);
    return 0;
  }
  const char *path = argc == 2 ? argv[1] : NULL;
  if (path == NULL) {
    print_usage(argv[0]);
    return 2;
  }

  Plot *plot = plot_load(path);
  if (plot == NULL)
    return 1;
  if (plot_show(plot) != 0) {
    plot_destroy(plot);
    return 1;
  }
  /* Keep the pipe and temporary data files alive until the user presses Enter. */
  fprintf(stderr, "Plot opened. Press Enter to close the helper.\n");
  (void)getchar();
  plot_destroy(plot);
  return 0;
}
