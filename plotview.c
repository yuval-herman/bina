#define _POSIX_C_SOURCE 200809L

#include "plot.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_index(const char *text, size_t *index) {
  char *end = NULL;
  errno = 0;
  const unsigned long value = strtoul(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value == 0)
    return -1;
  *index = (size_t)value;
  return (unsigned long)*index == value ? 0 : -1;
}

static void wait_for_windows_to_close(void) {
  fprintf(stderr,
          "Close the plot window(s), then press Enter here to exit and "
          "release plot data.\n");
  int ch;
  do {
    ch = getchar();
  } while (ch != '\n' && ch != EOF);
}

int main(int argc, char **argv) {
  int individual = 0;
  int select_subplot = 0;
  size_t row = 0;
  size_t column = 0;
  const char *path = NULL;

  if (argc == 2 && strcmp(argv[1], "--help") == 0) {
    printf("Usage:\n");
    printf("  %s FILE.cplot\n", argv[0]);
    printf("  %s --individual FILE.cplot\n", argv[0]);
    printf("  %s --subplot ROW COL FILE.cplot\n", argv[0]);
    printf("ROW and COL are 1-based for --subplot.\n");
    return 0;
  }
  if (argc == 2) {
    path = argv[1];
  } else if (argc == 3 && strcmp(argv[1], "--individual") == 0) {
    individual = 1;
    path = argv[2];
  } else if (argc == 5 && strcmp(argv[1], "--subplot") == 0) {
    select_subplot = 1;
    if (parse_index(argv[2], &row) != 0 ||
        parse_index(argv[3], &column) != 0) {
      fprintf(stderr, "plotview: ROW and COL must be positive integers.\n");
      return 2;
    }
    row--;
    column--;
    path = argv[4];
  } else {
    fprintf(stderr,
            "Usage: %s [--individual | --subplot ROW COL] FILE.cplot\n",
            argv[0]);
    return 2;
  }

  FILE *file = fopen(path, "r");
  if (file == NULL) {
    perror(path);
    return 1;
  }
  char magic[16];
  const int read_ok = fscanf(file, "%15s", magic) == 1;
  (void)fclose(file);
  if (!read_ok) {
    fprintf(stderr, "plotview: '%s' is empty or malformed.\n", path);
    return 1;
  }

  if (strcmp(magic, "CPLOTMULTI") == 0) {
    PlotMulti *multi = plot_multi_load(path);
    if (multi == NULL)
      return 1;

    int status;
    if (select_subplot) {
      Plot *plot = plot_multi_get(multi, row, column);
      if (plot == NULL) {
        fprintf(stderr, "plotview: subplot row %zu, column %zu does not exist.\n",
                row + 1, column + 1);
        plot_multi_destroy(multi);
        return 2;
      }
      status = plot_show(plot);
    } else if (individual) {
      status = plot_multi_show_individual(multi);
    } else {
      status = plot_multi_show(multi);
    }

    if (status == 0 && (individual || select_subplot))
      wait_for_windows_to_close();
    plot_multi_destroy(multi);
    return status == 0 ? 0 : 1;
  }

  if (strcmp(magic, "CPLOT") == 0 && !select_subplot && !individual) {
    Plot *plot = plot_load(path);
    if (plot == NULL)
      return 1;
    const int status = plot_show(plot);
    plot_destroy(plot);
    return status == 0 ? 0 : 1;
  }

  if (select_subplot || individual) {
    fprintf(stderr,
            "plotview: --individual and --subplot require a multiplot state "
            "file.\n");
    return 2;
  }

  fprintf(stderr, "plotview: '%s' is not a supported cplot state file.\n", path);
  return 1;
}
