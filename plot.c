#define _POSIX_C_SOURCE 200809L

#include "plot.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CPLOT_TEMP_TEMPLATE "/tmp/cplot-XXXXXX"
#define CPLOT_INITIAL_CAPACITY 16U

typedef struct {
  double x;
  double y;
} PlotPoint;

struct PlotSeries {
  char *name;
  PlotStyle style;
  PlotPoint *points;
  size_t length;
  size_t capacity;
  char *data_path;
};

struct Plot {
  char *title;
  char *x_label;
  char *y_label;
  PlotSeries **series;
  size_t series_length;
  size_t series_capacity;
  FILE *gnuplot;
};

struct PlotMulti {
  char *title;
  size_t rows;
  size_t columns;
  Plot **plots;
  FILE *gnuplot;
  int grid_shown;
  int individual_windows_shown;
};

static char *copy_string(const char *value) {
  if (value == NULL)
    value = "";
  const size_t length = strlen(value);
  if (length == SIZE_MAX)
    return NULL;
  char *copy = malloc(length + 1);
  if (copy != NULL)
    memcpy(copy, value, length + 1);
  return copy;
}

static int set_string(char **target, const char *value) {
  char *copy = copy_string(value);
  if (copy == NULL)
    return -1;
  free(*target);
  *target = copy;
  return 0;
}

static int owns_series(const Plot *plot, const PlotSeries *series) {
  if (plot == NULL || series == NULL)
    return 0;
  for (size_t i = 0; i < plot->series_length; i++) {
    if (plot->series[i] == series)
      return 1;
  }
  return 0;
}

static int reserve_points(PlotSeries *series, size_t required) {
  if (required <= series->capacity)
    return 0;
  if (required > SIZE_MAX / sizeof *series->points)
    return -1;

  size_t capacity =
      series->capacity == 0 ? CPLOT_INITIAL_CAPACITY : series->capacity;
  while (capacity < required) {
    if (capacity > SIZE_MAX / 2) {
      capacity = required;
      break;
    }
    capacity *= 2;
  }
  if (capacity > SIZE_MAX / sizeof *series->points)
    return -1;

  PlotPoint *points = realloc(series->points, capacity * sizeof *points);
  if (points == NULL)
    return -1;
  series->points = points;
  series->capacity = capacity;
  return 0;
}

static int path_is_executable(const char *path) {
  return path != NULL && path[0] != '\0' && access(path, X_OK) == 0;
}

int plot_is_available(void) {
  const char *path = getenv("PATH");
  if (path == NULL)
    return 0;

  const char *part = path;
  for (;;) {
    const char *end = strchr(part, ':');
    const size_t length = end == NULL ? strlen(part) : (size_t)(end - part);
    const char *directory = length == 0 ? "." : part;
    const size_t directory_length = length == 0 ? 1 : length;

    if (directory_length <= SIZE_MAX - sizeof "/gnuplot") {
      char *candidate = malloc(directory_length + sizeof "/gnuplot");
      if (candidate == NULL)
        return 0;
      memcpy(candidate, directory, directory_length);
      memcpy(candidate + directory_length, "/gnuplot", sizeof "/gnuplot");
      const int found = path_is_executable(candidate);
      free(candidate);
      if (found)
        return 1;
    }

    if (end == NULL)
      break;
    part = end + 1;
  }
  return 0;
}

Plot *plot_create(const char *title, const char *x_label, const char *y_label) {
  Plot *plot = calloc(1, sizeof *plot);
  if (plot == NULL)
    return NULL;
  plot->title = copy_string(title);
  plot->x_label = copy_string(x_label);
  plot->y_label = copy_string(y_label);
  if (plot->title == NULL || plot->x_label == NULL || plot->y_label == NULL) {
    plot_destroy(plot);
    return NULL;
  }
  return plot;
}

static void remove_data_file(PlotSeries *series) {
  if (series->data_path != NULL) {
    (void)unlink(series->data_path);
    free(series->data_path);
  }
}

void plot_destroy(Plot *plot) {
  if (plot == NULL)
    return;
  if (plot->gnuplot != NULL) {
    (void)fflush(plot->gnuplot);
    (void)pclose(plot->gnuplot);
  }
  for (size_t i = 0; i < plot->series_length; i++) {
    PlotSeries *series = plot->series[i];
    if (series == NULL)
      continue;
    remove_data_file(series);
    free(series->points);
    free(series->name);
    free(series);
  }
  free(plot->series);
  free(plot->title);
  free(plot->x_label);
  free(plot->y_label);
  free(plot);
}

PlotSeries *plot_add_series(Plot *plot, const char *name, PlotStyle style) {
  if (plot == NULL || style < PLOT_LINES || style > PLOT_IMPULSES)
    return NULL;
  if (plot->series_length == SIZE_MAX / sizeof *plot->series)
    return NULL;

  if (plot->series_length == plot->series_capacity) {
    size_t capacity = plot->series_capacity == 0 ? 4 : plot->series_capacity;
    if (plot->series_capacity != 0) {
      if (capacity > SIZE_MAX / 2)
        return NULL;
      capacity *= 2;
    }
    if (capacity > SIZE_MAX / sizeof *plot->series)
      return NULL;
    PlotSeries **series = realloc(plot->series, capacity * sizeof *series);
    if (series == NULL)
      return NULL;
    plot->series = series;
    plot->series_capacity = capacity;
  }

  PlotSeries *series = calloc(1, sizeof *series);
  if (series == NULL)
    return NULL;
  series->name = copy_string(name);
  if (series->name == NULL) {
    free(series);
    return NULL;
  }
  series->style = style;
  plot->series[plot->series_length++] = series;
  return series;
}

int plot_series_add(Plot *plot, PlotSeries *series, double x, double y) {
  if (!owns_series(plot, series) || !isfinite(x) || !isfinite(y) ||
      series->length == SIZE_MAX)
    return -1;
  if (reserve_points(series, series->length + 1) != 0)
    return -1;
  series->points[series->length++] = (PlotPoint){.x = x, .y = y};
  return 0;
}

int plot_series_set(Plot *plot, PlotSeries *series, size_t index, double x,
                    double y) {
  if (!owns_series(plot, series) || index >= series->length || !isfinite(x) ||
      !isfinite(y))
    return -1;
  series->points[index] = (PlotPoint){.x = x, .y = y};
  return 0;
}

double plot_series_get_x(Plot *plot, PlotSeries *series, size_t index) {
  if (!owns_series(plot, series) || index >= series->length)
    return NAN;
  return series->points[index].x;
}

double plot_series_get_y(Plot *plot, PlotSeries *series, size_t index) {
  if (!owns_series(plot, series) || index >= series->length)
    return NAN;
  return series->points[index].y;
}

int plot_series_truncate(Plot *plot, PlotSeries *series, size_t length) {
  if (!owns_series(plot, series) || length > series->length)
    return -1;
  series->length = length;
  return 0;
}

void plot_series_clear(Plot *plot, PlotSeries *series) {
  if (owns_series(plot, series))
    series->length = 0;
}

size_t plot_series_size(const PlotSeries *series) {
  return series == NULL ? 0 : series->length;
}

void plot_set_title(Plot *plot, const char *title) {
  if (plot != NULL)
    (void)set_string(&plot->title, title);
}

void plot_set_x_label(Plot *plot, const char *label) {
  if (plot != NULL)
    (void)set_string(&plot->x_label, label);
}

void plot_set_y_label(Plot *plot, const char *label) {
  if (plot != NULL)
    (void)set_string(&plot->y_label, label);
}

static int ensure_data_path(PlotSeries *series) {
  if (series->data_path != NULL)
    return 0;
  char path[] = CPLOT_TEMP_TEMPLATE;
  const int fd = mkstemp(path);
  if (fd < 0)
    return -1;
  if (close(fd) != 0) {
    (void)unlink(path);
    return -1;
  }
  series->data_path = copy_string(path);
  if (series->data_path == NULL) {
    (void)unlink(path);
    return -1;
  }
  return 0;
}

static int write_series_data(PlotSeries *series) {
  if (ensure_data_path(series) != 0)
    return -1;
  FILE *file = fopen(series->data_path, "w");
  if (file == NULL)
    return -1;
  int failed = 0;
  for (size_t i = 0; i < series->length; i++) {
    if (fprintf(file, "%.17g %.17g\n", series->points[i].x,
                series->points[i].y) < 0) {
      failed = 1;
      break;
    }
  }
  if (fclose(file) != 0)
    failed = 1;
  return failed ? -1 : 0;
}

static int write_quoted(FILE *file, const char *value) {
  if (fputc('"', file) == EOF)
    return -1;
  for (const unsigned char *p = (const unsigned char *)value; *p != '\0'; p++) {
    switch (*p) {
    case '\\':
      if (fputs("\\\\", file) == EOF)
        return -1;
      break;
    case '"':
      if (fputs("\\\"", file) == EOF)
        return -1;
      break;
    case '\n':
      if (fputs("\\n", file) == EOF)
        return -1;
      break;
    case '\r':
      if (fputs("\\r", file) == EOF)
        return -1;
      break;
    case '\t':
      if (fputs("\\t", file) == EOF)
        return -1;
      break;
    default:
      if (fputc(*p, file) == EOF)
        return -1;
      break;
    }
  }
  return fputc('"', file) == EOF ? -1 : 0;
}

static int write_gp_quoted(FILE *file, const char *value) {
  return write_quoted(file, value);
}

static const char *gnuplot_style(PlotStyle style) {
  switch (style) {
  case PLOT_POINTS:
    return "points";
  case PLOT_LINESPOINTS:
    return "linespoints";
  case PLOT_IMPULSES:
    return "impulses";
  case PLOT_LINES:
  default:
    return "lines";
  }
}

static int send_plot_commands_to(FILE *out, const Plot *plot,
                                 int key_outside) {
  if (fprintf(out, "set title ") < 0 ||
      write_gp_quoted(out, plot->title) != 0 ||
      fprintf(out, "\nset xlabel ") < 0 ||
      write_gp_quoted(out, plot->x_label) != 0 ||
      fprintf(out, "\nset ylabel ") < 0 ||
      write_gp_quoted(out, plot->y_label) != 0 ||
      fprintf(out, "\nset grid\n") < 0)
    return -1;

  if (plot->series_length == 0) {
    if (fprintf(out, "unset key\nplot NaN notitle\n") < 0)
      return -1;
    return 0;
  }

  if (fprintf(out, "set key %s\nplot ",
              key_outside ? "outside" : "top right") < 0)
    return -1;
  for (size_t i = 0; i < plot->series_length; i++) {
    const PlotSeries *series = plot->series[i];
    if (i != 0 && fprintf(out, ", \\\n     ") < 0)
      return -1;
    if (series->data_path == NULL ||
        write_gp_quoted(out, series->data_path) != 0 ||
        fprintf(out, " using 1:2 with %s",
                gnuplot_style(series->style)) < 0)
      return -1;
    if (series->name[0] == '\0') {
      if (fprintf(out, " notitle") < 0)
        return -1;
    } else if (fprintf(out, " title ") < 0 ||
               write_gp_quoted(out, series->name) != 0) {
      return -1;
    }
  }
  return fprintf(out, "\n") < 0 ? -1 : 0;
}

/*
 * Qt and X11 preserve the plot's original aspect ratio when a window is
 * resized unless replotonresize is enabled. Set it only on terminals that
 * support that option. Each Plot process gets its own terminal context, so
 * this also ensures that a resize uses that plot's current data and ranges.
 */
static int send_single_terminal_setup(FILE *out, const Plot *plot) {
  if (fprintf(out,
              "set macros\n"
              "cplot_terminal = GPVAL_TERM\n"
              "if (GPVAL_TERM eq \"qt\") {\n"
              "  set term @cplot_terminal title ") < 0 ||
      write_gp_quoted(out, plot->title) != 0 ||
      fprintf(out,
              " replotonresize\n"
              "}\n"
              "if (GPVAL_TERM eq \"x11\") {\n"
              "  set term @cplot_terminal title ") < 0 ||
      write_gp_quoted(out, plot->title) != 0 ||
      fprintf(out, " replotonresize\n}\n") < 0 || fflush(out) != 0)
    return -1;
  return 0;
}

/* Select a numbered terminal window, enabling resize-replot where available. */
static int send_numbered_terminal_setup(FILE *out, size_t window,
                                        const Plot *plot) {
  if (fprintf(out,
              "if (GPVAL_TERM eq \"qt\") {\n"
              "  set term @cplot_terminal %zu title ", window) < 0 ||
      write_gp_quoted(out, plot->title) != 0 ||
      fprintf(out,
              " replotonresize\n"
              "}\n"
              "if (GPVAL_TERM eq \"x11\") {\n"
              "  set term @cplot_terminal %zu title ", window) < 0 ||
      write_gp_quoted(out, plot->title) != 0 ||
      fprintf(out,
              " replotonresize\n"
              "}\n"
              "if (GPVAL_TERM ne \"qt\" && GPVAL_TERM ne \"x11\") {\n"
              "  set term @cplot_terminal %zu\n"
              "}\n", window) < 0)
    return -1;
  return 0;
}

static int send_plot_commands(Plot *plot) {
  if (send_plot_commands_to(plot->gnuplot, plot, 1) != 0)
    return -1;
  return fflush(plot->gnuplot) != 0 ? -1 : 0;
}

static int write_all_plot_data(Plot *plot) {
  for (size_t i = 0; i < plot->series_length; i++) {
    if (write_series_data(plot->series[i]) != 0) {
      fprintf(stderr, "cplot: could not write plot data: %s\n",
              strerror(errno));
      return -1;
    }
  }
  return 0;
}

static int write_all_multi_data(PlotMulti *multi);

static int send_multiplot_commands(PlotMulti *multi) {
  FILE *out = multi->gnuplot;
  /* Window 0 is reserved for the overview grid. */
  if (fprintf(out, "set term @cplot_terminal 0\n"
                   "set multiplot layout %zu,%zu rowsfirst title ",
              multi->rows, multi->columns) < 0 ||
      write_gp_quoted(out, multi->title) != 0 || fprintf(out, "\n") < 0)
    return -1;

  const size_t count = multi->rows * multi->columns;
  for (size_t i = 0; i < count; i++) {
    const Plot *plot = multi->plots[i];
    if (plot != NULL) {
      if (send_plot_commands_to(out, plot, 0) != 0)
        return -1;
    } else if (fprintf(out,
                       "set title \"\"\nunset key\nset xlabel \"\"\n"
                       "set ylabel \"\"\nset grid\nplot NaN notitle\n") < 0) {
      return -1;
    }
  }
  if (fprintf(out, "unset multiplot\n") < 0 || fflush(out) != 0)
    return -1;
  return 0;
}

/*
 * Numbered interactive-terminal windows let one gnuplot process serve every
 * panel. The terminal name is captured before switching to numbered windows;
 * the default terminal is window 0 and child panels use window index + 1.
 */
static int start_multi_gnuplot(PlotMulti *multi) {
  if (multi->gnuplot != NULL)
    return 0;
  multi->gnuplot = popen("gnuplot -persist", "w");
  if (multi->gnuplot == NULL) {
    fprintf(stderr, "cplot: could not start gnuplot: %s\n", strerror(errno));
    return -1;
  }
  if (fprintf(multi->gnuplot,
              "set macros\ncplot_terminal = GPVAL_TERM\n") < 0 ||
      fflush(multi->gnuplot) != 0) {
    fprintf(stderr, "cplot: could not initialize gnuplot terminal selection.\n");
    (void)pclose(multi->gnuplot);
    multi->gnuplot = NULL;
    return -1;
  }
  return 0;
}

static int send_individual_commands_shared(PlotMulti *multi) {
  FILE *out = multi->gnuplot;
  const size_t count = multi->rows * multi->columns;
  for (size_t i = 0; i < count; i++) {
    Plot *plot = multi->plots[i];
    if (plot == NULL)
      continue;
    /* Keep window IDs stable even if some grid cells are empty. */
    if (send_numbered_terminal_setup(out, i + 1, plot) != 0 ||
        send_plot_commands_to(out, plot, 1) != 0)
      return -1;
  }
  return fflush(out) != 0 ? -1 : 0;
}

static int refresh_shared_multi(PlotMulti *multi) {
  if (write_all_multi_data(multi) != 0)
    return -1;
  if (multi->grid_shown && send_multiplot_commands(multi) != 0) {
    fprintf(stderr, "cplot: failed to refresh multiplot window.\n");
    return -1;
  }
  if (multi->individual_windows_shown &&
      send_individual_commands_shared(multi) != 0) {
    fprintf(stderr, "cplot: failed to refresh individual plot windows.\n");
    return -1;
  }
  return 0;
}

int plot_show(Plot *plot) {
  if (plot == NULL)
    return -1;
  if (plot->gnuplot != NULL)
    return plot_refresh(plot);
  if (!plot_is_available()) {
    fprintf(stderr,
            "cplot: gnuplot was not found in PATH; install gnuplot or save "
            "the plot with plot_save().\n");
    return -1;
  }
  for (size_t i = 0; i < plot->series_length; i++) {
    if (write_series_data(plot->series[i]) != 0) {
      fprintf(stderr, "cplot: could not write temporary plot data: %s\n",
              strerror(errno));
      return -1;
    }
  }
  plot->gnuplot = popen("gnuplot -persist", "w");
  if (plot->gnuplot == NULL) {
    fprintf(stderr, "cplot: could not start gnuplot: %s\n", strerror(errno));
    return -1;
  }
  if (send_single_terminal_setup(plot->gnuplot, plot) != 0 ||
      send_plot_commands(plot) != 0) {
    fprintf(stderr, "cplot: failed to send commands to gnuplot.\n");
    (void)pclose(plot->gnuplot);
    plot->gnuplot = NULL;
    return -1;
  }
  return 0;
}

int plot_refresh(Plot *plot) {
  if (plot == NULL)
    return -1;
  if (plot->gnuplot == NULL)
    return plot_show(plot);
  for (size_t i = 0; i < plot->series_length; i++) {
    if (write_series_data(plot->series[i]) != 0) {
      fprintf(stderr, "cplot: could not update plot data: %s\n",
              strerror(errno));
      return -1;
    }
  }
  if (send_plot_commands(plot) != 0) {
    fprintf(stderr, "cplot: failed to refresh gnuplot window.\n");
    return -1;
  }
  return 0;
}


PlotMulti *plot_multi_create(const char *title, size_t rows, size_t columns) {
  if (rows == 0 || columns == 0 || rows > SIZE_MAX / columns)
    return NULL;
  const size_t count = rows * columns;
  if (count > SIZE_MAX / sizeof(Plot *))
    return NULL;

  PlotMulti *multi = calloc(1, sizeof *multi);
  if (multi == NULL)
    return NULL;
  multi->title = copy_string(title);
  multi->plots = calloc(count, sizeof *multi->plots);
  if (multi->title == NULL || multi->plots == NULL) {
    free(multi->title);
    free(multi->plots);
    free(multi);
    return NULL;
  }
  multi->rows = rows;
  multi->columns = columns;
  return multi;
}

void plot_multi_destroy(PlotMulti *multi) {
  if (multi == NULL)
    return;
  if (multi->gnuplot != NULL) {
    (void)fflush(multi->gnuplot);
    (void)pclose(multi->gnuplot);
  }
  const size_t count = multi->rows * multi->columns;
  for (size_t i = 0; i < count; i++)
    plot_destroy(multi->plots[i]);
  free(multi->plots);
  free(multi->title);
  free(multi);
}

Plot *plot_multi_add(PlotMulti *multi, size_t row, size_t column,
                     const char *title, const char *x_label,
                     const char *y_label) {
  if (multi == NULL || row >= multi->rows || column >= multi->columns)
    return NULL;
  const size_t index = row * multi->columns + column;
  if (multi->plots[index] != NULL)
    return NULL;
  Plot *plot = plot_create(title, x_label, y_label);
  if (plot == NULL)
    return NULL;
  multi->plots[index] = plot;
  return plot;
}

Plot *plot_multi_get(PlotMulti *multi, size_t row, size_t column) {
  if (multi == NULL || row >= multi->rows || column >= multi->columns)
    return NULL;
  return multi->plots[row * multi->columns + column];
}

static int write_all_multi_data(PlotMulti *multi) {
  const size_t count = multi->rows * multi->columns;
  for (size_t i = 0; i < count; i++) {
    if (multi->plots[i] != NULL && write_all_plot_data(multi->plots[i]) != 0)
      return -1;
  }
  return 0;
}

int plot_multi_show(PlotMulti *multi) {
  if (multi == NULL)
    return -1;
  if (!plot_is_available()) {
    fprintf(stderr,
            "cplot: gnuplot was not found in PATH; install gnuplot or save "
            "the multiplot with plot_multi_save().\n");
    return -1;
  }
  if (write_all_multi_data(multi) != 0 || start_multi_gnuplot(multi) != 0)
    return -1;
  multi->grid_shown = 1;
  return refresh_shared_multi(multi);
}

int plot_multi_show_individual(PlotMulti *multi) {
  if (multi == NULL)
    return -1;
  if (multi->individual_windows_shown) {
    fprintf(stderr,
            "cplot: this multiplot already uses shared-process individual "
            "windows; create a new PlotMulti to use per-window processes.\n");
    return -1;
  }
  if (!plot_is_available()) {
    fprintf(stderr,
            "cplot: gnuplot was not found in PATH; cannot open individual "
            "plot windows.\n");
    return -1;
  }

  /*
   * Each window gets its own gnuplot process. Gnuplot 6.0 keeps one active
   * terminal/plot context per process, so numbered windows from a shared
   * process cannot all provide reliable mouse zoom at once.
   */
  const size_t count = multi->rows * multi->columns;
  int opened_any = 0;
  for (size_t i = 0; i < count; i++) {
    Plot *plot = multi->plots[i];
    if (plot == NULL)
      continue;
    if (plot_show(plot) != 0) {
      fprintf(stderr, "cplot: failed to open subplot at row %zu, column %zu.\n",
              i / multi->columns + 1, i % multi->columns + 1);
      return -1;
    }
    opened_any = 1;
  }
  if (!opened_any) {
    fprintf(stderr, "cplot: multiplot has no populated panels.\n");
    return -1;
  }
  return 0;
}

int plot_multi_show_individual_shared(PlotMulti *multi) {
  if (multi == NULL)
    return -1;
  if (!plot_is_available()) {
    fprintf(stderr,
            "cplot: gnuplot was not found in PATH; cannot open individual "
            "plot windows.\n");
    return -1;
  }
  const size_t count = multi->rows * multi->columns;
  for (size_t i = 0; i < count; i++) {
    if (multi->plots[i] != NULL && multi->plots[i]->gnuplot != NULL) {
      fprintf(stderr,
              "cplot: cannot switch to shared-window mode after a subplot "
              "has already opened its own gnuplot process.\n");
      return -1;
    }
  }
  if (write_all_multi_data(multi) != 0 || start_multi_gnuplot(multi) != 0)
    return -1;
  multi->individual_windows_shown = 1;
  return refresh_shared_multi(multi);
}

int plot_multi_refresh(PlotMulti *multi) {
  if (multi == NULL)
    return -1;

  const size_t count = multi->rows * multi->columns;
  int has_independent_child_windows = 0;
  for (size_t i = 0; i < count; i++) {
    if (multi->plots[i] != NULL && multi->plots[i]->gnuplot != NULL) {
      has_independent_child_windows = 1;
      break;
    }
  }

  /* Match the original API: refresh on a never-shown grid opens its overview. */
  if (multi->gnuplot == NULL && !has_independent_child_windows)
    return plot_multi_show(multi);

  if (multi->gnuplot != NULL) {
    if (!multi->grid_shown && !multi->individual_windows_shown)
      multi->grid_shown = 1;
    if (refresh_shared_multi(multi) != 0)
      return -1;
  } else if (write_all_multi_data(multi) != 0) {
    return -1;
  }

  /* Preserve support for callers that show a child Plot directly. */
  for (size_t i = 0; i < count; i++) {
    Plot *plot = multi->plots[i];
    if (plot != NULL && plot->gnuplot != NULL &&
        send_plot_commands(plot) != 0) {
      fprintf(stderr, "cplot: failed to refresh an independently opened subplot.\n");
      return -1;
    }
  }
  return 0;
}

static int write_plot_contents(FILE *file, const Plot *plot,
                               const char *terminator) {
  if (fprintf(file, "TITLE ") < 0 || write_quoted(file, plot->title) != 0 ||
      fprintf(file, "\nXLABEL ") < 0 ||
      write_quoted(file, plot->x_label) != 0 ||
      fprintf(file, "\nYLABEL ") < 0 ||
      write_quoted(file, plot->y_label) != 0 || fputc('\n', file) == EOF)
    return -1;

  for (size_t i = 0; i < plot->series_length; i++) {
    const PlotSeries *series = plot->series[i];
    if (fprintf(file, "SERIES ") < 0 ||
        write_quoted(file, series->name) != 0 ||
        fprintf(file, " %u %zu\n", (unsigned)series->style,
                series->length) < 0)
      return -1;
    for (size_t j = 0; j < series->length; j++) {
      if (fprintf(file, "%.17g %.17g\n", series->points[j].x,
                  series->points[j].y) < 0)
        return -1;
    }
  }
  return fprintf(file, "%s\n", terminator) < 0 ? -1 : 0;
}

int plot_save(const Plot *plot, const char *path) {
  if (plot == NULL || path == NULL || path[0] == '\0')
    return -1;
  FILE *file = fopen(path, "w");
  if (file == NULL) {
    fprintf(stderr, "cplot: cannot save '%s': %s\n", path, strerror(errno));
    return -1;
  }
  int failed = fprintf(file, "CPLOT 1\n") < 0 ||
               write_plot_contents(file, plot, "END") != 0;
  if (fclose(file) != 0)
    failed = 1;
  if (failed)
    fprintf(stderr, "cplot: error while writing '%s'.\n", path);
  return failed ? -1 : 0;
}

int plot_multi_save(const PlotMulti *multi, const char *path) {
  if (multi == NULL || path == NULL || path[0] == '\0')
    return -1;
  FILE *file = fopen(path, "w");
  if (file == NULL) {
    fprintf(stderr, "cplot: cannot save '%s': %s\n", path, strerror(errno));
    return -1;
  }

  int failed = fprintf(file, "CPLOTMULTI 1\nTITLE ") < 0 ||
               write_quoted(file, multi->title) != 0 ||
               fprintf(file, "\nGRID %zu %zu\n", multi->rows,
                       multi->columns) < 0;
  const size_t count = multi->rows * multi->columns;
  for (size_t i = 0; !failed && i < count; i++) {
    const Plot *plot = multi->plots[i];
    if (plot == NULL) {
      failed = fprintf(file, "EMPTY\n") < 0;
      continue;
    }
    failed = fprintf(file, "PLOT\n") < 0 ||
             write_plot_contents(file, plot, "ENDPLOT") != 0;
  }
  if (!failed && fprintf(file, "END\n") < 0)
    failed = 1;
  if (fclose(file) != 0)
    failed = 1;
  if (failed)
    fprintf(stderr, "cplot: error while writing '%s'.\n", path);
  return failed ? -1 : 0;
}

static char *read_quoted(FILE *file) {
  int c;
  do {
    c = fgetc(file);
  } while (c != EOF && isspace((unsigned char)c));
  if (c != '"')
    return NULL;

  char *value = NULL;
  size_t length = 0;
  size_t capacity = 0;
  for (;;) {
    c = fgetc(file);
    if (c == EOF) {
      free(value);
      return NULL;
    }
    if (c == '"')
      break;
    if (c == '\\') {
      c = fgetc(file);
      if (c == EOF) {
        free(value);
        return NULL;
      }
      switch (c) {
      case 'n':
        c = '\n';
        break;
      case 'r':
        c = '\r';
        break;
      case 't':
        c = '\t';
        break;
      case '\\':
      case '"':
        break;
      default:
        free(value);
        return NULL;
      }
    }
    if (length == SIZE_MAX - 1) {
      free(value);
      return NULL;
    }
    if (length + 1 >= capacity) {
      size_t next_capacity = capacity == 0 ? 32 : capacity * 2;
      if (next_capacity < capacity || next_capacity > SIZE_MAX) {
        free(value);
        return NULL;
      }
      char *next = realloc(value, next_capacity);
      if (next == NULL) {
        free(value);
        return NULL;
      }
      value = next;
      capacity = next_capacity;
    }
    value[length++] = (char)c;
  }
  if (length + 1 > capacity) {
    char *next = realloc(value, length + 1);
    if (next == NULL) {
      free(value);
      return NULL;
    }
    value = next;
  }
  if (value == NULL) {
    value = malloc(1);
    if (value == NULL)
      return NULL;
  }
  value[length] = '\0';
  return value;
}

static Plot *read_plot_contents(FILE *file, const char *terminator,
                                const char *path) {
  char key[32];
  char *title = NULL;
  char *x_label = NULL;
  char *y_label = NULL;
  Plot *plot = NULL;

  if (fscanf(file, "%31s", key) != 1 || strcmp(key, "TITLE") != 0 ||
      (title = read_quoted(file)) == NULL ||
      fscanf(file, "%31s", key) != 1 || strcmp(key, "XLABEL") != 0 ||
      (x_label = read_quoted(file)) == NULL ||
      fscanf(file, "%31s", key) != 1 || strcmp(key, "YLABEL") != 0 ||
      (y_label = read_quoted(file)) == NULL) {
    fprintf(stderr, "cplot: malformed plot metadata in '%s'.\n", path);
    goto cleanup;
  }

  plot = plot_create(title, x_label, y_label);
  if (plot == NULL)
    goto cleanup;

  for (;;) {
    if (fscanf(file, "%31s", key) != 1) {
      fprintf(stderr, "cplot: unexpected end of '%s'.\n", path);
      plot_destroy(plot);
      plot = NULL;
      goto cleanup;
    }
    if (strcmp(key, terminator) == 0)
      break;
    if (strcmp(key, "SERIES") != 0) {
      fprintf(stderr, "cplot: unexpected token in '%s'.\n", path);
      plot_destroy(plot);
      plot = NULL;
      goto cleanup;
    }

    char *name = read_quoted(file);
    unsigned style_value;
    size_t count;
    if (name == NULL || fscanf(file, "%u %zu", &style_value, &count) != 2 ||
        style_value > (unsigned)PLOT_IMPULSES) {
      free(name);
      fprintf(stderr, "cplot: malformed series in '%s'.\n", path);
      plot_destroy(plot);
      plot = NULL;
      goto cleanup;
    }
    PlotSeries *series = plot_add_series(plot, name, (PlotStyle)style_value);
    free(name);
    if (series == NULL) {
      plot_destroy(plot);
      plot = NULL;
      goto cleanup;
    }
    for (size_t i = 0; i < count; i++) {
      double x;
      double y;
      if (fscanf(file, "%lf %lf", &x, &y) != 2 ||
          plot_series_add(plot, series, x, y) != 0) {
        fprintf(stderr, "cplot: invalid point data in '%s'.\n", path);
        plot_destroy(plot);
        plot = NULL;
        goto cleanup;
      }
    }
  }

cleanup:
  free(title);
  free(x_label);
  free(y_label);
  return plot;
}

Plot *plot_load(const char *path) {
  if (path == NULL || path[0] == '\0')
    return NULL;
  FILE *file = fopen(path, "r");
  if (file == NULL) {
    fprintf(stderr, "cplot: cannot open '%s': %s\n", path, strerror(errno));
    return NULL;
  }

  char magic[16];
  unsigned version;
  if (fscanf(file, "%15s %u", magic, &version) != 2 ||
      strcmp(magic, "CPLOT") != 0 || version != 1) {
    fprintf(stderr, "cplot: '%s' is not a supported cplot state file.\n", path);
    (void)fclose(file);
    return NULL;
  }

  Plot *plot = read_plot_contents(file, "END", path);
  (void)fclose(file);
  return plot;
}

PlotMulti *plot_multi_load(const char *path) {
  if (path == NULL || path[0] == '\0')
    return NULL;
  FILE *file = fopen(path, "r");
  if (file == NULL) {
    fprintf(stderr, "cplot: cannot open '%s': %s\n", path, strerror(errno));
    return NULL;
  }

  char magic[16];
  char key[32];
  unsigned version;
  char *title = NULL;
  PlotMulti *multi = NULL;
  if (fscanf(file, "%15s %u", magic, &version) != 2 ||
      strcmp(magic, "CPLOTMULTI") != 0 || version != 1 ||
      fscanf(file, "%31s", key) != 1 || strcmp(key, "TITLE") != 0 ||
      (title = read_quoted(file)) == NULL) {
    fprintf(stderr, "cplot: '%s' is not a supported multiplot state file.\n",
            path);
    goto cleanup;
  }

  size_t rows;
  size_t columns;
  if (fscanf(file, "%31s %zu %zu", key, &rows, &columns) != 3 ||
      strcmp(key, "GRID") != 0) {
    fprintf(stderr, "cplot: malformed multiplot grid in '%s'.\n", path);
    goto cleanup;
  }
  multi = plot_multi_create(title, rows, columns);
  if (multi == NULL) {
    fprintf(stderr, "cplot: invalid grid or allocation failure in '%s'.\n",
            path);
    goto cleanup;
  }

  const size_t count = rows * columns;
  for (size_t i = 0; i < count; i++) {
    if (fscanf(file, "%31s", key) != 1) {
      fprintf(stderr, "cplot: unexpected end of '%s'.\n", path);
      plot_multi_destroy(multi);
      multi = NULL;
      goto cleanup;
    }
    if (strcmp(key, "EMPTY") == 0)
      continue;
    if (strcmp(key, "PLOT") != 0) {
      fprintf(stderr, "cplot: unexpected cell token in '%s'.\n", path);
      plot_multi_destroy(multi);
      multi = NULL;
      goto cleanup;
    }
    multi->plots[i] = read_plot_contents(file, "ENDPLOT", path);
    if (multi->plots[i] == NULL) {
      plot_multi_destroy(multi);
      multi = NULL;
      goto cleanup;
    }
  }

  if (fscanf(file, "%31s", key) != 1 || strcmp(key, "END") != 0) {
    fprintf(stderr, "cplot: missing multiplot terminator in '%s'.\n", path);
    plot_multi_destroy(multi);
    multi = NULL;
  }

cleanup:
  free(title);
  (void)fclose(file);
  return multi;
}
