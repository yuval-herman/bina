#ifndef CPLOT_H
#define CPLOT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Plot Plot;
typedef struct PlotSeries PlotSeries;
typedef struct PlotMulti PlotMulti;

typedef enum {
  PLOT_LINES = 0,
  PLOT_POINTS,
  PLOT_LINESPOINTS,
  PLOT_IMPULSES
} PlotStyle;

/* The library itself does not require gnuplot to be installed. */
Plot *plot_create(const char *title, const char *x_label, const char *y_label);
void plot_destroy(Plot *plot);

PlotSeries *plot_add_series(Plot *plot, const char *name, PlotStyle style);
int plot_series_add(Plot *plot, PlotSeries *series, double x, double y);
int plot_series_set(Plot *plot, PlotSeries *series, size_t index, double x,
                    double y);
int plot_series_truncate(Plot *plot, PlotSeries *series, size_t length);
void plot_series_clear(Plot *plot, PlotSeries *series);
size_t plot_series_size(const PlotSeries *series);

/* Returns nan if no series or index in the series exists */
double plot_series_get_x(Plot *plot, PlotSeries *series, size_t index);
/* Returns nan if no series or index in the series exists */
double plot_series_get_y(Plot *plot, PlotSeries *series, size_t index);

void plot_set_title(Plot *plot, const char *title);
void plot_set_x_label(Plot *plot, const char *label);
void plot_set_y_label(Plot *plot, const char *label);

/* Returns 1 when gnuplot can be found in PATH, otherwise 0. */
int plot_is_available(void);

/* Opens a persistent gnuplot window. plot_refresh() redraws an open window. */
int plot_show(Plot *plot);
int plot_refresh(Plot *plot);

/* Saves/loads all plot metadata, series, styles, and data points. */
int plot_save(const Plot *plot, const char *path);
Plot *plot_load(const char *path);

/*
 * A PlotMulti owns a fixed grid of independent plots. Every cell has its own
 * title, labels, and automatically scaled axes. Row/column indices are
 * zero-based. Plots returned by plot_multi_add/get are owned by the grid and
 * must not be destroyed individually.
 */
PlotMulti *plot_multi_create(const char *title, size_t rows, size_t columns);
void plot_multi_destroy(PlotMulti *multi);
Plot *plot_multi_add(PlotMulti *multi, size_t row, size_t column,
                     const char *title, const char *x_label,
                     const char *y_label);
Plot *plot_multi_get(PlotMulti *multi, size_t row, size_t column);
int plot_multi_show(PlotMulti *multi);
/*
 * Opens each populated subplot in its own gnuplot process/window. This gives
 * every window an independent mouse/zoom context. Qt and X11 windows enable
 * replot-on-resize so the graph uses the resized window's full area.
 * plot_multi_refresh() updates these windows and any separately shown grid.
 */
int plot_multi_show_individual(PlotMulti *multi);
/*
 * Alternative: open numbered windows in one shared gnuplot process. This is
 * lighter-weight, but gnuplot 6.0 has one active mouse/zoom plot context per
 * process, so independent mouse zoom is not reliable in every window.
 */
int plot_multi_show_individual_shared(PlotMulti *multi);
int plot_multi_refresh(PlotMulti *multi);
int plot_multi_save(const PlotMulti *multi, const char *path);
PlotMulti *plot_multi_load(const char *path);

#ifdef __cplusplus
}
#endif

#endif
