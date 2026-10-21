#ifndef CPLOT_H
#define CPLOT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Plot Plot;
typedef struct PlotSeries PlotSeries;

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

#ifdef __cplusplus
}
#endif

#endif
