#ifndef CPLOT_H
#define CPLOT_H

#include <stddef.h>

#if defined(__GNUC__) || defined(__clang__)
#define CPLOT_PRINTF_FORMAT(format_index, first_argument) \
  __attribute__((format(printf, format_index, first_argument)))
#else
#define CPLOT_PRINTF_FORMAT(format_index, first_argument)
#endif

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

typedef enum {
  PLOT_TEXT_LEFT = 0,
  PLOT_TEXT_CENTER,
  PLOT_TEXT_RIGHT
} PlotTextAlign;

/* Fixed, intent-based annotation placements. MANUAL is used only with the
 * coordinate-taking plot_set_annotation() / plot_set_annotation_atf() APIs. */
typedef enum {
  PLOT_ANNOTATION_MANUAL = 0,
  PLOT_ANNOTATION_RIGHT_BOTTOM,
  PLOT_ANNOTATION_RIGHT_BELOW_LEGEND,
  PLOT_ANNOTATION_LEFT_BOTTOM,
  PLOT_ANNOTATION_LEFT_BELOW_LEGEND,
  PLOT_ANNOTATION_ABOVE_LEFT,
  PLOT_ANNOTATION_ABOVE_CENTER,
  PLOT_ANNOTATION_ABOVE_RIGHT,
  PLOT_ANNOTATION_BELOW_LEFT,
  PLOT_ANNOTATION_BELOW_CENTER,
  PLOT_ANNOTATION_BELOW_RIGHT
} PlotAnnotationPosition;

typedef enum {
  /* Preserve contextual default: outside-right for standalone windows,
   * inside top-right for a multiplot grid. */
  PLOT_LEGEND_DEFAULT = 0,
  PLOT_LEGEND_HIDDEN,
  PLOT_LEGEND_TOP_RIGHT,
  PLOT_LEGEND_TOP_LEFT,
  PLOT_LEGEND_BOTTOM_RIGHT,
  PLOT_LEGEND_BOTTOM_LEFT,
  PLOT_LEGEND_OUTSIDE_RIGHT,
  PLOT_LEGEND_OUTSIDE_LEFT,
  PLOT_LEGEND_OUTSIDE_TOP,
  PLOT_LEGEND_OUTSIDE_BOTTOM
} PlotLegendPosition;

/* The library itself does not require gnuplot to be installed. */
Plot *plot_create(const char *title, const char *x_label, const char *y_label);
void plot_destroy(Plot *plot);

PlotSeries *plot_add_series(Plot *plot, const char *name, PlotStyle style);
/* printf-style variant; formatted storage is owned directly by the series. */
PlotSeries *plot_add_seriesf(Plot *plot, PlotStyle style,
                             const char *format, ...)
    CPLOT_PRINTF_FORMAT(3, 4);
int plot_series_add(Plot *plot, PlotSeries *series, double x, double y);
int plot_series_set(Plot *plot, PlotSeries *series, size_t index, double x,
                    double y);
int plot_series_truncate(Plot *plot, PlotSeries *series, size_t length);
void plot_series_clear(Plot *plot, PlotSeries *series);
size_t plot_series_size(const PlotSeries *series);

/* Returns nan if no series or index in the series exists. */
double plot_series_get_x(Plot *plot, PlotSeries *series, size_t index);
double plot_series_get_y(Plot *plot, PlotSeries *series, size_t index);

void plot_set_title(Plot *plot, const char *title);
void plot_set_x_label(Plot *plot, const char *label);
void plot_set_y_label(Plot *plot, const char *label);
int plot_set_titlef(Plot *plot, const char *format, ...)
    CPLOT_PRINTF_FORMAT(2, 3);
int plot_set_x_labelf(Plot *plot, const char *format, ...)
    CPLOT_PRINTF_FORMAT(2, 3);
int plot_set_y_labelf(Plot *plot, const char *format, ...)
    CPLOT_PRINTF_FORMAT(2, 3);

/* Manual placement, kept for cases where exact graph coordinates are wanted. */
int plot_set_annotation(Plot *plot, const char *text, double x, double y,
                        PlotTextAlign align);
int plot_set_annotation_atf(Plot *plot, double x, double y,
                            PlotTextAlign align, const char *format, ...)
    CPLOT_PRINTF_FORMAT(5, 6);
/* Intent-based placement lets the library reserve margins and align the note. */
int plot_set_annotation_position(Plot *plot, const char *text,
                                 PlotAnnotationPosition position);
int plot_set_annotationf(Plot *plot, PlotAnnotationPosition position,
                         const char *format, ...)
    CPLOT_PRINTF_FORMAT(3, 4);

/* Hide or position the legend. Changes take effect on the next refresh. */
int plot_set_legend_position(Plot *plot, PlotLegendPosition position);

/* Returns 1 when gnuplot can be found in PATH, otherwise 0. */
int plot_is_available(void);

/* Opens a persistent gnuplot window. plot_refresh() redraws an open window. */
int plot_show(Plot *plot);
int plot_refresh(Plot *plot);

/* Saves/loads all plot metadata, series, styles, and data points. */
int plot_save(const Plot *plot, const char *path);
Plot *plot_load(const char *path);

/* A PlotMulti owns a fixed grid of independent plots. Row/column indices are
 * zero-based; child plots must not be destroyed individually. */
PlotMulti *plot_multi_create(const char *title, size_t rows, size_t columns);
void plot_multi_destroy(PlotMulti *multi);
void plot_multi_set_title(PlotMulti *multi, const char *title);
int plot_multi_set_titlef(PlotMulti *multi, const char *format, ...)
    CPLOT_PRINTF_FORMAT(2, 3);
Plot *plot_multi_add(PlotMulti *multi, size_t row, size_t column,
                     const char *title, const char *x_label,
                     const char *y_label);
Plot *plot_multi_get(PlotMulti *multi, size_t row, size_t column);
int plot_multi_show(PlotMulti *multi);
int plot_multi_show_individual(PlotMulti *multi);
int plot_multi_show_individual_shared(PlotMulti *multi);
int plot_multi_refresh(PlotMulti *multi);
int plot_multi_save(const PlotMulti *multi, const char *path);
PlotMulti *plot_multi_load(const char *path);

#ifdef __cplusplus
}
#endif

#endif
