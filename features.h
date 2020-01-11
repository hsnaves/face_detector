
#ifndef __FEATURES_H
#define __FEATURES_H

#include "image.h"
#include "window.h"

#define MAX_OPT_POINTS    12

/* Data structures and types */
#ifdef SVAL_DOUBLE
typedef double sval;
#else
typedef int sval;
#endif

typedef
struct features_st {
	unsigned int width, height, stride;
	unsigned int capacity;
	unsigned int capacity_hog;
	unsigned int nbins;
	sval *sat, *sat2;
	sval *hog;
} features;

typedef
struct feature_index_st {
	int idx;
	window w;
} feature_index;

typedef
struct feature_index_opt_st {
	unsigned int num_opt_points;
	unsigned int point[MAX_OPT_POINTS];
	sval weight[MAX_OPT_POINTS];
} feature_index_opt;

typedef
struct feature_enumerator_st {
	feature_index fi;
	window max_w;
	unsigned int count;
	unsigned int width, height;
	unsigned int mx, my;
	int use_hog;
} feature_enumerator;

typedef void (*features_rect_cb)(void *arg, sval weight,
                                 unsigned int left, unsigned int top,
                                 unsigned int width, unsigned int height);

typedef void (*features_sat_cb)(void *arg, sval weight, unsigned int pos);

/* Functions */
void features_reset(features *f);
void features_init(features *f);
void features_cleanup(features *f);
int features_precompute(features *f, const image *img);
int features_precompute_hog(features *f, const image *img,
                            unsigned int nbins);

double features_stddev(const features *f, const window *w);
void features_crop(const features *f, const window *w, double alpha,
                   sval *sat, unsigned int stride);
void features_crop_hog(const features *f, const window *w,
                       sval *hog, unsigned int stride);
double features_evaluate(const sval *sat, unsigned int stride,
                         const feature_index *fi);
void features_evaluate_hog(const sval *hog, sval *out,
                           unsigned int stride, unsigned int nbins,
                           const feature_index *fi);

int features_emit_rectangle(const feature_index *fi, features_rect_cb cb,
                            void *arg);
int features_emit_sat(const feature_index *fi, unsigned int stride,
                      features_sat_cb cb, void *arg);
int features_optimize(const feature_index *fi, feature_index_opt *fo,
                      unsigned int stride);

double features_evaluate_fast(const sval *sat, const feature_index_opt *fo);

void feature_enumerator_start(feature_enumerator *fe, unsigned int width,
                              unsigned int height, int use_hog);
int feature_enumerator_next(feature_enumerator *fe);
int feature_enumerator_advance(feature_enumerator *fe, unsigned int count);
unsigned int feature_enumerator_count(feature_enumerator *fe);

#endif /* __FEATURES_H */
