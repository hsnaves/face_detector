
#ifndef __CASCADE_H
#define __CASCADE_H

#include "features.h"
#include "image.h"
#include "window.h"

/* Data structures */
typedef
struct classifier_st {
	struct classifier_st *next;
	feature_index fi;
	feature_index_opt fo;
	double coef, intercept, thresh;
} classifier;

typedef
struct cascade_stage_st {
	struct cascade_stage_st *next, *prev;
	unsigned int *num_classifiers;
	classifier **cl;
	double *intercept;
} cascade_stage;

typedef
struct detected_object_st {
	window w;
	window comp;
	unsigned int sel_parallel;
	double *score;
} detected_object;

typedef
struct cascade_st {
	unsigned int num_stages;
	unsigned int num_parallels;
	cascade_stage *st, *lst;

	int multi_exit;
	unsigned int width, height;
	unsigned int step;
	double scale, min_stddev;
	double match_thresh, overlap_thresh;
	const image *src;
	image img;
	features f;

	unsigned int min_width, min_height;
	unsigned int max_width, max_height;
	unsigned int pyramid_min, pyramid_max;

	classifier *clfree, *clalloc;
	cascade_stage *stfree, *stalloc;

	detected_object *detected_objects;
	double *scores;
	unsigned int num_detected_objects;
	unsigned int num_jumbled_objects;
	unsigned int capacity_objects;
} cascade;

/* Functions */
void cascade_reset(cascade *c);
int cascade_init(cascade *c, unsigned int width, unsigned int height,
                 unsigned int num_parallels);
void cascade_cleanup(cascade *c);

void cascade_get_params(const cascade *c, double *scale, double *min_stddev,
                        unsigned int *step, double *match_thresh,
                        double *overlap_thresh, int *multi_exit);
void cascade_set_params(cascade *c, double scale, double min_stddev,
                        unsigned int step, double match_thresh,
                        double overlap_thresh, int multi_exit);
void cascade_set_scan(cascade *c,
                      unsigned int min_width, unsigned int min_height,
                      unsigned int max_width, unsigned int max_height);

int cascade_overlap(const cascade *c, const window *w1, const window *w2);
void cascade_clear(cascade *c);
void cascade_remove_last_stage(cascade *c);
void cascade_consolidate_stage(cascade *c, cascade_stage *st);

int cascade_copy(const cascade *from, cascade *to);
unsigned int cascade_max_classifiers(const cascade *c, unsigned int parallel);

classifier *cascade_new_classifier(cascade *c, cascade_stage *st,
                                   unsigned int parallel);
cascade_stage *cascade_new_stage(cascade *c);

int cascade_set_image(cascade *c, const image *img);
void cascade_separate(cascade *c, unsigned int offset);
int cascade_detect(cascade *c, int separate_detected);
void cascade_real_window(const cascade *c, const window *comp, window *w);
int cascade_extract(cascade *c, const window *comp, sval *sat);

int cascade_load(cascade *c, const char *filename, int reset);
int cascade_save(const cascade *c, const char *filename);

#endif /* __CASCADE_H */

