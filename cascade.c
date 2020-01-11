
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cascade.h"
#include "features.h"
#include "image.h"
#include "window.h"
#include "utils.h"

#define ALLOC_NUM                512
#define DETECTED_ALLOC_NUM      8192

void cascade_reset(cascade *c)
{
	c->st = NULL;
	c->lst = NULL;
	c->stalloc = NULL;
	c->stfree = NULL;
	c->clalloc = NULL;
	c->clfree = NULL;
	c->detected_objects = NULL;
	c->scores = NULL;

	image_reset(&c->img);
	features_reset(&c->f);
}

int cascade_init(cascade *c, unsigned int width, unsigned int height,
                 unsigned int num_parallels)
{
	unsigned int i;
	detected_object *objs;
	size_t size;

	cascade_reset(c);
	c->num_stages = 0;

	features_init(&c->f);
	image_init(&c->img);

	size = DETECTED_ALLOC_NUM * sizeof(detected_object);
	objs = (detected_object *) xmalloc(size);
	if (!objs) goto error_init;

	c->detected_objects = objs;
	c->capacity_objects = DETECTED_ALLOC_NUM;
	c->num_detected_objects = 0;

	c->num_parallels = num_parallels;

	size = DETECTED_ALLOC_NUM * num_parallels * sizeof(double);
	c->scores = (double *) xmalloc(size);
	if (!c->scores) goto error_init;

	for (i = 0; i < c->capacity_objects; i++) {
		objs[i].score = &c->scores[i * num_parallels];
	}

	c->width = width;
	c->height = height;
	c->min_width = width;
	c->min_height = height;
	c->max_width = 0;
	c->max_height = 0;

	return TRUE;

error_init:
	cascade_cleanup(c);
	return FALSE;
}

void cascade_cleanup(cascade *c)
{
	features_cleanup(&c->f);
	image_cleanup(&c->img);

	if (c->detected_objects) {
		free(c->detected_objects);
		c->detected_objects = NULL;
	}

	if (c->scores) {
		free(c->scores);
		c->scores = NULL;
	}

	while (c->clalloc) {
		classifier *cl = c->clalloc;
		c->clalloc = cl->next;
		free(cl);
	}

	while (c->stalloc) {
		cascade_stage *st = c->stalloc;
		free(st->num_classifiers);
		free(st->intercept);
		free(st->cl);
		c->stalloc = st->next;
		free(st);
	}
}

void cascade_get_params(const cascade *c, double *scale, double *min_stddev,
                        unsigned int *step, double *match_thresh,
                        double *overlap_thresh, int *multi_exit)
{
	*scale = c->scale;
	*min_stddev = c->min_stddev;
	*step = c->step;
	*match_thresh = c->match_thresh;
	*overlap_thresh = c->overlap_thresh;
	*multi_exit =c->multi_exit;
}

void cascade_set_params(cascade *c, double scale, double min_stddev,
                        unsigned int step, double match_thresh,
                        double overlap_thresh, int multi_exit)
{
	c->scale = scale;
	c->min_stddev = min_stddev;
	c->step = step;
	c->match_thresh = match_thresh;
	c->overlap_thresh = overlap_thresh;
	c->multi_exit = multi_exit;
}

void cascade_set_scan(cascade *c,
                      unsigned int min_width, unsigned int min_height,
                      unsigned int max_width, unsigned int max_height)
{
	c->min_width = MAX(min_width, c->width);
	c->min_height = MAX(min_height, c->height);
	c->max_width = max_width;
	c->max_height = max_height;
}

int cascade_overlap(const cascade *c, const window *w1, const window *w2)
{
	return window_overlap(w1, w2, c->match_thresh, c->overlap_thresh);
}

static
void cascade_free_stage(cascade *c, cascade_stage *st)
{
	unsigned int k;

	for (k = 0; k < c->num_parallels; k++) {
		while (st->cl[k]) {
			classifier *cl = st->cl[k];
			st->cl[k] = cl->next;
			cl->next = c->clfree;
			c->clfree = cl;
		}
	}

	if (st->next)
		st->next->prev = st->prev;
	else
		c->lst = st->prev;

	if (st->prev)
		st->prev->next = st->next;
	else
		c->st = st->next;

	st->next = c->stfree;
	c->stfree = st;
	c->num_stages--;
}

void cascade_clear(cascade *c)
{
	while (c->st) {
		cascade_free_stage(c, c->st);
	}

	c->st = NULL;
	c->lst = NULL;
	c->num_stages = 0;
}

void cascade_remove_last_stage(cascade *c)
{
	if (c->lst)
		cascade_free_stage(c, c->lst);
}

void cascade_consolidate_stage(cascade *c, cascade_stage *st)
{
	classifier *cl;
	unsigned int k;

	for (k = 0; k < c->num_parallels; k++) {
		st->intercept[k] = 0;
		for (cl = st->cl[k]; cl; cl = cl->next)
			st->intercept[k] += cl->intercept;
	}
}

int cascade_copy(const cascade *from, cascade *to)
{
	cascade_stage *st, *nst;
	classifier *cl, *ncl;
	unsigned int k;

	cascade_clear(to);
	cascade_set_params(to, from->scale, from->min_stddev, from->step,
	                   from->match_thresh, from->overlap_thresh,
	                   from->multi_exit);
	cascade_set_scan(to, from->min_width, from->min_height,
	                 from->max_width, from->max_height);

	for (st = from->st; st; st = st->next) {
		nst = cascade_new_stage(to);
		if (!nst) return FALSE;

		for (k = 0; k < from->num_parallels; k++) {
			for (cl = st->cl[k]; cl; cl = cl->next) {
				ncl = cascade_new_classifier(to, nst, k);
				if (!ncl) return FALSE;
				ncl->fi = cl->fi;
				ncl->coef = cl->coef;
				ncl->thresh = cl->thresh;
				ncl->intercept = cl->intercept;
			}
		}
		cascade_consolidate_stage(to, nst);
	}

	return TRUE;
}

unsigned int cascade_max_classifiers(const cascade *c, unsigned int parallel)
{
	unsigned int max_classifiers = 0, num_classifiers;
	cascade_stage *st;

	for (st = c->st; st; st = st->next) {
		num_classifiers = st->num_classifiers[parallel];
		if (max_classifiers < num_classifiers)
			max_classifiers = num_classifiers;
	}
	return max_classifiers;
}

classifier *cascade_new_classifier(cascade *c, cascade_stage *st,
                                   unsigned int parallel)
{
	classifier *cl;
	if (!c->clfree) {
		unsigned int i;

		cl = (classifier *) xmalloc(ALLOC_NUM * sizeof(classifier));
		if (!cl) return NULL;

		cl->next = c->clalloc;
		c->clalloc = cl;
		for (i = 1; i < ALLOC_NUM; i++) {
			cl[i].next = &cl[i + 1];
		}
		cl[i - 1].next = NULL;
		c->clfree = &cl[1];
	}
	cl = c->clfree;
	c->clfree = cl->next;
	cl->next = st->cl[parallel];
	st->cl[parallel] = cl;
	st->num_classifiers[parallel]++;
	return cl;
}

cascade_stage *cascade_new_stage(cascade *c)
{
	unsigned int k;
	cascade_stage *st;

	if (!c->stfree) {
		unsigned int i;
		unsigned int *num_classifiers;
		classifier **ptrs;
		double *intercept;
		size_t size;

		size = ALLOC_NUM * sizeof(cascade_stage);
		st = (cascade_stage *) xmalloc(size);
		if (!st) return NULL;

		size = ALLOC_NUM * c->num_parallels * sizeof(unsigned int);
		num_classifiers = (unsigned int *) xmalloc(size);
		if (!num_classifiers) {
			free(st);
			return NULL;
		}

		size = ALLOC_NUM * c->num_parallels * sizeof(double);
		intercept = (double *) xmalloc(size);
		if (!intercept) {
			free(st);
			free(num_classifiers);
			return NULL;
		}

		size = ALLOC_NUM * c->num_parallels * sizeof(classifier *);
		ptrs = (classifier **) xmalloc(size);

		if (!ptrs) {
			free(st);
			free(num_classifiers);
			free(intercept);
			return NULL;
		}

		st->num_classifiers = num_classifiers;
		st->cl = ptrs;
		st->intercept = intercept;

		st->next = c->stalloc;
		c->stalloc = st;
		for (i = 1; i < ALLOC_NUM; i++) {
			st[i].num_classifiers =
			    &num_classifiers[i * c->num_parallels];
			st[i].intercept = &intercept[i * c->num_parallels];
			st[i].cl = &ptrs[i * c->num_parallels];
			st[i].next = &st[i + 1];
		}
		st[i - 1].next = NULL;
		c->stfree = &st[1];
	}
	st = c->stfree;
	c->stfree = st->next;

	c->num_stages++;
	st->next = NULL;
	for (k = 0; k < c->num_parallels; k++) {
		st->num_classifiers[k] = 0;
		st->intercept[k] = 0;
		st->cl[k] = NULL;
	}

	if (c->lst) {
		c->lst->next = st;
		st->prev = c->lst;
		c->lst = st;
	} else {
		c->st = st;
		c->lst = st;
		st->prev = NULL;
	}

	return st;
}

int cascade_set_image(cascade *c, const image *img)
{
	unsigned int max_width, max_height;
	unsigned int pyramid_min, pyramid_max;
	double width, height;

	c->src = img;
	if (c->max_width == 0)
		max_width = img->width;
	else
		max_width = MIN(c->max_width, img->width);

	if (c->max_height == 0)
		max_height = img->height;
	else
		max_height = MIN(c->max_height, img->height);

	width = c->width;
	height = c->height;

	pyramid_min = 0;
	while (width < c->min_width || height < c->min_height) {
		width *= c->scale;
		height *= c->scale;
		pyramid_min++;
	}

	pyramid_max = pyramid_min;
	while (width <= max_width && height <= max_height) {
		width *= c->scale;
		height *= c->scale;
		pyramid_max++;
	}

	c->pyramid_min = pyramid_min;
	c->pyramid_max = pyramid_max;
	return TRUE;
}

static
void cascade_precomp(cascade *c, unsigned int stride)
{
	cascade_stage *st;
	classifier *cl;

	for (st = c->st; st; st = st->next) {
		unsigned int k;
		for (k = 0; k < c->num_parallels; k++) {
			for (cl = st->cl[k]; cl; cl = cl->next) {
				features_optimize(&cl->fi, &cl->fo, stride);
			}
		}
	}
}

static
double cascade_evaluate1(cascade *c, unsigned int offset,
                         unsigned int stride, double factor)
{
	cascade_stage *st;
	classifier *cl;
	double val, t;
	detected_object *obj;

	obj = &c->detected_objects[c->num_jumbled_objects];
	obj->sel_parallel = 0;
	val = 0;

	for (st = c->st; st; st = st->next) {

		if (c->multi_exit)
			val += st->intercept[0];
		else
			val = st->intercept[0];

		for (cl = st->cl[0]; cl; cl = cl->next) {
			t = features_evaluate_fast(&c->f.sat[offset],
			                           &cl->fo);
			if (t >= factor * cl->thresh)
				val += cl->coef;
		}
		if (val < 0) {
			return val;
		}
	}
	obj->score[0] = val;
	return val;
}

static
double cascade_evaluate(cascade *c, unsigned int offset,
                        unsigned int stride, double factor)
{
	cascade_stage *st;
	classifier *cl;
	double val, t;
	detected_object *obj;
	double *score;
	unsigned int k, sel;

	if (c->num_parallels == 1)
		return cascade_evaluate1(c, offset, stride, factor);

	obj = &c->detected_objects[c->num_jumbled_objects];
	score = obj->score;
	for (k = 0; k < c->num_parallels; k++)
		score[k] = 0;

	for (st = c->st; st; st = st->next) {
		sel = 0;
		for (k = 0; k < c->num_parallels; k++) {

			if (c->multi_exit)
				score[k] += st->intercept[k];
			else
				score[k] = st->intercept[k];

			for (cl = st->cl[k]; cl; cl = cl->next) {
				t = features_evaluate_fast(&c->f.sat[offset],
				                           &cl->fo);
				if (t >= factor * cl->thresh)
					score[k] += cl->coef;
			}
			if (score[k] > score[sel])
				sel = k;
		}
		obj->sel_parallel = sel;
		if (score[sel] < 0) {
			return score[sel];
		}
	}
	val = score[obj->sel_parallel];
	return val;
}

static
int new_object(cascade *c, const window *comp)
{
	detected_object *obj;

	if (c->num_jumbled_objects >= c->capacity_objects - 1) {
		detected_object *new_objs;
		size_t size;
		unsigned int i;
		double *new_scores;

		size = 2 * c->capacity_objects * sizeof(detected_object);
		new_objs = xmalloc(size);
		if (!new_objs) return FALSE;

		size = 2 * c->capacity_objects * c->num_parallels
		         * sizeof(double);
		new_scores = xmalloc(size);
		if (!new_scores) {
			free(new_objs);
			return FALSE;
		}

		size = c->capacity_objects * c->num_parallels
		       * sizeof(double);
		memcpy(new_scores, c->scores, size);

		for (i = 0; i < c->capacity_objects; i++) {
			detected_object *obj;
			unsigned int idx;

			obj = &c->detected_objects[i];
			idx = (unsigned int) (obj->score - c->scores);
			obj->score = &new_scores[idx];
		}

		for (; i < 2 * c->capacity_objects; i++) {
			new_objs[i].score = &new_scores[i * c->num_parallels];
		}

		free(c->scores);
		c->scores = new_scores;

		size = c->capacity_objects * sizeof(detected_object);
		memcpy(new_objs, c->detected_objects, size);
		free(c->detected_objects);

		c->capacity_objects *= 2;
		c->detected_objects = new_objs;
	}

	obj = &c->detected_objects[c->num_jumbled_objects++];
	obj->comp = *comp;
	cascade_real_window(c, &obj->comp, &obj->w);
	return TRUE;
}

static
int cmp_objects(const void *ptr1, const void *ptr2)
{
	const detected_object *o1 = (const detected_object *) ptr1;
	const detected_object *o2 = (const detected_object *) ptr2;
	double score1, score2;
	score1 = o1->score[o1->sel_parallel];
	score2 = o2->score[o2->sel_parallel];
	if (score1 < score2) return +1;
	if (score1 > score2) return -1;
	return 0;
}

void cascade_separate(cascade *c, unsigned int offset)
{
	unsigned int i, j, l;
	detected_object *objs;

	if (c->num_jumbled_objects == 0)
		return;

	if (offset < c->num_jumbled_objects) {
		qsort(&c->detected_objects[offset],
		      c->num_jumbled_objects - offset,
		      sizeof(detected_object), &cmp_objects);
	}

	j = 1;
	objs = c->detected_objects;
	for (i = 1; i < c->num_jumbled_objects; i++) {
		for (l = 0; l < j; l++) {
			if (window_overlap(&objs[l].w, &objs[i].w,
			                   c->match_thresh,
			                   c->overlap_thresh))
				break;
		}
		if (l < j) continue;

		if (i != j) {
			detected_object temp = objs[j];
			objs[j] = objs[i];
			objs[i] = temp;
		}
		j++;
	}
	c->num_detected_objects = j;
}

int cascade_detect(cascade *c, int separate_detected)
{
	unsigned int i, offset, istep;
	double score, stddev, factor;
	double step, width, height;
	window comp, inner;
	int error;

	c->num_detected_objects = 0;
	c->num_jumbled_objects = 0;
	comp.width = c->src->width;
	comp.height = c->src->height;
	inner.width = c->width;
	inner.height = c->height;

	istep = c->step;

	width = comp.width;
	height = comp.height;
	step = istep;

	for (i = 0; i < c->pyramid_min; i++) {
		step /= c->scale;
		width /= c->scale;
		height /= c->scale;
	}

	error = FALSE;
	for (; i < c->pyramid_max; i++) {
		comp.width = (unsigned int) floor(0.5 + width);
		comp.height = (unsigned int) floor(0.5 + height);
		istep = (unsigned int) ceil(step);

		step /= c->scale;
		width /= c->scale;
		height /= c->scale;

		if (!image_resize(c->src, &c->img,
		                  comp.width, comp.height))
			return FALSE;
		if (!features_precompute(&c->f, &c->img))
			return FALSE;

		cascade_precomp(c, c->f.stride);
		comp.top = 0;
		while (comp.top <= comp.height - c->height) {
			comp.left = 0;
			while (comp.left <= comp.width - c->width) {
				inner.left = comp.left;
				inner.top = comp.top;
				stddev = features_stddev(&c->f, &inner);
				if (stddev <= c->min_stddev) {
					comp.left += istep;
					continue;
				}

				factor = stddev;
				offset = inner.top * c->f.stride + inner.left;
				score = cascade_evaluate(c, offset,
				                         c->f.stride, factor);
				if (score >= 0.0) {
					{
						if (!new_object(c, &comp))
							error = TRUE;
					}
				}
				comp.left += istep;
			}
			comp.top += istep;
		}
	}

	if (error) return FALSE;
	if (separate_detected)
		cascade_separate(c, 0);
	return TRUE;
}

void cascade_real_window(const cascade *c, const window *comp, window *w)
{
	double factor;
	factor = ((double) c->src->width) / comp->width;
	w->left = (unsigned int) (comp->left * factor);
	w->width = (unsigned int) (c->width * factor);
	w->top = (unsigned int) (comp->top * factor);
	w->height = (unsigned int) (c->height * factor);
}

int cascade_extract(cascade *c, const window *comp, sval *sat)
{
	double stddev;
	window aux;

	if (!image_resize(c->src, &c->img, comp->width, comp->height))
		return FALSE;

	if (!features_precompute(&c->f, &c->img))
		return FALSE;

	aux.left = comp->left;
	aux.top = comp->top;
	aux.width = c->width;
	aux.height = c->height;
	stddev = features_stddev(&c->f, &aux);
	features_crop(&c->f, &aux, 1.0 / stddev, sat, c->width + 1);
	return TRUE;
}

int cascade_load(cascade *c, const char *filename, int reset)
{
	unsigned int width, height;
	unsigned int num_stages, num_classifiers;
	unsigned int i, j;
	unsigned int k, num_parallels;
	cascade_stage *st;
	classifier *cl;
	FILE *fp = NULL;

	if (reset)
		cascade_reset(c);

	fp = fopen(filename, "r");
	if (!fp) {
		error("can't open `%s' for reading", filename);
		return FALSE;
	}

	if (fscanf(fp, "%u %u", &width, &height) != 2)
		goto error_parse;

	if (fscanf(fp, "%u", &num_parallels) != 1)
		goto error_parse;

	if (reset) {
		if (!cascade_init(c, width, height, num_parallels))
			goto error_load;
	} else {
		if (c->width != width || c->height != height
		    || c->num_parallels != num_parallels) {
			error("wrong parameters for cascade load");
			goto error_load;
		}
		cascade_clear(c);
	}

	if (fscanf(fp, "%lg %lg %u %lg %lg", &c->scale, &c->min_stddev,
	           &c->step, &c->match_thresh, &c->overlap_thresh) != 5)
		goto error_parse;

	if (fscanf(fp, "%d", &c->multi_exit) != 1)
		goto error_parse;

	if (fscanf(fp, "%u", &num_stages) != 1)
		goto error_parse;

	for (i = 0; i < num_stages; i++) {
		st = cascade_new_stage(c);
		if (!st) goto error_load;

		for (k = 0; k < c->num_parallels; k++) {
			if (fscanf(fp, "%u", &num_classifiers) != 1)
				goto error_parse;

			for (j = 0; j < num_classifiers; j++) {
				cl = cascade_new_classifier(c, st, k);
				if (!cl) goto error_load;

				if (fscanf(fp, "%lg %lg %lg %d %u %u %u %u",
				           &cl->coef, &cl->thresh,
				           &cl->intercept,  &cl->fi.idx,
				           &cl->fi.w.left, &cl->fi.w.top,
				           &cl->fi.w.width, &cl->fi.w.height)
				    != 8) goto error_parse;
			}
		}
		cascade_consolidate_stage(c, st);
	}

	fclose(fp);
	return TRUE;

error_parse:
	error("can't parse `%s'", filename);

error_load:
	if (fp) fclose(fp);
	if (reset) cascade_cleanup(c);
	return FALSE;
}

int cascade_save(const cascade *c, const char *filename)
{
	cascade_stage *st;
	classifier *cl;
	unsigned int k;
	FILE *fp;

	fp = fopen(filename, "w");
	if (!fp) {
		error("can't open `%s' for writing", filename);
		return FALSE;
	}
	fprintf(fp, "%u %u %u\n", c->width, c->height, c->num_parallels);
	fprintf(fp, "%g %g %u %g %g\n", c->scale, c->min_stddev, c->step,
	        c->match_thresh, c->overlap_thresh);
	fprintf(fp, "%d\n", c->multi_exit);

	fprintf(fp, "%u\n", c->num_stages);
	for (st = c->st; st; st = st->next) {
		for (k = 0; k < c->num_parallels; k++) {
			fprintf(fp, "%u\n", st->num_classifiers[k]);
			for (cl = st->cl[k]; cl; cl = cl->next) {
				fprintf(fp, "%g %g %g %u %u %u %u %u\n",
				        cl->coef, cl->thresh, cl->intercept,
				        cl->fi.idx, cl->fi.w.left,
				        cl->fi.w.top, cl->fi.w.width,
				        cl->fi.w.height);
			}
		}
	}
	fclose(fp);
	return TRUE;
}

