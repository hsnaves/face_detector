
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "trainer.h"
#include "boosting.h"
#include "cpa.h"
#include "detector.h"
#include "cascade.h"
#include "samples.h"
#include "image.h"
#include "features.h"
#include "window.h"
#include "thread_pool.h"
#include "stopwatch.h"
#include "random.h"
#include "utils.h"

#define MAX_ITERATIONS       100

void trainer_reset(trainer_data *td)
{
	detector_reset(&td->dt);
	samples_reset(&td->smp);
	image_reset(&td->img);
	thread_pool_reset(&td->tp);
	td->sat_buffer = NULL;
	td->sat = NULL;
	td->y = NULL;
	td->tinfos = NULL;
}

int trainer_init(trainer_data *td, const char *samples_filename,
                 unsigned int width, unsigned int height,
                 unsigned int pos_samples, unsigned int neg_samples,
                 unsigned int num_buckets, unsigned int num_parallels,
                 unsigned int num_threads, unsigned int nbins,
                 unsigned int max_planes)
{
	feature_enumerator fe;
	unsigned int i;
	size_t size;

	trainer_reset(td);
	if (!samples_read(&td->smp, samples_filename))
		goto error_init;

	td->inum_pos = 0;
	td->inum_neg = 0;

	for (i = 0; i < td->smp.num_items; i++) {
		sample_item *item;
		item = &td->smp.items[i];
		if (item->positive) {
			td->inum_pos++;
		} else {
			td->inum_neg++;
		}
	}

	if (neg_samples > 0)
		td->inum_neg = neg_samples;
	if (pos_samples > 0) {
		if (td->inum_pos > pos_samples)
			td->inum_pos = pos_samples;
	}

	td->width = width;
	td->height = height;
	td->nbins = nbins;
	td->max_n = td->inum_pos + td->inum_neg;

	size = td->max_n * (width + 1) * (height + 1) * sizeof(sval);
	td->sat_buffer = (sval *) xmalloc(size);
	if (!td->sat_buffer) goto error_init;

	size = td->max_n * sizeof(sval *);
	td->sat = (sval **) xmalloc(size);
	if (!td->sat) goto error_init;

	size = (width + 1) * (height + 1);
	for (i = 0; i < td->max_n; i++) {
		td->sat[i] = &td->sat_buffer[size * i];
	}

	size = td->max_n * sizeof(int);
	td->y = (int *) xmalloc(size);
	if (!td->y) goto error_init;

	td->num_threads = num_threads;
	size = num_threads * sizeof(trainer_job_info);
	td->tinfos = (trainer_job_info *) xmalloc(size);
	if (!td->tinfos) goto error_init;

	for (i = 0; i < num_threads; i++) {
		td->tinfos[i].id = i;
		td->tinfos[i].td = td;
		boosting_reset(&td->tinfos[i].bs);
		cpa_reset(&td->tinfos[i].c);
		td->tinfos[i].feat_vals = NULL;
	}

	for (i = 0; i < num_threads; i++) {
		if (!boosting_init(&td->tinfos[i].bs, td->max_n,
		                   num_buckets, num_parallels))
			goto error_init;

		if (nbins > 0) {
			if (!cpa_init(&td->tinfos[i].c, 4 * nbins,
			              td->max_n, max_planes))
				goto error_init;
		}

		size = td->max_n * sizeof(double);
		td->tinfos[i].feat_vals = (double *) xmalloc(size);
		if (!td->tinfos[i].feat_vals)
			goto error_init;
	}

	image_init(&td->img);

	if (!thread_pool_init(&td->tp, num_threads))
		goto error_init;

	if (!detector_init(&td->dt, width, height, num_parallels,
	                   num_threads, num_threads, &td->tp))
		goto error_init;

	feature_enumerator_start(&fe, width, height, FALSE);
	td->total_num_features = feature_enumerator_count(&fe);

	return TRUE;

error_init:
	trainer_cleanup(td);
	return FALSE;
}

void trainer_cleanup(trainer_data *td)
{
	unsigned int i;

	thread_pool_cleanup(&td->tp);
	detector_cleanup(&td->dt);
	samples_cleanup(&td->smp);
	image_cleanup(&td->img);

	if (td->tinfos) {
		for (i = 0; i < td->num_threads; i++) {
			boosting_cleanup(&td->tinfos[i].bs);
			cpa_cleanup(&td->tinfos[i].c);
			if (td->tinfos[i].feat_vals) {
				free(td->tinfos[i].feat_vals);
				td->tinfos[i].feat_vals = NULL;
			}
		}
		free(td->tinfos);
		td->tinfos = NULL;
	}

	if (td->sat_buffer) {
		free(td->sat_buffer);
		td->sat_buffer = NULL;
	}

	if (td->sat) {
		free(td->sat);
		td->sat = NULL;
	}

	if (td->y) {
		free(td->y);
		td->y = NULL;
	}
}

void trainer_boost_params(trainer_data *td, double Cp, double Cn,
                          double bkt_min, double bkt_max)
{
	boosting_set_params(&td->tinfos[0].bs, Cp, Cn,
	                    bkt_min, bkt_max);
}

static
void trainer_learn_overlap(const trainer_data *td, double *match_thresh,
                           double *overlap_thresh)
{
	unsigned int i, j;

	*match_thresh = 0;
	*overlap_thresh = 0;
	for (i = 0; i < td->smp.num_items; i++) {
		sample_item *item;

		item = &td->smp.items[i];
		if (!item->positive)
			continue;

		for (; i <= item->same_last; i++) {
			for (j = i + 1; j <= item->same_last; j++) {
				window_compute_overlap(&td->smp.items[i].w,
				                       &td->smp.items[j].w,
				                       match_thresh,
				                       overlap_thresh);
			}
		}
		i = item->same_last;
	}
	*match_thresh *= 1 + EPS;
	*overlap_thresh *= 1 + EPS;
	printf("match_thresh = %g, overlap_thresh = %g\n",
	       *match_thresh, *overlap_thresh);
}

void trainer_cascade_params(trainer_data *td, int multi_exit,
                            double scale, double min_stddev, unsigned int step,
                            double match_thresh, double overlap_thresh,
                            int learn_overlap)
{
	if (learn_overlap)
		trainer_learn_overlap(td, &match_thresh, &overlap_thresh);

	detector_set_params(&td->dt, scale, min_stddev, step,
	                    match_thresh, overlap_thresh, multi_exit);
}

void trainer_cpa_params(trainer_data *td, double Cp, double Cn,
                        double eps, double eps_qp, double mu,
                        unsigned int max_unused)
{
	unsigned int i;
	for (i = 0; i < td->num_threads; i++) {
		cpa_set_params(&td->tinfos[i].c, Cp, Cn, eps, eps_qp,
		               mu, max_unused);
	}
}

void trainer_params(trainer_data *td, unsigned int max_stages,
                    unsigned int max_classifiers, unsigned int min_jumbled,
                    unsigned int min_negative, double max_false_positive,
                    double max_false_negative, double feature_prob,
                    double min_similarity, int cycle_parallels)
{
	td->max_stages = max_stages;
	td->max_classifiers = max_classifiers;
	td->min_jumbled = min_jumbled;
	td->min_negative = min_negative;
	td->max_false_positive = max_false_positive;
	td->max_false_negative = max_false_negative;
	td->feature_prob = feature_prob;
	td->min_similarity = min_similarity;
	td->cycle_parallels = cycle_parallels;
}

int trainer_load(trainer_data *td, const char *cascade_filename)
{
	return detector_load(&td->dt, cascade_filename, FALSE,
	                     td->num_threads, td->num_threads);
}

int trainer_save(const trainer_data *td, const char *cascade_filename)
{
	return detector_save(&td->dt, cascade_filename);
}

static
int process_sample_item(detector_job_info *info)
{
	unsigned int i, j, l, offset, count;
	detected_object *objs;
	sample_item *item;
	cascade *c;

	item = (sample_item *) info->extra;
	c = &info->c;

	if (!item->positive)
		return TRUE;


	count = item->same_last - item->same_first + 1;
	for (i = 0; i < count; i++) {
		item[i].mark2 = 0;
		item[i].similarity = 0;
	}

	objs = c->detected_objects;
	for (j = 0; j < c->num_jumbled_objects; j++) {
		for (i = 0; i < count; i++) {
			double sim;
			sim = window_similarity(&item[i].w, &objs[j].w);
			if (item[i].similarity < sim) {
				item[i].similarity = sim;
				item[i].mark2 = j + 1;
				break;
			}
		}
	}

	offset = 0;
	for (i = 0; i < count; i++) {
		detected_object temp;
		if (item[i].mark2 == 0)
			continue;

		j = item[i].mark2 - 1;
		temp = objs[offset];
		objs[offset] = objs[j];
		objs[j] = temp;

		for (l = i + 1; l < count; l++) {
			if (item[l].mark2 == offset + 1)
				item[l].mark2 = j + 1;
		}
		offset++;
	}
	cascade_separate(c, offset);

	return TRUE;
}

static
int consume_sample_item(trainer_data *td)
{
	unsigned int i, j, id, count;
	unsigned int num_objects, pos_objects, neg_objects;
	unsigned int k;
	detector_job_info *info;
	sample_item *item;
	boosting *bs;
	int status;

	id = detector_dequeue(&td->dt);
	if (id == 0)
		return FALSE;

	info = &td->dt.infos[id - 1];
	if (!info->success) {
		detector_release(&td->dt, id);
		return FALSE;
	}

	bs = &td->tinfos[0].bs;
	item = (sample_item *) info->extra;
	printf("`%s' has %u detected objects (%u jumbled)... ",
	       item->filename, info->c.num_detected_objects,
	       info->c.num_jumbled_objects);

	item->mark1 = info->c.num_jumbled_objects;

	if (item->mark1 <= td->min_jumbled && !item->positive)
		num_objects = item->mark1;
	else
		num_objects = info->c.num_detected_objects;

	pos_objects = 0;
	neg_objects = 0;
	count = item->same_last - item->same_first + 1;
	for (j = 0; j < num_objects; j++) {
		detected_object *obj;

		obj = &info->c.detected_objects[j];
		status = 0;
		if (item->positive) {
			for (i = 0; i < count; i++) {
				double sim;

				sim = window_similarity(&obj->w, &item[i].w);
				if (sim >= td->min_similarity) status |= 1;

				if (cascade_overlap(&info->c, &obj->w,
				                    &item[i].w))
					status |= 2;
			}
		}

		if (status & 1) {
			if (td->num_pos >= td->inum_pos)
				continue;
			i = td->num_pos++;
			pos_objects++;
		} else if (status == 0) {
			if (td->num_neg >= td->inum_neg)
				continue;
			i = td->inum_pos + td->num_neg++;
			neg_objects++;
		} else {
			continue;
		}

		if (!cascade_extract(&info->c, &obj->comp, td->sat[i]))
			return FALSE;

		td->y[i] = (status & 1) ? 1 : -1;
		if (info->c.multi_exit) {
			for (k = 0; k < bs->num_parallels; k++) {
				bs->vals[k][i] = obj->score[k];
			}
		} else {
			for (k = 0; k < bs->num_parallels; k++)
				bs->vals[k][i] = 0;
		}
	}

	printf("(pos = %u, neg = %u)\n", pos_objects, neg_objects);
	detector_release(&td->dt, id);
	return TRUE;
}

static
int update_samples(trainer_data *td, int reset_markers)
{
	unsigned int i, j;
	unsigned int k;
	sample_item *item;
	boosting *bs;
	int ret;

	bs = &td->tinfos[0].bs;
	if (!detector_prepare(&td->dt, &detector_load_sample_item,
	                      &process_sample_item, TRUE))
		return FALSE;

	td->num_pos = 0;
	td->num_neg = 0;

	if (reset_markers) {
		for (i = 0; i < td->smp.num_items; i++) {
			td->smp.items[i].mark1 = 1;
		}
	}

	for (i = 0; i < td->smp.num_items; i++) {
		while (detector_peek(&td->dt)) {
			if (!consume_sample_item(td))
				return FALSE;
		}

		item = &td->smp.items[i];
		if (item->mark1 == 0) continue;

		if (item->positive) {
			if (td->num_pos >= td->inum_pos
			    || item->same_first != i)
				continue;
		} else {
			if (td->num_neg >= td->inum_neg)
				break;
		}

		while (TRUE) {
			ret = detector_enqueue(&td->dt, NULL, item,
			                       !item->positive);
			if (ret < 0) return FALSE;
			if (ret > 0) break;

			if (!consume_sample_item(td))
				return FALSE;
		}
	}

	while (detector_pending(&td->dt, TRUE, TRUE)) {
		if (!consume_sample_item(td))
			return FALSE;
	}

	td->n = td->num_pos + td->num_neg;
	j = td->inum_pos + td->num_neg;
	for (i = td->num_pos; i < td->inum_pos; i++) {
		sval *temp;
		if (j-- == td->inum_pos) break;
		td->y[i] = td->y[j];
		for (k = 0; k < bs->num_parallels; k++)
			bs->vals[k][i] = bs->vals[k][j];

		temp = td->sat[i];
		td->sat[i] = td->sat[j];
		td->sat[j] = temp;
	}
	return TRUE;
}

static
void trainer_job(void *arg)
{
	trainer_job_info *info;
	trainer_data *td;
	boosting *bs;
	feature_enumerator fe;
	feature_index_opt fo;
	unsigned int i, fstart, fend, id;
	double *feat_vals;

	info = (trainer_job_info *) arg;
	td = info->td;
	id = info->id;
	bs = &info->bs;
	boosting_prepare(bs);

	if (id == 0) printf("Init: val = %g\n", bs->best_val);

	feat_vals = info->feat_vals;

	fstart = id * td->total_num_features / td->num_threads;
	fend = (id + 1) * td->total_num_features / td->num_threads;

	feature_enumerator_start(&fe, td->width, td->height, FALSE);
	feature_enumerator_advance(&fe, fstart);
	do {
		if (fe.count >= fend) break;
		if (td->feature_prob < 1) {
			if (genrand_real1() >= td->feature_prob)
				continue;
		}
		features_optimize(&fe.fi, &fo, td->width + 1);
		for (i = 0; i < td->n; i++) {
			feat_vals[i] = features_evaluate_fast(td->sat[i], &fo);
		}

		boosting_train(&info->bs, info->feat_vals, fe.count + 1,
		               info->parallel);
	} while (feature_enumerator_next(&fe));
}

static
int train_classifier(trainer_data *td, cascade *c, cascade_stage *st,
                     unsigned int parallel, unsigned int *pfp,
                     unsigned int *pfn)
{
	boosting *bs;
	classifier *cl;
	feature_enumerator fe;
	feature_index fi;
	feature_index_opt fo;
	unsigned int i;
	double *feat_vals;

	bs = &td->tinfos[0].bs;
	feat_vals = td->tinfos[0].feat_vals;
	for (i = 1; i < td->num_threads; i++)
		boosting_copy(bs, &td->tinfos[i].bs);

	for (i = 0; i < td->num_threads; i++) {
		td->tinfos[i].parallel = parallel;
		if (!thread_pool_enqueue(&td->tp, &trainer_job,
		                         &td->tinfos[i])) {
			error("could not start training job");
			return -1;
		}
	}
	thread_pool_wait(&td->tp);
	thread_pool_flush_done(&td->tp);

	for (i = 1; i < td->num_threads; i++)
		boosting_merge_best(bs, &td->tinfos[i].bs);

	if (bs->best_index == 0) {
		printf("Did not improve!\n");
		return 0;
	}

	cl = cascade_new_classifier(c, st, bs->best_parallel);
	if (!cl) return -1;

	feature_enumerator_start(&fe, td->width, td->height, FALSE);
	feature_enumerator_advance(&fe, bs->best_index - 1);
	fi = fe.fi;

	features_optimize(&fe.fi, &fo, td->width + 1);
	for (i = 0; i < td->n; i++) {
		feat_vals[i] = features_evaluate_fast(td->sat[i], &fo);
	}
	boosting_compute_best(bs, feat_vals);
	boosting_update(bs, feat_vals, bs->best_parallel, bs->best_coef,
	                bs->best_intercept, bs->best_threshold);
	bs->best_val = boosting_set_weights(bs, pfp, pfn);

	printf("Best: index = %d (%u, %u, %u, %u), thresh = %g\n", fi.idx,
	       fi.w.left, fi.w.top, fi.w.width, fi.w.height,
	       bs->best_threshold);
	printf("parallel = %u, coef = %g, intercept = %g\n",
	       bs->best_parallel, bs->best_coef, bs->best_intercept);
	printf("val = %g, fp = %u, fn = %u\n", bs->best_val, *pfp, *pfn);
	printf("Sum_pf = %g, Sum_pn = %g, Sum_nf = %g, Sum_nn = %g\n",
	       bs->best_sum_pf, bs->best_sum_pn,
	       bs->best_sum_nf, bs->best_sum_nn);

	cl->fi = fi;
	cl->thresh = bs->best_threshold;
	cl->coef = bs->best_coef;
	cl->intercept = bs->best_intercept;

	td->best_val = bs->best_val;
	return 1;
}

static
int optimize_classifier(trainer_data *td, cascade_stage *st, classifier *cl,
                        unsigned int parallel)
{
	boosting *bs;
	double *feat_vals;
	unsigned int i, fp, fn;
	feature_index_opt fo;
	int changed;

	bs = &td->tinfos[0].bs;
	feat_vals = td->tinfos[0].feat_vals;

	features_optimize(&cl->fi, &fo, td->width + 1);
	for (i = 0; i < td->n; i++) {
		feat_vals[i] = features_evaluate_fast(td->sat[i], &fo);
	}
	td->best_val = bs->best_val;
	changed = boosting_refine(bs, feat_vals, parallel, cl->coef,
	                          cl->intercept, cl->thresh, &fp, &fn);

	if (changed > 0) {
		printf("Changed to: val = %g (other = %g, diff = %g), "
		       "fp = %u, fn = %u\n", bs->best_val, td->best_val,
		       td->best_val - bs->best_val, fp, fn);
		printf("Sum_pf = %g, Sum_pn = %g, Sum_nf = %g, Sum_nn = %g\n",
		       bs->best_sum_pf, bs->best_sum_pn,
		       bs->best_sum_nf, bs->best_sum_nn);
		printf("(index = %d (%u, %u, %u, %u), thresh = %g)",
		       cl->fi.idx, cl->fi.w.left, cl->fi.w.top,
		       cl->fi.w.width, cl->fi.w.height,
		       bs->best_threshold);
		printf(" coef = %g, intercept = %g, parallel = %u\n",
		       bs->best_coef, bs->best_intercept, parallel);

		cl->coef = bs->best_coef;
		cl->intercept = bs->best_intercept;
		cl->thresh = bs->best_threshold;

		changed = 0;
		if ((td->best_val - bs->best_val) > EPS * bs->best_val)
			changed = 1;
		else if ((bs->best_val - td->best_val) > EPS * td->best_val)
			changed = -1;
	}
	td->best_val = bs->best_val;
	return changed;
}

static
int optimize_stage(trainer_data *td, cascade *c, cascade_stage *st)
{
	classifier *cl;
	int ret, changed;
	unsigned int count = 0;
	unsigned int k;

	do {
		changed = FALSE;
		for (k = 0; k < c->num_parallels; k++) {
			for (cl = st->cl[k]; cl; cl = cl->next) {
				ret = optimize_classifier(td, st, cl, k);
				if (ret < 0) return TRUE;
				if (ret > 0) changed = TRUE;
			}
		}
		count++;
	} while (changed && count < MAX_ITERATIONS);
	return TRUE;
}

static
int train_stage(trainer_data *td, int *done)
{
	unsigned int i, num_weak, fp, fn;
	unsigned int parallel, done_count = 0;
	boosting *bs;
	cascade *c;
	cascade_stage *st;
	int ret;

	c = &td->dt.infos[0].c;
	st = cascade_new_stage(c);
	if (!st) return FALSE;

	bs = &td->tinfos[0].bs;
	boosting_set_samples(bs, td->y, td->n);

	td->best_val = -1;
	for (num_weak = i = 0; num_weak < td->max_classifiers; i++) {
		double elapsed, cpu_time;
		stopwatch sw;

		printf("\nTraining weak classifier %u...\n", num_weak);
		printf("Num positive = %u, Num negative = %u, "
		       "Target fn = %g, target fp = %g\n",
		       td->num_pos, td->num_neg,
		       td->num_pos * td->max_false_negative,
		       td->num_neg * td->max_false_positive);

		stopwatch_start(&sw);

		if (td->cycle_parallels)
			parallel = i % bs->num_parallels;
		else
			parallel = bs->num_parallels;
		ret = train_classifier(td, c, st, parallel, &fp, &fn);

		if (ret < 0) return FALSE;

		stopwatch_stop(&sw, &elapsed, &cpu_time);
		printf("Time = %f, Cpu time = %f\n", elapsed, cpu_time);

		if (ret == 0) {
			done_count++;
			if (td->cycle_parallels) {
				if (done_count >= bs->num_parallels) {
					*done = TRUE;
				}
			} else {
				*done = TRUE;
			}
			if (*done) break;
		} else {
			done_count = 0;
			num_weak++;
			if (!optimize_stage(td, c, st))
				return FALSE;

			if ((fp <= td->num_neg * td->max_false_positive)
			    && (fn <= td->num_pos * td->max_false_negative))
				break;
		}
	}

	cascade_consolidate_stage(c, st);
	if (i == td->max_classifiers)
		*done = TRUE;

	return TRUE;
}

int trainer_train(trainer_data *td, const char *cascade_filename,
                  const char *data_directory)
{
	unsigned int stage;
	char current_directory[MAX_DIRECTORY_SIZE];
	int done;

	if (access(cascade_filename, F_OK) != -1) {
		if (!trainer_load(td, cascade_filename))
			return FALSE;
	}

	if (!getcwd(current_directory, sizeof(current_directory))) {
		error("can not obtain current directory");
		return FALSE;
	}

	if (chdir(data_directory)) {
		error("missing directory `%s'", data_directory);
		return FALSE;
	}

	done = FALSE;
	stage = td->dt.infos[0].c.num_stages;
	if (!update_samples(td, TRUE))
		return FALSE;

	for (; stage < td->max_stages; stage++) {
		if (td->num_neg < td->min_negative) {
			printf("Stopping before stage %u - "
			       "not enough negative samples - "
			       "have: %u, required: %u\n", stage,
			       td->num_neg, td->min_negative);
			break;
		}
		printf("\n\n\nTraining stage %u:\n", stage);

		if (!train_stage(td, &done))
			return FALSE;

		if (chdir(current_directory)) {
			error("can not change directory back to `%s'",
			      current_directory);
			return FALSE;
		}

		if (!done) {
			if (!trainer_save(td, cascade_filename))
				return FALSE;
		}

		if (chdir(data_directory)) {
			error("missing directory `%s'", data_directory);
			return FALSE;
		}

		if (done) {
			printf("Removing last stage\n");
			cascade_remove_last_stage(&td->dt.infos[0].c);
			break;
		}

		if (!update_samples(td, FALSE))
			return FALSE;
	}

	if (chdir(current_directory)) {
		error("can not change directory back to `%s'",
		      current_directory);
		return FALSE;
	}

	if (!trainer_save(td, cascade_filename))
		return FALSE;

	return TRUE;
}
