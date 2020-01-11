#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "boosting.h"
#include "utils.h"

#define TAU (EPS * EPS)

void boosting_reset(boosting *bs)
{
	bs->vals = NULL;
	bs->weights = NULL;
	bs->sel_parallel = NULL;
	bs->init_sum_p = NULL;
	bs->init_sum_n = NULL;
	bs->buckets = NULL;
	bs->next = NULL;
}

int boosting_init(boosting *bs, unsigned int max_n,
                  unsigned int num_buckets, unsigned int num_parallels)
{
	unsigned int k;
	size_t size;
	boosting_reset(bs);

	bs->y = NULL;
	bs->max_n = max_n;
	bs->num_buckets = num_buckets;
	bs->Cn = bs->Cp = 1;
	bs->n = 0;

	bs->num_parallels = num_parallels;
	size = num_parallels * sizeof(double *);
	bs->vals = (double **) xmalloc(size);
	if (!bs->vals) goto error_init;

	size = num_parallels * max_n * sizeof(double);
	bs->vals[0] = (double *) xmalloc(size);
	if (!bs->vals[0]) goto error_init;

	for (k = 1; k < num_parallels; k++) {
		bs->vals[k] = (bs->vals[k - 1]) + max_n;
	}

	size = num_parallels * sizeof(double *);
	bs->weights = (double **) xmalloc(size);
	if (!bs->weights) goto error_init;

	size = num_parallels * max_n * sizeof(double);
	bs->weights[0] = (double *) xmalloc(size);
	if (!bs->weights[0]) goto error_init;

	for (k = 1; k < num_parallels; k++) {
		bs->weights[k] = (bs->weights[k - 1]) + max_n;
	}

	size = max_n * sizeof(unsigned int);
	bs->sel_parallel = (unsigned int *) xmalloc(size);
	if (!bs->sel_parallel) goto error_init;

	size = max_n * sizeof(unsigned int);
	bs->next = (unsigned int *) xmalloc(size);
	if (!bs->next) goto error_init;

	size = (num_buckets + 1) * sizeof(unsigned int);
	bs->buckets = (unsigned int *) xmalloc(size);
	if (!bs->buckets) goto error_init;

	size = num_parallels * sizeof(double);
	bs->init_sum_p = (double *) xmalloc(size);
	if (!bs->init_sum_p) goto error_init;

	size = num_parallels * sizeof(double);
	bs->init_sum_n = (double *) xmalloc(size);
	if (!bs->init_sum_n) goto error_init;

	return TRUE;

error_init:
	boosting_cleanup(bs);
	return FALSE;
}

void boosting_cleanup(boosting *bs)
{
	if (bs->vals) {
		if (bs->vals[0])
			free(bs->vals[0]);

		free(bs->vals);
		bs->vals = NULL;
	}

	if (bs->weights) {
		if (bs->weights[0])
			free(bs->weights[0]);

		free(bs->weights);
		bs->weights = NULL;
	}

	if (bs->sel_parallel) {
		free(bs->sel_parallel);
		bs->sel_parallel = NULL;
	}

	if (bs->init_sum_p) {
		free(bs->init_sum_p);
		bs->init_sum_p = NULL;
	}

	if (bs->init_sum_n) {
		free(bs->init_sum_n);
		bs->init_sum_n = NULL;
	}

	if (bs->buckets) {
		free(bs->buckets);
		bs->buckets = NULL;
	}

	if (bs->next) {
		free(bs->next);
		bs->next = NULL;
	}
}

int boosting_set_params(boosting *bs, double Cp, double Cn,
                        double bkt_min, double bkt_max)
{
	bs->Cp = Cp;
	bs->Cn = Cn;

	bs->bkt_min = bkt_min;
	bs->bkt_max = bkt_max;

	return TRUE;
}

void boosting_set_samples(boosting *bs, int *y, unsigned int n)
{
	bs->n = n;
	bs->y = y;
}

void boosting_update(boosting *bs, const double *feat_vals,
                     unsigned int parallel, double coef, double intercept,
                     double thresh)
{
	unsigned int i;

	for (i = 0; i < bs->n; i++) {
		bs->vals[parallel][i] += intercept;
		if (feat_vals[i] >= thresh) {
			bs->vals[parallel][i] += coef;
		}
	}
}

double boosting_set_weights(boosting *bs, unsigned int *pfp,
                            unsigned int *pfn)
{
	unsigned int i, fp, fn;
	unsigned int k, sel;
	double total, mult;
	int y;

	for (k = 0; k < bs->num_parallels; k++) {
		bs->init_sum_p[k] = 0;
		bs->init_sum_n[k] = 0;
	}

	fp = fn = 0;
	for (i = 0; i < bs->n; i++) {
		y = bs->y[i];
		mult = (y > 0) ? bs->Cn : bs->Cp;
		for (k = 0; k < bs->num_parallels; k++) {
			bs->weights[k][i] = mult * exp(-y * bs->vals[k][i]);
		}

		sel = 0;
		for (k = 1; k < bs->num_parallels; k++) {
			if (bs->vals[k][i] > bs->vals[sel][i])
				sel = k;
		}
		bs->sel_parallel[i] = sel;
		if (y > 0) {
			bs->init_sum_p[sel] += bs->weights[sel][i];

			if (bs->vals[sel][i] <= 0)
				fn++;
		} else {
			for (k = 0; k < bs->num_parallels; k++)
				bs->init_sum_n[k] += bs->weights[k][i];

			if (bs->vals[sel][i] >= 0)
				fp++;
		}
	}

	total = 0;
	for (k = 0; k < bs->num_parallels; k++)
		total += bs->init_sum_p[k] + bs->init_sum_n[k];

	if (pfp) *pfp = fp;
	if (pfn) *pfn = fn;
	return total;
}

void boosting_prepare(boosting *bs)
{
	bs->init_val = boosting_set_weights(bs, NULL, NULL);
	bs->best_val = bs->init_val;
	bs->best_threshold = -INFINITY;
	bs->best_thresh_index = 0;
	bs->best_parallel = 0;
	bs->best_coef = 0;
	bs->best_intercept = 0;
	bs->best_index = 0;
}

static
void make_buckets(boosting *bs, const double *feat_vals)
{
	unsigned int i, j;
	double bkt_range;

	memset(bs->buckets, 0, (bs->num_buckets + 1) * sizeof(unsigned int));
	bkt_range = bs->bkt_max - bs->bkt_min;
	for (i = 0; i < bs->n; i++) {
		double fval = feat_vals[i];

		if (fval < bs->bkt_min) {
			j = 0;
		} else if (fval >= bs->bkt_max)  {
			j = bs->num_buckets;
		} else {
			fval = (fval - bs->bkt_min) / bkt_range;
			j = (unsigned int) (fval * bs->num_buckets);
		}

		bs->next[i] = bs->buckets[j];
		bs->buckets[j] = i + 1;
	}
}

static
void train_aux(boosting *bs, unsigned int index, unsigned int k)
{
	unsigned int i, j;
	double val, init_val, sum_pf, sum_pn, sum_nf, sum_nn;
	double sum2_pf, sum2_pn, sum2_nf, sum2_nn;

	sum_pn = 0;
	sum_nn = 0;
	sum_pf = bs->init_sum_p[k];
	sum_nf = bs->init_sum_n[k];

	init_val = bs->init_val - sum_pf - sum_nf;

	for (i = 0; i < bs->num_buckets; i++) {
		j = bs->buckets[i];
		if (j == 0) continue;
		do {
			j--;
			if (bs->y[j] > 0 && bs->sel_parallel[j] == k) {
				sum_pf -= bs->weights[k][j];
				sum_pn += bs->weights[k][j];
			} else if (bs->y[j] < 0) {
				sum_nf -= bs->weights[k][j];
				sum_nn += bs->weights[k][j];
			}
			j = bs->next[j];
		} while (j > 0);

		sum2_pf = MAX(TAU, sum_pf);
		sum2_nf = MAX(TAU, sum_nf);
		sum2_pn = MAX(TAU, sum_pn);
		sum2_nn = MAX(TAU, sum_nn);

		val = init_val;
		val += 2 * sqrt(sum2_pf * sum2_nf);
		val += 2 * sqrt(sum2_pn * sum2_nn);
		if (val < bs->best_val) {
			bs->best_val = val;
			bs->best_index = index;
			bs->best_parallel = k;
			bs->best_thresh_index = i;
			bs->best_sum_pn = sum2_pn;
			bs->best_sum_nn = sum2_nn;
			bs->best_sum_pf = sum2_pf;
			bs->best_sum_nf = sum2_nf;
		}
	}
}

void boosting_train(boosting *bs, const double *feat_vals, unsigned int index,
                    unsigned int parallel)
{
	unsigned int k;
	make_buckets(bs, feat_vals);

	if (parallel == bs->num_parallels) {
		for (k = 0; k < bs->num_parallels; k++) {
			train_aux(bs, index, k);
		}
	} else {
		train_aux(bs, index, parallel);
	}
}

void boosting_compute_best(boosting *bs, const double *feat_vals)
{
	double bkt_range;

	if (bs->best_index == 0) return;

	bs->best_intercept = .5 * log(bs->best_sum_pn / bs->best_sum_nn);
	bs->best_coef = .5 * log(bs->best_sum_pf / bs->best_sum_nf);
	bs->best_coef -= bs->best_intercept;

	bkt_range = bs->bkt_max - bs->bkt_min;
	if (bs->best_thresh_index == bs->num_buckets) {
		bs->best_threshold = INFINITY;
	} else {
		bs->best_threshold = (bs->best_thresh_index + 1) * bkt_range
		                     / bs->num_buckets;
		bs->best_threshold += bs->bkt_min;
	}
}

int boosting_refine(boosting *bs, const double *feat_vals,
                    unsigned int parallel, double coef, double intercept,
                    double thresh, unsigned int *pfp, unsigned int *pfn)
{
	double old_best_val;

	old_best_val = bs->best_val;
	boosting_update(bs, feat_vals, parallel, -coef, -intercept, thresh);
	boosting_prepare(bs);
	boosting_train(bs, feat_vals, 1, parallel);
	boosting_compute_best(bs, feat_vals);

	if (old_best_val - bs->best_val > EPS * bs->best_val) {
		boosting_update(bs, feat_vals, bs->best_parallel,
		                bs->best_coef, bs->best_intercept,
		                bs->best_threshold);
		bs->best_val = boosting_set_weights(bs, pfp, pfn);
		return 1;
	} else {
		boosting_update(bs, feat_vals, parallel,
		                coef, intercept, thresh);
		bs->best_val = boosting_set_weights(bs, pfp, pfn);
		if (bs->best_val - old_best_val > EPS * old_best_val) {
			error("best_val increased: %g < %g, diff = %g",
			      old_best_val, bs->best_val,
			      bs->best_val - old_best_val);
			return -1;
		}
		return 0;
	}

}

void boosting_copy(const boosting *from, boosting *to)
{
	unsigned int k;
	size_t size;

	to->n = from->n;
	to->y = from->y;
	to->bkt_min = from->bkt_min;
	to->bkt_max = from->bkt_max;

	to->Cp = from->Cp;
	to->Cn = from->Cn;

	size = from->n * sizeof(double);
	for (k = 0; k < from->num_parallels; k++)
		memcpy(to->vals[k], from->vals[k], size);
}

void boosting_merge_best(boosting *bs, const boosting *other)
{
	if (bs->best_val > other->best_val) {
		bs->best_val = other->best_val;
		bs->best_index = other->best_index;
		bs->best_parallel = other->best_parallel;
		bs->best_thresh_index = other->best_thresh_index;
		bs->best_sum_pn = other->best_sum_pn;
		bs->best_sum_pf = other->best_sum_pf;
		bs->best_sum_nn = other->best_sum_nn;
		bs->best_sum_nf = other->best_sum_nf;
	}
}
