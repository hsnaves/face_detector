#ifndef __BOOSTING_H
#define __BOOSTING_H

/* Data structures */
typedef
struct boosting_st {
	unsigned int n, max_n, num_buckets;
	unsigned int num_parallels;
	double Cp, Cn;
	double bkt_min, bkt_max;

	int *y;
	double **vals, **weights;
	unsigned int *sel_parallel;
	unsigned int *buckets;
	unsigned int *next;

	double init_val;
	double *init_sum_p, *init_sum_n;
	double best_val, best_threshold;
	double best_coef, best_intercept;
	double best_sum_pf, best_sum_pn;
	double best_sum_nf, best_sum_nn;
	unsigned int best_index, best_thresh_index;
	unsigned int best_parallel;
} boosting;

/* Functions */
void boosting_reset(boosting *bs);
int boosting_init(boosting *bs, unsigned int max_n,
                  unsigned int num_buckets, unsigned int num_parallels);
void boosting_cleanup(boosting *bs);

int boosting_set_params(boosting *bs, double Cp, double Cn,
                        double bkt_min, double bkt_max);
void boosting_set_samples(boosting *bs, int *y, unsigned int n);
void boosting_update(boosting *bs, const double *feat_vals,
                     unsigned int parallel, double coef, double intercept,
                     double thresh);
double boosting_set_weights(boosting *bs, unsigned int *pfp,
                            unsigned int *pfn);
void boosting_prepare(boosting *bs);
void boosting_train(boosting *bs, const double *feat_vals, unsigned int index,
                    unsigned int parallel);
void boosting_compute_best(boosting *bs, const double *feat_vals);
int boosting_refine(boosting *bs, const double *feat_vals,
                    unsigned int parallel, double coef, double intercept,
                    double thresh, unsigned int *pfp, unsigned int *pfn);

void boosting_copy(const boosting *from, boosting *to);
void boosting_merge_best(boosting *bs, const boosting *other);

#endif /* __BOOSTING_H */

