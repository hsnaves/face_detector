#ifndef __TRAINER_H
#define __TRAINER_H

#include "boosting.h"
#include "cpa.h"
#include "detector.h"
#include "cascade.h"
#include "samples.h"
#include "image.h"
#include "features.h"
#include "thread_pool.h"

/* Dara structures */
struct trainer_data_st;

typedef
struct trainer_job_info_st {
	unsigned int id;
	struct trainer_data_st *td;
	boosting bs;
	cpa c;
	unsigned int parallel;
	double *feat_vals;
} trainer_job_info;

typedef
struct trainer_data_st {
	unsigned int width, height;
	unsigned int n, max_n;
	unsigned int nbins;
	unsigned int num_pos, num_neg, inum_pos, inum_neg;
	unsigned int num_threads;
	unsigned int total_num_features;
	thread_pool tp;
	trainer_job_info *tinfos;
	detector dt;
	samples smp;
	image img;

	unsigned int max_stages, max_classifiers;
	unsigned int min_jumbled, min_negative;
	double max_false_positive, max_false_negative;
	double feature_prob, min_similarity;
	int cycle_parallels;

	double best_val;
	sval *sat_buffer;
	sval **sat;
	int *y;
} trainer_data;

/* Functions */
void trainer_reset(trainer_data *td);
int trainer_init(trainer_data *td, const char *samples_filename,
                 unsigned int width, unsigned int height,
                 unsigned int pos_samples, unsigned int neg_samples,
                 unsigned int num_buckets, unsigned int num_parallels,
                 unsigned int num_threads, unsigned int nbins,
                 unsigned int max_planes);
void trainer_cleanup(trainer_data *td);

void trainer_boost_params(trainer_data *td, double Cp, double Cn,
                          double bkt_min, double bkt_max);
void trainer_cascade_params(trainer_data *td, int multi_exit,
                            double scale, double min_stddev, unsigned int step,
                            double match_thresh, double overlap_thresh,
                            int learn_overlap);
void trainer_cpa_params(trainer_data *td, double Cp, double Cn,
                        double eps, double eps_qp,  double mu,
                        unsigned int max_unused);
void trainer_params(trainer_data *td, unsigned int max_stages,
                    unsigned int max_classifiers, unsigned int min_jumbled,
                    unsigned int min_negative, double max_false_positive,
                    double max_false_negative, double feature_prob,
                    double min_similarity, int cycle_parallels);
int trainer_load(trainer_data *td, const char *cascade_filename);
int trainer_save(const trainer_data *td, const char *cascade_filename);

int trainer_train(trainer_data *td, const char *cascade_filename,
                  const char *data_directory);

#endif /* __TRAINER_H */
