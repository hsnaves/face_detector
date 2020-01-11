
#ifndef __DETECTOR_H
#define __DETECTOR_H

#include "image.h"
#include "cascade.h"
#include "samples.h"
#include "thread_pool.h"

/* Data structures and types */
struct detector_st;
struct detector_job_info_st;

typedef int (*detector_callback)(struct detector_job_info_st *info);

typedef
struct detector_job_info_st {
	int success;
	int separate_detected;
	unsigned int id, idx, next;
	cascade c;
	image img;
	void *extra;
	struct detector_st *dt;
} detector_job_info;

typedef
struct detector_st {
	unsigned int num_cascades, num_threads;
	thread_pool *tp, mtp;
	detector_job_info *infos;

	int enforce_order;
	detector_callback pre_fn, post_fn;

	unsigned int done_idx, curr_idx;
	unsigned int free, done;
} detector;

/* Functions */
void detector_reset(detector *dt);
int detector_init(detector *dt, unsigned int width, unsigned int height,
                  unsigned int num_parallels, unsigned int num_cascades,
                  unsigned int num_threads, thread_pool *tp);

void detector_cleanup(detector *dt);
void detector_get_params(const detector *dt, double *scale, double *min_stddev,
                         unsigned int *step, double *match_thresh,
                         double *overlap_thresh, int *multi_exit);
void detector_set_params(detector *dt, double scale, double min_stddev,
                         unsigned int step, double match_thresh,
                         double overlap_thresh, int multi_exit);
void detector_set_scan(detector *dt,
                       unsigned int min_width, unsigned int min_height,
                       unsigned int max_width, unsigned int max_height);

int detector_prepare(detector *dt, detector_callback pre_fn,
                     detector_callback post_fn, int enforce_order);
int detector_enqueue(detector *dt, const image *img, void *extra,
                     int separate_detected);
int detector_peek(const detector *dt);
unsigned int detector_dequeue(detector *dt);
void detector_release(detector *dt, unsigned int id);
int detector_pending(detector *dt, int remaining, int done);

int detector_load(detector *dt, const char *filename, int reset,
                  unsigned int num_cascades, unsigned int num_threads);
int detector_save(const detector *dt, const char *filename);

int detector_load_sample_item(detector_job_info *info);
int detector_evaluate(detector *dt, const samples *smp,
                      const char *data_directory);

#endif /* __DETECTOR_H */
