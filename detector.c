
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "detector.h"
#include "image.h"
#include "cascade.h"
#include "thread_pool.h"
#include "samples.h"
#include "utils.h"


void detector_reset(detector *dt)
{
	thread_pool_reset(&dt->mtp);
	dt->tp = NULL;
	dt->infos = NULL;
}

static
int detector_preinit(detector *dt, unsigned int num_cascades,
                     unsigned int num_threads, thread_pool *tp)
{
	unsigned int i;
	size_t size;
	detector_reset(dt);

	dt->num_cascades = num_cascades;
	dt->num_threads = num_threads;
	size = num_cascades * sizeof(detector_job_info);
	dt->infos = (detector_job_info *) xmalloc(size);
	if (!dt->infos) goto error_preinit;

	for (i = 0; i < num_cascades; i++) {
		dt->infos[i].dt = dt;
		dt->infos[i].id = i + 1;
		image_reset(&dt->infos[i].img);
		cascade_reset(&dt->infos[i].c);
	}

	for (i = 0; i < num_cascades; i++) {
		image_init(&dt->infos[i].img);
	}

	if (!tp) {
		if (!thread_pool_init(&dt->mtp, num_threads))
			goto error_preinit;
		dt->tp = &dt->mtp;
	} else {
		dt->tp = tp;
	}

	return TRUE;

error_preinit:
	detector_cleanup(dt);
	return FALSE;
}

int detector_init(detector *dt, unsigned int width, unsigned int height,
                  unsigned int num_parallels, unsigned int num_cascades,
                  unsigned int num_threads, thread_pool *tp)
{
	unsigned int i;

	if (!detector_preinit(dt, num_cascades, num_threads, tp))
		return FALSE;

	for (i = 0; i < num_cascades; i++) {
		if (!cascade_init(&dt->infos[i].c, width, height,
		                  num_parallels))
			goto error_init;
	}

	return TRUE;
error_init:
	detector_cleanup(dt);
	return FALSE;
}

void detector_cleanup(detector *dt)
{
	thread_pool_cleanup(&dt->mtp);
	dt->tp = NULL;

	if (dt->infos) {
		unsigned int i;
		for (i = 0; i < dt->num_cascades; i++) {
			image_cleanup(&dt->infos[i].img);
			cascade_cleanup(&dt->infos[i].c);
		}
		free(dt->infos);
		dt->infos = NULL;
	}
}

void detector_get_params(const detector *dt, double *scale, double *min_stddev,
                         unsigned int *step, double *match_thresh,
                         double *overlap_thresh, int *multi_exit)
{
	cascade_get_params(&dt->infos[0].c, scale, min_stddev, step,
	                   match_thresh, overlap_thresh, multi_exit);
}

void detector_set_params(detector *dt, double scale, double min_stddev,
                         unsigned int step, double match_thresh,
                         double overlap_thresh, int multi_exit)
{
	cascade_set_params(&dt->infos[0].c, scale, min_stddev, step,
	                   match_thresh, overlap_thresh, multi_exit);
}

void detector_set_scan(detector *dt,
                       unsigned int min_width, unsigned int min_height,
                       unsigned int max_width, unsigned int max_height)
{
	cascade_set_scan(&dt->infos[0].c, min_width, min_height,
	                 max_width, max_height);
}

int detector_prepare(detector *dt, detector_callback pre_fn,
                     detector_callback post_fn, int enforce_order)
{
	unsigned int i;
	for (i = 1; i < dt->num_cascades; i++) {
		if (!cascade_copy(&dt->infos[0].c, &dt->infos[i].c))
			return FALSE;
	}

	dt->done = 0;
	dt->free = 1;
	for (i = 0; i < dt->num_cascades; i++) {
		dt->infos[i].next = i + 2;
	}
	dt->infos[i - 1].next = 0;

	dt->curr_idx = 0;
	dt->done_idx = 0;
	dt->pre_fn = pre_fn;
	dt->post_fn = post_fn;
	dt->enforce_order = enforce_order;

	return TRUE;
}

static
void detector_job(void *arg)
{
	detector_job_info *info;
	detector *dt;
	cascade *c;
	image *img;

	info = (detector_job_info *) arg;
	dt = info->dt;
	c = &info->c;
	img = &info->img;

	info->success = FALSE;
	if (dt->pre_fn) {
		if (!dt->pre_fn(info))
			return;
	}

	if (!cascade_set_image(c, img))
		return;

	if (!cascade_detect(c, info->separate_detected))
		return;

	if (dt->post_fn) {
		if (!dt->post_fn(info))
			return;
	}
	info->success = TRUE;
}

int detector_enqueue(detector *dt, const image *img, void *extra,
                     int separate_detected)
{
	unsigned int id;
	detector_job_info *info;

	id = dt->free;
	if (id == 0) return 0;

	info = &dt->infos[id - 1];
	info->idx = dt->curr_idx;
	info->extra = extra;
	info->separate_detected = separate_detected;

	if (img) {
		if (!image_copy(img, &info->img)) {
			error("could not copy image");
			return -1;
		}
	}

	if (!thread_pool_enqueue(dt->tp, &detector_job, info)) {
		error("could not start detector job");
		return -1;
	}

	dt->free = info->next;
	dt->curr_idx++;
	return 1;
}

int detector_peek(const detector *dt)
{
	unsigned int id;

	if (!dt->enforce_order)
		return (dt->done != 0);

	for (id = dt->done; id; id = dt->infos[id - 1].next) {
		if (dt->infos[id - 1].idx == dt->done_idx)
			return TRUE;
	}

	return FALSE;
}

unsigned int detector_dequeue(detector *dt)
{
	detector_job_info *info;
	unsigned int id, pid;

	if (!dt->enforce_order) {
		if (dt->done) {
			id = dt->done;
			info = &dt->infos[id - 1];
			dt->done = info->next;
			info->next = 0;
			return id;
		}

		info = (detector_job_info *)
		       thread_pool_dequeue(dt->tp, FALSE);

		if (!info) {
			error("dequeue called on empty queue");
			return 0;
		}

		info->next = 0;
		return info->id;
	}

	pid = 0;
	for (id = dt->done; id; id = info->next) {
		info = &dt->infos[id - 1];
		if (info->idx == dt->done_idx) {
			if (pid) {
				dt->infos[pid - 1].next = info->next;
			} else {
				dt->done = info->next;
			}
			info->next = 0;
			dt->done_idx++;
			return id;
		}
		pid = id;
	}

	while (TRUE) {
		info = (detector_job_info *)
		       thread_pool_dequeue(dt->tp, FALSE);

		if (!info) {
			error("dequeue called on empty queue");
			return 0;
		}

		info->next = 0;
		if (info->idx == dt->done_idx) {
			dt->done_idx++;
			return info->id;
		}

		info->next = dt->done;
		dt->done = info->id;
	}
}

void detector_release(detector *dt, unsigned int id)
{
	dt->infos[id - 1].next = dt->free;
	dt->free = id;
}

int detector_pending(detector *dt, int remaining, int done)
{
	if (dt->done && done) return TRUE;
	return thread_pool_pending(dt->tp, remaining, done);
}

int detector_load(detector *dt, const char *filename, int reset,
                  unsigned int num_cascades, unsigned int num_threads)
{
	unsigned int i;
	cascade *c;

	if (reset) {
		if (!detector_preinit(dt, num_cascades, num_threads, NULL))
			return FALSE;
	} else {
		if (dt->num_cascades != num_cascades
		    || dt->num_threads != num_threads) {
			error("invalid parameters for detector load");
			goto error_load;
		}
	}

	c = &dt->infos[0].c;
	if (!cascade_load(c, filename, reset))
		goto error_load;

	if (reset) {
		for (i = 1; i < num_cascades; i++) {
			if (!cascade_init(&dt->infos[i].c,
			                  c->width, c->height,
			                  c->num_parallels))
				goto error_load;
		}
	}

	return TRUE;

error_load:
	if (reset) detector_cleanup(dt);
	return FALSE;
}

int detector_save(const detector *dt, const char *filename)
{
	return cascade_save(&dt->infos[0].c, filename);
}

int detector_load_sample_item(detector_job_info *info)
{
	sample_item *item = (sample_item *) info->extra;
	return image_read(&info->img, item->filename);
}

static
int process_sample_item(detector_job_info *info)
{
	unsigned int i, j, count;
	sample_item *item;
	cascade *c;

	c = &info->c;
	item = (sample_item *) info->extra;
	item->mark2 = c->num_detected_objects;
	if (!item->positive)
		return TRUE;

	count = item->same_last - item->same_first + 1;
	for (i = 0; i < c->num_detected_objects; i++) {
		window *w = &c->detected_objects[i].w;
		for (j = 0; j < count; j++) {
			sample_item *other;

			other = &item[j];
			if (other->mark1 == 1)
				continue;

			if (cascade_overlap(c, &other->w, w)) {
				other->mark1 = 1;
				break;
			}
		}
	}

	return TRUE;
}

static
int evaluate_sample_item(detector *dt, unsigned int *tfp, unsigned int *tfn,
                         unsigned int *tobjs)
{
	unsigned int id, i;
	unsigned int fp, fn, objs;
	sample_item *item;
	detector_job_info *info;

	id = detector_dequeue(dt);
	if (id == 0)
		return FALSE;

	info = &dt->infos[id - 1];
	if (!info->success) {
		detector_release(dt, id);
		return FALSE;
	}

	item = (sample_item *) info->extra;

	if (!item->positive) {
		fp = item->mark2;
		fn = 0;
		objs = 0;
	} else {
		fp = fn = 0;
		objs = item->same_last - item->same_first + 1;
		for (i = 0; i < objs; i++) {
			if (item[i].mark1 == 0)
				fn++;
		}
		fp = item->mark2 + fn - objs;
	}

	printf("%s -> fp: %u, fn: %u, objects: %u\n",
	       item->filename, fp, fn, objs);
	detector_release(dt, id);

	*tfp += fp;
	*tfn += fn;
	*tobjs += objs;
	return TRUE;
}

int detector_evaluate(detector *dt, const samples *smp,
                      const char *data_directory)
{
	unsigned int i;
	unsigned int tfp, tfn, tobjs;
	char current_directory[MAX_DIRECTORY_SIZE];

	tfp = tfn = tobjs = 0;
	if (!getcwd(current_directory, sizeof(current_directory))) {
		error("can not obtain current directory");
		return FALSE;
	}

	if (chdir(data_directory)) {
		error("missing directory `%s'", data_directory);
		return FALSE;
	}

	for (i = 0; i < smp->num_items; i++) {
		smp->items[i].mark1 = 0;
	}

	if (!detector_prepare(dt, &detector_load_sample_item,
	                      &process_sample_item, TRUE))
		return FALSE;

	for (i = 0; i < smp->num_items; i++) {
		sample_item *item;
		int ret;

		while (detector_peek(dt)) {
			if (!evaluate_sample_item(dt, &tfp, &tfn, &tobjs))
				return FALSE;
		}

		item = &smp->items[i];
		while (TRUE) {
			ret = detector_enqueue(dt, NULL, item, TRUE);
			if (ret < 0) return FALSE;
			if (ret > 0) break;

			if (!evaluate_sample_item(dt, &tfp, &tfn, &tobjs))
				return FALSE;
		}
		i = item->same_last;
	}

	while (detector_pending(dt, TRUE, TRUE)) {
		if (!evaluate_sample_item(dt, &tfp, &tfn, &tobjs))
			return FALSE;
	}

	if (chdir(current_directory)) {
		error("can not change directory back to `%s'",
		      current_directory);
		return FALSE;
	}

	printf("total fp = %u, total fn = %u, total objects = %u\n",
	       tfp, tfn, tobjs);
	return TRUE;
}
