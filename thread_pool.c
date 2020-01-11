
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#include "thread_pool.h"
#include "utils.h"

#define NUM_JOBS   16

static void *worker_function(void *arg);
static void pool_end(thread_pool *tp);

void thread_pool_reset(thread_pool *tp)
{
	tp->initialized = FALSE;
	tp->threads = NULL;
	tp->first = NULL;
	tp->last = NULL;
	tp->done_first = NULL;
	tp->done_last = NULL;
	tp->allocated = NULL;
	tp->free = NULL;
}

int thread_pool_init(thread_pool *tp, unsigned int num_threads)
{
	unsigned int i = 0;

	thread_pool_reset(tp);
	if (pthread_mutex_init(&tp->q_mtx, NULL)) {
		error("can't create mutex");
		return FALSE;
	}
	if (pthread_cond_init(&tp->q_cnd, NULL)) {
		error("can't create condition");
		pthread_mutex_destroy(&tp->q_mtx);
		return FALSE;
	}
	if (pthread_cond_init(&tp->q_cnd_master, NULL)) {
		error("can't create master condition");
		pthread_mutex_destroy(&tp->q_mtx);
		pthread_cond_destroy(&tp->q_cnd);
		return FALSE;
	}

	tp->num_threads = num_threads;
	tp->cancelled = FALSE;
	tp->num_remaining = 0;
	tp->num_done = 0;

	tp->threads = (pthread_t *) xmalloc(num_threads * sizeof(pthread_t));
	if (!tp->threads) goto error_init;

	for (i = 0; i < num_threads; i++) {
		if (pthread_create(&tp->threads[i], NULL,
		                   &worker_function, tp)) {
			error("can't create thread");
			goto error_init;
		}
	}

	tp->initialized = TRUE;
	return TRUE;

error_init:
	pthread_mutex_destroy(&tp->q_mtx);
	pthread_cond_destroy(&tp->q_cnd);
	pthread_cond_destroy(&tp->q_cnd_master);

	if (tp->threads) {
		tp->num_threads = i;
		pool_end(tp);
		free(tp->threads);
		tp->threads = NULL;
	}

	return FALSE;
}

static
void pool_end(thread_pool *tp)
{
	unsigned int i;

	pthread_mutex_lock(&tp->q_mtx);
	tp->cancelled = TRUE;
	pthread_cond_broadcast(&tp->q_cnd);
	pthread_mutex_unlock(&tp->q_mtx);

	for (i = 0; i < tp->num_threads; i++) {
		pthread_join(tp->threads[i], NULL);
	}
}

void thread_pool_cleanup(thread_pool *tp)
{
	if (!tp->initialized) return;

	pool_end(tp);
	pthread_mutex_destroy(&tp->q_mtx);
	pthread_cond_destroy(&tp->q_cnd);
	pthread_cond_destroy(&tp->q_cnd_master);
	free(tp->threads);

	while (tp->allocated) {
		job_item *job = tp->allocated;
		tp->allocated = job->next;
		free(job);
	}

	thread_pool_reset(tp);
}

static
job_item *allocate_new_job(thread_pool *tp)
{
	job_item *job;
	if (!tp->free) {
		unsigned int i;

		job = (job_item *) xmalloc(NUM_JOBS * sizeof(job_item));
		if (!job) return NULL;

		job[0].next = tp->allocated;
		tp->allocated = job;

		for (i = 1; i < NUM_JOBS; i++) {
			job[i].next = tp->free;
			tp->free = &job[i];
		}
	}

	job = tp->free;
	tp->free = job->next;
	return job;
}

int thread_pool_enqueue(thread_pool *tp, job_cb cb, void *arg)
{
	job_item *job;

	pthread_mutex_lock(&tp->q_mtx);
	job = allocate_new_job(tp);
	if (!job) {
		pthread_mutex_unlock(&tp->q_mtx);
		return FALSE;
	}

	job->cb = cb;
	job->arg = arg;
	job->next = NULL;

	if (tp->last != NULL) tp->last->next = job;
	if (tp->first == NULL) tp->first = job;
	tp->last = job;
	tp->num_remaining++;
	pthread_cond_signal(&tp->q_cnd);
	pthread_mutex_unlock(&tp->q_mtx);

	return TRUE;
}

void *thread_pool_dequeue(thread_pool *tp, int wait)
{
	void *arg;
	pthread_mutex_lock(&tp->q_mtx);
	while (!tp->cancelled && tp->num_done == 0
	       && (wait || tp->num_remaining > 0)) {
		pthread_cond_wait(&tp->q_cnd_master, &tp->q_mtx);
	}
	if (tp->num_done == 0) {
		arg = NULL;
	} else {
		job_item *job = tp->done_first;
		tp->done_first = job->next;
		if (!tp->done_first)
			tp->done_last = NULL;
		tp->num_done--;

		arg = job->arg;

		job->next = tp->free;
		tp->free = job;
	}
	pthread_mutex_unlock(&tp->q_mtx);
	return arg;
}

void thread_pool_wait(thread_pool *tp)
{
	pthread_mutex_lock(&tp->q_mtx);
	while (!tp->cancelled && tp->num_remaining > 0) {
		pthread_cond_wait(&tp->q_cnd_master, &tp->q_mtx);
	}
	pthread_mutex_unlock(&tp->q_mtx);
}

int thread_pool_pending(thread_pool *tp, int remaining, int done)
{
	int ret;
	pthread_mutex_lock(&tp->q_mtx);
	if (tp->cancelled) ret = FALSE;
	else if (tp->num_remaining > 0 && remaining) ret = TRUE;
	else if (tp->num_done > 0 && done) ret = TRUE;
	else ret = FALSE;
	pthread_mutex_unlock(&tp->q_mtx);
	return ret;
}

void thread_pool_flush_done(thread_pool *tp)
{
	pthread_mutex_lock(&tp->q_mtx);
	while (tp->done_first) {
		job_item *job = tp->done_first;
		tp->done_first = job->next;
		job->next = tp->free;
		tp->free = job;
	}
	tp->num_done = 0;
	tp->done_last = NULL;
	pthread_mutex_unlock(&tp->q_mtx);
}

static
void *worker_function(void *arg)
{
	thread_pool *tp = (thread_pool *) arg;
	job_item *job;

	while (TRUE) {
		pthread_mutex_lock(&tp->q_mtx);
		while (!tp->cancelled && tp->first == NULL) {
			pthread_cond_wait(&tp->q_cnd, &tp->q_mtx);
		}
		if (tp->cancelled) {
			pthread_mutex_unlock(&tp->q_mtx);
			return NULL;
		}
		job = tp->first;
		tp->first = job->next;
		tp->last = (job == tp->last ? NULL : tp->last);
		pthread_mutex_unlock(&tp->q_mtx);

		job->cb(job->arg);

		pthread_mutex_lock(&tp->q_mtx);
		tp->num_remaining--;
		job->next = NULL;
		if (tp->done_last) tp->done_last->next = job;
		if (tp->done_first == NULL) tp->done_first = job;
		tp->done_last = job;
		tp->num_done++;
		pthread_cond_broadcast(&tp->q_cnd_master);
		pthread_mutex_unlock(&tp->q_mtx);
	}
	return NULL;
}
