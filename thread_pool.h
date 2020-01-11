
#ifndef __THREAD_POOL_H
#define __THREAD_POOL_H

#include <pthread.h>

/* Data structures and types */
typedef void (*job_cb)(void *arg);

typedef
struct job_item_st {
	struct job_item_st *next;
	void *arg;
	job_cb cb;
} job_item;

typedef
struct thread_pool_st {
	int initialized, cancelled;
	unsigned int num_remaining, num_done;
	unsigned int num_threads;
	job_item *first, *done_first;
	job_item *last, *done_last;
	job_item *allocated, *free;
	pthread_mutex_t q_mtx;
	pthread_cond_t q_cnd, q_cnd_master;
	pthread_t *threads;
} thread_pool;

/* Functions */
void thread_pool_reset(thread_pool *tp);
int thread_pool_init(thread_pool *tp, unsigned int num_threads);
void thread_pool_cleanup(thread_pool *tp);
int thread_pool_enqueue(thread_pool *tp, job_cb cb, void *arg);
void *thread_pool_dequeue(thread_pool *tp, int wait);
void thread_pool_wait(thread_pool *tp);
int thread_pool_pending(thread_pool *tp, int remaining, int done);
void thread_pool_flush_done(thread_pool *tp);

#endif /* __THREAD_POOL_H */
