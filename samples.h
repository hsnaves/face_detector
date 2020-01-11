
#ifndef __SAMPLES_H
#define __SAMPLES_H

#include "window.h"

/* Data structures */
typedef
struct sample_item_st {
	window w;
	int positive;
	unsigned int same_first, same_last;
	char *filename;

	unsigned int mark1, mark2;
	double similarity;
} sample_item;

typedef
struct string_buffer_chunk_st {
	char *buffer;
	struct string_buffer_chunk_st *prev;
	unsigned int position;
	unsigned int length;
} string_buffer_chunk;

typedef
struct samples_st {
	unsigned int num_items;
	unsigned int capacity;
	sample_item *items;

	string_buffer_chunk *chunk;
} samples;

/* Functions */
void samples_reset(samples *smp);
int samples_init(samples *smp);
void samples_cleanup(samples *smp);
int samples_read(samples *smp, const char *filename);

#endif /* __SAMPLES_H */
