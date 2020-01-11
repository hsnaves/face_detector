
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "samples.h"
#include "window.h"
#include "csv_reader.h"
#include "utils.h"

#define NUM_ITEMS       1024
#define CHUNK_SIZE      4096

void samples_reset(samples *smp)
{
	smp->items = NULL;
	smp->chunk = NULL;
}

int samples_init(samples *smp)
{
	char *ptr;

	samples_reset(smp);
	ptr = (char *) xmalloc(NUM_ITEMS * sizeof(sample_item));
	if (!ptr) {
		samples_cleanup(smp);
		return FALSE;
	}
	smp->items = (sample_item *) ptr;
	smp->num_items = 0;
	smp->capacity = NUM_ITEMS;

	ptr = (char *) xmalloc(CHUNK_SIZE);
	if (!ptr) {
		samples_cleanup(smp);
		return FALSE;
	}
	smp->chunk = (string_buffer_chunk *) ptr;
	smp->chunk->buffer = &ptr[sizeof(string_buffer_chunk)];
	smp->chunk->prev = NULL;
	smp->chunk->position = 0;
	smp->chunk->length = CHUNK_SIZE - sizeof(string_buffer_chunk);
	return TRUE;
}

void samples_cleanup(samples *smp)
{
	string_buffer_chunk *chunk, *prev;

	if (smp->items) {
		free(smp->items);
		smp->items = NULL;
	}

	chunk = smp->chunk;
	smp->chunk = NULL;
	while (chunk) {
		prev = chunk->prev;
		free(chunk);
		chunk = prev;
	}
}

static
char *samples_strdup(samples *smp, const char *str, unsigned int len)
{
	unsigned int capacity;
	string_buffer_chunk *chunk;
	char *ptr;

	chunk = smp->chunk;
	if (!chunk) return NULL;

	capacity = chunk->length - chunk->position;
	if (len >= capacity) {
		unsigned int size;

		size = len + 1 + ((unsigned int) sizeof(string_buffer_chunk));
		if (size < CHUNK_SIZE)
			size = CHUNK_SIZE;
		ptr = (char *) xmalloc(size);
		if (!ptr) return NULL;

		chunk = (string_buffer_chunk *) ptr;
		chunk->buffer = &ptr[sizeof(string_buffer_chunk)];
		chunk->position = 0;
		chunk->length = size;
		chunk->length -= (unsigned int) sizeof(string_buffer_chunk);
		chunk->prev = smp->chunk;
		smp->chunk = chunk;
	}

	chunk = smp->chunk;
	ptr = &chunk->buffer[chunk->position];
	memcpy(ptr, str, len);
	chunk->position += len + 1;
	ptr[len] = '\0';
	return ptr;
}

static
sample_item *samples_new_item(samples *smp)
{
	if (smp->num_items >= smp->capacity) {
		size_t new_size;
		void *ptr;

		new_size = 2 * smp->capacity * sizeof(sample_item);
		ptr = xrealloc(smp->items, new_size);
		if (!ptr) return NULL;
		smp->items = (sample_item *) ptr;
		smp->capacity *= 2;
	}
	return &smp->items[smp->num_items++];
}

static
int cmp_items(const void *p1, const void *p2)
{
	const sample_item *i1 = (const sample_item *) p1;
	const sample_item *i2 = (const sample_item *) p2;
	if (i1->positive && !i2->positive) return -1;
	if (!i1->positive && i2->positive) return +1;
	return strcmp(i1->filename, i2->filename);
}

static
void samples_sort(samples *smp)
{
	char *filename;
	unsigned int i, j, first;

	if (smp->num_items == 0)
		return;

	qsort(smp->items, smp->num_items, sizeof(sample_item), &cmp_items);

	first = 0;
	filename = smp->items[0].filename;
	smp->items[0].same_first = first;
	for (i = 1; i < smp->num_items; i++) {
		if (strcmp(filename, smp->items[i].filename) == 0) {
			smp->items[i].same_first = first;
			continue;
		}
		for (j = first; j < i; j++)
			smp->items[j].same_last = i - 1;
		first = i;
		filename = smp->items[i].filename;
		smp->items[i].same_first = first;
	}

	for (j = first; j < i; j++)
		smp->items[j].same_last = i - 1;
}

int samples_read(samples *smp, const char *filename)
{
	const char *fieldname, *pname, *pvalue, *errstr;
	const char *pnames[] = { "delimiter", "quote_char",
	                         "has_header", "check_field_count" };
	const char *pvals[] = { ",", "\"", "yes", "yes" };
	unsigned int i;
	csv_reader csv;

	if (!samples_init(smp))
		return FALSE;

	if (!csv_reader_init(&csv, filename)) {
		error("csv_reader_init(`%s'): %s",
		      filename, csv_reader_error(&csv));
		goto error_read;
	}

	for (i = 0; i < sizeof(pnames) / sizeof(char *); i++) {
		pname = pnames[i];
		pvalue = pvals[i];
		if (!csv_reader_set_param(&csv, pname, pvalue)) {
			error("csv_reader_set_param: could not set `%s' "
			      "to `%s': %s", pname, pvalue,
			      csv_reader_error(&csv));
			goto error_read;
		}
	}

	while (TRUE) {
		sample_item *item;
		unsigned int len;
		const char *field;

		if (!csv_reader_read(&csv)) {
			if (csv.last_error == CSV_READER_ERROR_EOF)
				break;
			else {
				errstr = csv_reader_error(&csv);
				error("csv_reader_read: %s at line %u",
				      errstr, csv_reader_row_number(&csv));
				goto error_read;
			}
		}

		if (csv_reader_row_number(&csv) == 1)
			continue;

		item = samples_new_item(smp);
		if (!item) goto error_read;

		fieldname = "positive";
		if (!csv_reader_get_field_by_name(&csv, fieldname,
		                                  &field, &len))
			goto error_field;
		item->positive = (strcmp("y", field) == 0);

		fieldname = "filename";
		if (!csv_reader_get_field_by_name(&csv, fieldname,
		                                  &field, &len))
			goto error_field;
		item->filename = samples_strdup(smp, field, len);
		if (!item->filename) goto error_read;

		fieldname = "left";
		if (!csv_reader_get_field_by_name(&csv, fieldname,
		                                  &field, &len))
			goto error_field;
		item->w.left = (unsigned int) strtol(field, NULL, 10);

		fieldname = "top";
		if (!csv_reader_get_field_by_name(&csv, fieldname,
		                                  &field, &len))
			goto error_field;
		item->w.top = (unsigned int) strtol(field, NULL, 10);

		fieldname = "width";
		if (!csv_reader_get_field_by_name(&csv, fieldname,
		                                  &field, &len))
			goto error_field;
		item->w.width = (unsigned int) strtol(field, NULL, 10);

		fieldname = "height";
		if (!csv_reader_get_field_by_name(&csv, fieldname,
		                                  &field, &len))
			goto error_field;
		item->w.height = (unsigned int) strtol(field, NULL, 10);
	}

	csv_reader_close(&csv);
	samples_sort(smp);
	return TRUE;

error_field:
	errstr = csv_reader_error(&csv);
	error("csv_reader_get_field(`%s'): %s at line %u",
	      fieldname, errstr, csv_reader_row_number(&csv));
error_read:
	samples_cleanup(smp);
	csv_reader_close(&csv);
	return FALSE;
}

