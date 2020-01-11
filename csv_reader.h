
#ifndef __CSV_READER_H
#define __CSV_READER_H

#include <stdio.h>

/* Error constants */
#define CSV_READER_NOERROR                0
#define CSV_READER_ERROR_MEMORY          -1
#define CSV_READER_ERROR_NOFILE          -2
#define CSV_READER_ERROR_INVALID_PNAME   -3
#define CSV_READER_ERROR_INVALID_PVALUE  -4
#define CSV_READER_ERROR_FIELD_MISMATCH  -5
#define CSV_READER_ERROR_PARSE           -6
#define CSV_READER_ERROR_NOTREADY        -7
#define CSV_READER_ERROR_INVALID_FIELD   -8
#define CSV_READER_ERROR_EOF             -9

/* Data structures */
typedef
struct csv_reader_st {
	int has_header;
	int check_field_count;
	char delimiter;
	char quote_char;

	FILE *fp;
	char *filename;
	int last_char;
	int ended;

	unsigned int row_num, field_num;
	unsigned int num_fields;
	unsigned int *fields;
	unsigned int *headers;
	unsigned int fields_len;

	char *buf, *hbuf;
	unsigned int buf_len, buf_pos;
	int last_error;
} csv_reader;

/* Functions */
void csv_reader_reset(csv_reader *csv);
int csv_reader_init(csv_reader *csv, const char *filename);
void csv_reader_close(csv_reader *csv);
int csv_reader_set_param(csv_reader *csv, const char *pname,
                         const char *pvalue);
int csv_reader_read(csv_reader *csv);

unsigned int csv_reader_num_fields(csv_reader *csv);
unsigned int csv_reader_row_number(csv_reader *csv);

int csv_reader_find_field(csv_reader *csv, const char *colname,
                          unsigned int *colnum);
int csv_reader_get_field(csv_reader *csv, unsigned int colnum,
                         const char **field, unsigned int *field_len);
int csv_reader_get_field_by_name(csv_reader *csv, const char *colname,
                                 const char **field, unsigned int *field_len);
const char *csv_reader_error(const csv_reader *csv);

#endif /* __CSV_READER_H */
