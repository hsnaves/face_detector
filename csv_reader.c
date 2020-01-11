
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "csv_reader.h"
#include "utils.h"

#define DEFAULT_BUFFER_LENGTH 4096
#define DEFAULT_FIELDS_LENGTH 32

void csv_reader_reset(csv_reader *csv)
{
	csv->fp = NULL;
	csv->filename = NULL;
	csv->fields = NULL;
	csv->headers = NULL;
	csv->buf = NULL;
	csv->hbuf = NULL;
	csv->last_error = CSV_READER_NOERROR;
}

int csv_reader_init(csv_reader *csv, const char *filename)
{
	csv_reader_reset(csv);

	csv->row_num = 1;
	csv->field_num = 1;
	csv->num_fields = 0;
	csv->last_char = EOF;
	csv->ended = FALSE;

	csv->delimiter = ',';
	csv->quote_char = '"';
	csv->has_header = FALSE;
	csv->check_field_count = FALSE;

	csv->filename = xstrdup(filename);
	if (!csv->filename) {
		csv_reader_close(csv);
		csv->last_error = CSV_READER_ERROR_MEMORY;
		return FALSE;
	}

	csv->fields = xmalloc(DEFAULT_FIELDS_LENGTH);
	csv->fields_len = DEFAULT_FIELDS_LENGTH;
	if (!csv->fields) {
		csv_reader_close(csv);
		csv->last_error = CSV_READER_ERROR_MEMORY;
		return FALSE;
	}

	csv->buf = xmalloc(DEFAULT_BUFFER_LENGTH);
	csv->buf_len = DEFAULT_BUFFER_LENGTH;
	csv->buf_pos = 0;
	if (!csv->buf) {
		csv_reader_close(csv);
		csv->last_error = CSV_READER_ERROR_MEMORY;
		return FALSE;
	}

	csv->fp = fopen(filename, "rb");
	if (!csv->fp) {
		csv_reader_close(csv);
		csv->last_error = CSV_READER_ERROR_NOFILE;
		return FALSE;
	}

	return TRUE;
}

void csv_reader_close(csv_reader *csv)
{
	if (csv->filename) {
		free(csv->filename);
		csv->filename = NULL;
	}

	if (csv->fields) {
		free(csv->fields);
		csv->fields = NULL;
	}

	if (csv->headers) {
		free(csv->headers);
		csv->headers = NULL;
	}

	if (csv->buf) {
		free(csv->buf);
		csv->buf = NULL;
	}

	if (csv->hbuf) {
		free(csv->hbuf);
		csv->hbuf = NULL;
	}

	if (csv->fp) {
		fclose(csv->fp);
		csv->fp = NULL;
	}
}

int csv_reader_set_param(csv_reader *csv, const char *pname,
                         const char *pvalue)
{
	if (strcmp("delimiter", pname) == 0) {
		if (csv->quote_char == pvalue[0]) {
			csv->last_error = CSV_READER_ERROR_INVALID_PVALUE;
			return FALSE;
		}

		csv->delimiter = pvalue[0];
	} else if (strcmp("quote_char", pname) == 0) {
		if (csv->delimiter == pvalue[0]) {
			csv->last_error = CSV_READER_ERROR_INVALID_PVALUE;
			return FALSE;
		}

		csv->quote_char = pvalue[0];
	} else if (strcmp("has_header", pname) == 0) {
		csv->has_header = (strcmp("yes", pvalue) == 0);
	} else if (strcmp("check_field_count", pname) == 0) {
		csv->check_field_count = (strcmp("yes", pvalue) == 0);
	} else {
		csv->last_error = CSV_READER_ERROR_INVALID_PNAME;
		return FALSE;
	}

	csv->last_error = CSV_READER_NOERROR;
	return TRUE;
}

static
int csv_reader_putc(csv_reader *csv, char c)
{
	if (csv->buf_pos >= csv->buf_len) {
		void *ptr = xrealloc(csv->buf, 2 * csv->buf_len);
		if (!ptr) {
			csv->last_error = CSV_READER_ERROR_MEMORY;
			return FALSE;
		}
		csv->buf_len *= 2;
		csv->buf = (char *) ptr;
	}
	csv->buf[csv->buf_pos++] = c;
	return TRUE;
}

static
int csv_reader_put_field(csv_reader *csv, unsigned int pos)
{
	if (csv->field_num > csv->fields_len) {
		void *ptr = xrealloc(csv->fields, 2 * csv->fields_len);
		if (!ptr) {
			csv->last_error = CSV_READER_ERROR_MEMORY;
			return FALSE;
		}
		csv->fields_len *= 2;
		csv->fields = (unsigned int *) ptr;
	}
	csv->fields[csv->field_num - 1] = pos;
	return TRUE;
}

static
int csv_reader_start_row(csv_reader *csv)
{
	csv->buf_pos = 0;
	csv->field_num = 1;
	return TRUE;
}

static
int csv_reader_end_row(csv_reader *csv)
{
	if ((csv->row_num)++ == 1) {
		csv->num_fields = csv->field_num - 1;
		if (csv->has_header) {
			unsigned int size;
			void *ptr;
			ptr = xmalloc(csv->buf_len);
			if (!ptr) {
				csv->last_error = CSV_READER_ERROR_MEMORY;
				return FALSE;
			}

			memcpy(ptr, csv->buf, csv->buf_len);
			csv->hbuf = (char *) ptr;

			size = csv->num_fields;
			size *= ((unsigned int) sizeof(unsigned int));
			ptr = xmalloc(size);
			if (!ptr) {
				csv->last_error = CSV_READER_ERROR_MEMORY;
				return FALSE;
			}
			memcpy(ptr, csv->fields, size);
			csv->headers = (unsigned int *) ptr;
		}
	} else {
		if (csv->check_field_count
		    && csv->field_num != (csv->num_fields + 1)) {
			csv->last_error = CSV_READER_ERROR_FIELD_MISMATCH;
			return FALSE;
		}
	}
	return TRUE;
}

static
int csv_reader_start_field(csv_reader *csv)
{
	return csv_reader_put_field(csv, csv->buf_pos);
}

static
int csv_reader_end_field(csv_reader *csv)
{
	csv->field_num++;
	return csv_reader_putc(csv, '\0');
}

int csv_reader_read(csv_reader *csv)
{
	int c;
	int state;

	csv->last_error = CSV_READER_NOERROR;
	if (csv->ended) {
		csv->last_error = CSV_READER_ERROR_EOF;
		return FALSE;
	}

	state = -1;
	while (TRUE) {
		if (csv->last_char != EOF) {
			c = csv->last_char;
			csv->last_char = EOF;
		} else {
			c = fgetc(csv->fp);
		}

		if (c == EOF) {
			csv->ended = TRUE;
			if (state == -1) {
				csv->last_error = CSV_READER_ERROR_EOF;
				return FALSE;
			}
		} else if (state == -1) {
			state = 0;

			if (!csv_reader_start_row(csv))
				return FALSE;

			if (!csv_reader_start_field(csv))
				return FALSE;
		}

		if (state == 5) {
			if (c != '\n' && c != EOF) {
				csv->last_char = c;
			}
			return TRUE;
		}

		if (c == csv->delimiter) {
			if (state == 2) {
				if (!csv_reader_putc(csv, (char) c))
					return FALSE;
			} else {
				if (!csv_reader_end_field(csv))
					return FALSE;

				if (!csv_reader_start_field(csv))
					return FALSE;

				state = 0;
			}
		} else if (c == csv->quote_char) {
			if (state == 0) {
				state = 2;
			} else if (state == 1) {
				csv->last_error = CSV_READER_ERROR_PARSE;
				return FALSE;
			} else if (state == 2) {
				state = 3;
			} else if (state == 3) {
				if (!csv_reader_putc(csv, csv->quote_char))
					return FALSE;

				state = 2;
			}
		} else if (c == '\r' || c == '\n' || c == EOF) {
			if (state == 2) {
				if (c == EOF) {
					csv->last_error =
					     CSV_READER_ERROR_PARSE;
					return FALSE;
				}

				if (!csv_reader_putc(csv, (char) c))
					return FALSE;
			} else {
				if (!csv_reader_end_field(csv))
					return FALSE;

				if (!csv_reader_end_row(csv))
					return FALSE;

				if (c == '\r')
					state = 5;
				else
					return TRUE;
			}
		} else {
			if (state == 3) {
				csv->last_error = CSV_READER_ERROR_PARSE;
				return FALSE;
			}

			if (!csv_reader_putc(csv, (char) c))
				return FALSE;

			if (state == 0)
				state = 1;
		}
	}
}

unsigned int csv_reader_num_fields(csv_reader *csv)
{
	csv->last_error = CSV_READER_NOERROR;
	return csv->field_num - 1;
}

unsigned int csv_reader_row_number(csv_reader *csv)
{
	csv->last_error = CSV_READER_NOERROR;
	return csv->row_num - 1;
}

int csv_reader_find_field(csv_reader *csv, const char *fieldname,
                          unsigned int *fieldnum)
{
	unsigned int i;

	csv->last_error = CSV_READER_NOERROR;
	if (!csv->hbuf || !csv->headers) {
		csv->last_error = CSV_READER_ERROR_NOTREADY;
		return FALSE;
	}

	for (i = 0; i < csv->num_fields; i++) {
		if (strcmp(&csv->hbuf[csv->headers[i]], fieldname) == 0) {
			*fieldnum = i + 1;
			return TRUE;
		}
	}

	csv->last_error = CSV_READER_ERROR_INVALID_FIELD;
	return FALSE;
}

int csv_reader_get_field(csv_reader *csv, unsigned int fieldnum,
                         const char **field, unsigned int *field_len)
{
	csv->last_error = CSV_READER_NOERROR;
	if (fieldnum >= csv->field_num || fieldnum == 0) {
		csv->last_error = CSV_READER_ERROR_INVALID_FIELD;
		return FALSE;
	}

	if (fieldnum == (csv->field_num - 1)) {
		*field_len = csv->buf_len - csv->fields[fieldnum - 1] - 1;
	} else {
		*field_len = csv->fields[fieldnum]
		    - csv->fields[fieldnum - 1] - 1;
	}
	*field = (const char *) &csv->buf[csv->fields[fieldnum - 1]];
	return TRUE;
}

int csv_reader_get_field_by_name(csv_reader *csv, const char *fieldname,
                                 const char **field, unsigned int *field_len)
{
	unsigned int fieldnum;
	if (!csv_reader_find_field(csv, fieldname, &fieldnum))
		return FALSE;

	return csv_reader_get_field(csv, fieldnum, field, field_len);
}

static
const char *error_codes[] = {
	"ok",
	"memory exhausted",
	"file does not exist or can not be opened",
	"invalid property name",
	"invalid property value",
	"invalid number of fields",
	"parse error",
	"waiting to parse header line",
	"invalid field name",
	"end of file",
};

const char *csv_reader_error(const csv_reader *csv)
{
	int error_code = csv->last_error;
	if (error_code > 0 || error_code < CSV_READER_ERROR_EOF)
		return "invalid";
	return error_codes[-error_code];
}

