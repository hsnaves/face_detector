
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "utils.h"

void error(const char *fmt, ...)
{
	va_list ap;
	va_start(ap, fmt);
	fprintf(stderr, "error: ");
	vfprintf(stderr, fmt, ap);
	fprintf(stderr, "\n");
	va_end(ap);
}

void *xmalloc(size_t size)
{
	void *ptr = malloc(size);
	if (!ptr) error("memory exhausted");
	return ptr;
}

void *xrealloc(void *ptr, size_t size)
{
	void *nptr = realloc(ptr, size);
	if (!nptr) error("memory exhausted");
	return nptr;
}

char *xstrdup(const char *str)
{
	size_t size;
	char *s;

	size = strlen(str);
	s = xmalloc(size + 1);
	if (s) memcpy(s, str, size + 1);
	return s;
}
