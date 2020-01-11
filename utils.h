
#ifndef __UTILS_H
#define __UTILS_H

#include <stddef.h>

/* Useful macros */
#ifndef TRUE
#	define TRUE 1
#endif
#ifndef FALSE
#	define FALSE 0
#endif

#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define EPS                  1e-6
#define MAX_DIRECTORY_SIZE   1024

/* Functions */
void error(const char *fmt, ...);
void *xmalloc(size_t size);
void *xrealloc(void *ptr, size_t size);
char *xstrdup(const char *str);

#endif /* __UTILS_H */

