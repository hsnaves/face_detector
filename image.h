#ifndef __IMAGE_H
#define __IMAGE_H

#include "window.h"

/* Data structures */
typedef
struct image_st {
	unsigned int width, height, stride;
	unsigned int capacity;
	unsigned char *pixels;
} image;

/* Functions */
void image_reset(image *img);
void image_init(image *img);
int image_allocate(image *img, unsigned int width, unsigned int height);
void image_cleanup(image *img);

int image_copy(const image *from, image *to);
int image_resize(const image *img, image *t,
                 unsigned int width, unsigned int height);

int image_read(image *img, const char *filename);
int image_write(const image *img, const char *filename);
void image_draw_window(image *img, const window *w,
                       unsigned char color, unsigned int thickness);

#endif /* __IMAGE_H */
