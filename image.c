
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* #include <setjmp.h> */
#include <png.h>
#include <zlib.h>
#include <jpeglib.h>
#include <jerror.h>

#include "image.h"
#include "window.h"
#include "utils.h"

void image_reset(image *img)
{
	img->pixels = NULL;
}

void image_init(image *img)
{
	image_reset(img);
	img->width = 0;
	img->height = 0;
	img->stride = 0;
	img->capacity = 0;
}

void image_cleanup(image *img)
{
	if (img->pixels) {
		free(img->pixels);
		img->pixels = NULL;
	}
}

int image_allocate(image *img, unsigned int width, unsigned int height)
{
	unsigned int size;

	size = width * height;
	if (img->capacity < size) {
		void *ptr;
		ptr = xrealloc(img->pixels, size);
		if (!ptr) return FALSE;
		img->pixels = (unsigned char *) ptr;
		img->capacity = size;
	}

	img->width = width;
	img->height = height;
	img->stride = width;
	return TRUE;
}

int image_copy(const image *from, image *to)
{
	if (!image_allocate(to, from->width, from->height))
		return FALSE;

	if (from->stride == from->width) {
		memcpy(to->pixels, from->pixels, from->width * from->height);
	} else {
		unsigned int row;
		for (row = 0; row < to->height; row++) {
			memcpy(&to->pixels[row * to->stride],
			       &from->pixels[row * from->stride],
			       from->width);
		}
	}
	return TRUE;
}

int image_resize(const image *img, image *t,
                 unsigned int width, unsigned int height)
{
	unsigned int row, col, trow, tcol, pos, tpos;
	unsigned int stride, tstride;
	unsigned int drow, dcol;
	unsigned char *pxls;
#ifdef USE_LINEAR_FILTER
	double x, y, dx, dy;
#else
	unsigned int x, y, dx, dy;
#endif

	if (!image_allocate(t, width, height))
		return FALSE;

	stride = img->stride;
	tstride = t->stride;

	drow = img->height / height;
	dcol = img->width / width;

#ifdef USE_LINEAR_FILTER
	dy = ((double) img->height) / height;
	dy -= drow;
	dx = ((double) img->width) / width;
	dx -= dcol;
#else
	dy = img->height % height;
	dx = img->width % width;
#endif

	pxls = img->pixels;
	row = 0;
	y = 0;
	for (trow = 0; trow < height; trow++) {
		col = 0;
		x = 0;
		for (tcol = 0; tcol < width; tcol++) {
#ifdef USE_LINEAR_FILTER
			double val;
#endif
			pos = stride * row + col;
			tpos = tstride * trow + tcol;
#ifdef USE_LINEAR_FILTER
			val = (1 - x) * (1 - y) * ((double) pxls[pos]);

			if (x > EPS) {
				if (y > EPS) {
					val += x * (1 - y)
					 * ((double) pxls[pos + 1]);
					val += y * (1 - x)
					 * ((double) pxls[pos + stride]);
					val += x * y
					 * ((double) pxls[pos + stride + 1]);
				} else {
					val += x
					 * ((double) pxls[pos + 1]);
				}
			} else {
				if (y > EPS) {
					val += y
					 * ((double) pxls[pos + stride]);
				}
			}
			t->pixels[tpos] = (unsigned char) val;
#else /* !USE_LINEAR_FILTER */
			t->pixels[tpos] = pxls[pos];
#endif

			col += dcol;
			x += dx;
#ifdef USE_LINEAR_FILTER
			if (x >= 1) {
				x -= 1;
#else
			if (x >= width) {
				x -= width;
#endif
				col++;
			}
		}

		row += drow;
		y += dy;
#ifdef USE_LINEAR_FILTER
		if (y >= 1) {
			y -= 1;
#else
		if (y >= height) {
			y -= height;
#endif
			row++;
		}
	}

	return TRUE;
	return TRUE;
}

struct my_jpeg_error_mgr {
	struct jpeg_error_mgr pub;
	jmp_buf setjmp_buffer;
};
typedef struct my_jpeg_error_mgr *my_jpeg_error_ptr;

static
void my_jpeg_error_exit(j_common_ptr cinfo)
{
	my_jpeg_error_ptr myerr = (my_jpeg_error_ptr) cinfo->err;
	(*cinfo->err->output_message)(cinfo);
	longjmp(myerr->setjmp_buffer, 1);
}

static
void my_jpeg_output_message(j_common_ptr cinfo)
{
	char buffer[JMSG_LENGTH_MAX];
	(*cinfo->err->format_message)(cinfo, buffer);
	fprintf(stderr, "%s\n", buffer);
}

static
int read_jpeg_file(image *img, const char *filename)
{
	struct jpeg_decompress_struct cinfo;
	struct my_jpeg_error_mgr jerr;

	JSAMPARRAY buffer;
	JDIMENSION stride;
	FILE *fp;

	fp = fopen(filename, "rb");
	if (!fp) {
		error("can't open `%s'", filename);
		return FALSE;
	}

	/* Step 1: allocate and initialize JPEG decompression object */

	/* We set up the normal JPEG error routines, then override both
	 * error_exit and output_message.
	 */
	cinfo.err = jpeg_std_error(&jerr.pub);
	jerr.pub.error_exit = &my_jpeg_error_exit;
	jerr.pub.output_message = &my_jpeg_output_message;
	if (setjmp(jerr.setjmp_buffer)) {
		jpeg_destroy_decompress(&cinfo);
		fclose(fp);
		return FALSE;
	}
	/* Now we can initialize the JPEG decompression object. */
	jpeg_create_decompress(&cinfo);

	/* Step 2: specify data source (eg, a file) */
	jpeg_stdio_src(&cinfo, fp);

	/* Step 3: read file parameters with jpeg_read_header() */
	(void) jpeg_read_header(&cinfo, TRUE);
	/* We can ignore the return value from jpeg_read_header since
	 *   (a) suspension is not possible with the stdio data source, and
	 *   (b) we passed TRUE to reject a tables-only JPEG file as an error.
	 * See libjpeg.txt for more info.
	 */

	/* Step 4: set parameters for decompression */
	cinfo.out_color_space = JCS_GRAYSCALE;

	/* Step 5: Allocate some auxiliary memory */
	jpeg_calc_output_dimensions(&cinfo);
	stride = cinfo.output_width;
	stride *= (JDIMENSION) cinfo.output_components;
	buffer = (*cinfo.mem->alloc_sarray)
	       ((j_common_ptr) &cinfo, JPOOL_IMAGE, stride, 1);

	if (!image_allocate(img, cinfo.output_width, cinfo.output_height)) {
		jpeg_destroy_decompress(&cinfo);
		fclose(fp);
		return FALSE;
	}

	if (setjmp(jerr.setjmp_buffer)) {
		image_cleanup(img);
		jpeg_destroy_decompress(&cinfo);
		fclose(fp);
		return FALSE;
	}

	/* Step 6: Start decompressor */
	(void) jpeg_start_decompress(&cinfo);
	/* We can ignore the return value since suspension is not possible
	 * with the stdio data source.
	 */

	/* Step 7: while (scan lines remain to be read) */
	/*           jpeg_read_scanlines(...); */

	while (cinfo.output_scanline < cinfo.output_height) {
		(void) jpeg_read_scanlines(&cinfo, buffer, 1);
		memcpy(&img->pixels[img->stride
		          * (cinfo.output_scanline - 1)], buffer[0],
		       img->width);
	}

	/* Step 8: Finish decompression */
	(void) jpeg_finish_decompress(&cinfo);
	/* We can ignore the return value since suspension is not possible
	 * with the stdio data source.
	 */

	/* Step 9: Release JPEG decompression object */
	jpeg_destroy_decompress(&cinfo);
	fclose(fp);

	/* At this point you may want to check to see whether any corrupt-data
	 * warnings occurred (test whether jerr.pub.num_warnings is nonzero).
	 */

	/* And we're done! */
	return TRUE;
}

static
int read_png_file(image *img, const char *filename)
{
	png_structp png_ptr;
	png_infop info_ptr;
	unsigned char header[8]; /* 8 is the max size that can be checked */
	png_uint_32 width, height, y;
	png_byte color_type; /*, bit_depth; */
	png_bytep *row_pointers;

	/* open file and test for it being a png */
	FILE *fp = fopen(filename, "rb");
	if (!fp) {
		error("can't open `%s'", filename);
		return FALSE;
	}

	if (fread(header, 1, 8, fp) != 8) {
		fclose(fp);
		error("could not read file `%s'", filename);
		return FALSE;
	}

	if (png_sig_cmp(header, 0, 8)) {
		fclose(fp);
		error("file `%s' is not recognized as a "
		      "PNG file", filename);
		return FALSE;
	}

	/* initialize stuff */
	png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING,
	                                 NULL, NULL, NULL);

	if (!png_ptr) {
		fclose(fp);
		return FALSE;
	}

	info_ptr = png_create_info_struct(png_ptr);
	if (!info_ptr) {
		png_destroy_read_struct(&png_ptr, NULL, NULL);
		fclose(fp);
		return FALSE;
	}

	if (setjmp(png_jmpbuf(png_ptr))) {
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		fclose(fp);
		return FALSE;
	}

	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, 8);

	png_read_info(png_ptr, info_ptr);

	width = png_get_image_width(png_ptr, info_ptr);
	height = png_get_image_height(png_ptr, info_ptr);
	color_type = png_get_color_type(png_ptr, info_ptr);
	/* bit_depth = png_get_bit_depth(png_ptr, info_ptr); */

	if (color_type == PNG_COLOR_TYPE_RGB ||
	    color_type == PNG_COLOR_TYPE_RGB_ALPHA)
		png_set_rgb_to_gray(png_ptr, 1, -1.0, -1.0);

	(void) png_set_interlace_handling(png_ptr);
	png_read_update_info(png_ptr, info_ptr);

	/* read file */
	if (!image_allocate(img, (unsigned int) width,
	                   (unsigned int) height)) {
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		fclose(fp);
		return FALSE;
	}

	row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
	if (!row_pointers) {
		image_cleanup(img);
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		fclose(fp);
		return FALSE;
	}

	if (setjmp(png_jmpbuf(png_ptr))) {
		free(row_pointers);
		image_cleanup(img);
		png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
		fclose(fp);
		return FALSE;
	}

	for (y = 0; y < height; y++) {
		row_pointers[y] = (png_byte *)
		    &img->pixels[img->stride * y];
	}

	png_read_image(png_ptr, row_pointers);

	free(row_pointers);
	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
	fclose(fp);
	return TRUE;
}

#define IMAGE_TYPE_INVALID      -1
#define IMAGE_TYPE_PNG           0
#define IMAGE_TYPE_JPEG          1

#define MAX_HEADER_SIZE          8
static struct {
	int type;
	unsigned int length;
	unsigned char header[MAX_HEADER_SIZE];
} known_headers[] = {
	{ IMAGE_TYPE_PNG,  8, {137, 80, 78, 71, 13, 10, 26, 10} },
	{ IMAGE_TYPE_JPEG, 3, {0xFF, 0xD8, 0xFF} },
};
#define KNOWN_HEADERS_LEN (sizeof(known_headers) / sizeof(known_headers[0]))

static
int image_type(const char *filename)
{
	unsigned char header[MAX_HEADER_SIZE];
	unsigned int i;
	FILE *fp;

	fp = fopen(filename, "rb");
	if (!fp) {
		error("can't open `%s'", filename);
		return IMAGE_TYPE_INVALID;
	}

	if (fread(header, 1, MAX_HEADER_SIZE, fp) != MAX_HEADER_SIZE) {
		fclose(fp);
		return IMAGE_TYPE_INVALID;
	}
	fclose(fp);

	for (i = 0; i < KNOWN_HEADERS_LEN; i++) {
		if (memcmp(&known_headers[i].header, header,
		           known_headers[i].length) == 0) {
			return known_headers[i].type;
		}
	}
	return IMAGE_TYPE_INVALID;
}

int image_read(image *img, const char *filename)
{
	int type = image_type(filename);
	if (type == IMAGE_TYPE_PNG) {
		return read_png_file(img, filename);
	} else if (type == IMAGE_TYPE_JPEG) {
		return read_jpeg_file(img, filename);
	}
	return FALSE;
}

int image_write(const image *img, const char *filename)
{
	FILE *fp;
	unsigned int i, j, temp, size;
	unsigned int row, pos, width, rowsize, height;
	unsigned int values[3], offsets[] = {2, 18, 22};
	char rgbquad[4];
	char padding[4] =  { 0, 0, 0, 0 };
	char header[] = {
	  'B', 'M',    /* Signature */
	  0, 0, 0, 0,  /* Size of the BMP in bytes */
	  0, 0, 0, 0,  /* reserved */
	  54, 4, 0, 0, /* Offset of the beginning of bitmap data */
	  40, 0, 0, 0, /* Size of BITMAPINFOHEADER */
	  0, 0, 0, 0,  /* width */
	  0, 0, 0, 0,  /* height */
	  1, 0,        /* planes */
	  8, 0,        /* Bit count */
	  0, 0, 0, 0,  /* Compression */
	  0, 0, 0, 0,  /* SizeImage */
	  0, 0, 0, 0,  /* XPixelsPerMeter */
	  0, 0, 0, 0,  /* YPixelsPerMeter */
	  0, 0, 0, 0,  /* ClrUsed */
	  0, 0, 0, 0,  /* ClrImportant */
	};

	fp = fopen(filename, "wb");
	if (!fp) {
		error("can't open `%s'", filename);
		return FALSE;
	}

	width = img->width;
	height = img->height;

	rowsize = width;
	if ((width % 4) != 0)
		rowsize += 4 - (width % 4);

	size = 1078 + rowsize * height;
	values[0] = size;
	values[1] = width;
	values[2] = height;

	for (j = 0; j < 3; j++) {
		temp = values[j];
		for (i = 0; i < 4; i++) {
			header[offsets[j] + i] = (char) (temp & 255);
			temp >>= 8;
		}
	}

	if (fwrite(header, sizeof(header), 1, fp) != 1) {
		error("can't write to `%s'", filename);
		fclose(fp);
		return FALSE;
	}

	for (i = 0; i < 256; i++) {
		rgbquad[0] = (char) i;
		rgbquad[1] = (char) i;
		rgbquad[2] = (char) i;
		rgbquad[3] = 0;

		if (fwrite(rgbquad, sizeof(rgbquad), 1, fp) != 1) {
			error("can't write to `%s'", filename);
			fclose(fp);
			return FALSE;
		}
	}

	for (row = height; row > 0; row--) {
		pos = img->stride * (row - 1);
		if (fwrite(&img->pixels[pos], 1, width, fp) != width) {
			error("can't write to `%s'", filename);
			fclose(fp);
			return FALSE;
		}

		if (rowsize != width) {
			temp = rowsize - width;
			if (fwrite(padding, 1, temp, fp) != temp) {
				error("can't write to `%s'", filename);
				fclose(fp);
				return FALSE;
			}
		}
	}

	fclose(fp);
	return TRUE;
}

void image_draw_window(image *img, const window *w,
                       unsigned char color, unsigned int thickness)
{
	unsigned int row, col, pos;
	unsigned int left, top, right, bottom;

	if (!img->pixels)
		return;

	left = w->left;
	top = w->top;
	if (left >= img->width || top >= img->height)
		return;

	right = MIN(w->left + w->width, img->width);
	bottom = MIN(w->top + w->height, img->height);

	for (col = left; col < right; col++) {
		for (row = top; row < top + thickness; row++) {
			pos = row * img->stride + col;
			img->pixels[pos] = color;
		}

		for (row = bottom - thickness; row < bottom; row++) {
			pos = row * img->stride + col;
			img->pixels[pos] = color;
		}
	}

	for (row = top; row < bottom; row++) {
		for (col = left; col < left + thickness; col++) {
			pos = row * img->stride + col;
			img->pixels[pos] = color;
		}

		for (col = right - thickness; col < right; col++) {
			pos = row * img->stride + col;
			img->pixels[pos] = color;
		}
	}
}
