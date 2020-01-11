
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "features.h"
#include "image.h"
#include "window.h"
#include "utils.h"

void features_reset(features *f)
{
	f->sat = NULL;
	f->sat2 = NULL;
	f->hog = NULL;
}

void features_init(features *f)
{
	features_reset(f);
	f->capacity = 0;
	f->capacity_hog = 0;
}

void features_cleanup(features *f)
{
	f->capacity = 0;
	f->capacity_hog = 0;

	if (f->sat) {
		free(f->sat);
		f->sat = NULL;
	}

	if (f->sat2) {
		free(f->sat2);
		f->sat2 = NULL;
	}

	if (f->hog) {
		free(f->hog);
		f->hog = NULL;
	}
}

static
int features_allocate(features *f, unsigned int width, unsigned int height,
                      unsigned int nbins)
{
	unsigned int capacity;
	capacity = width * height;

	if (nbins == 0 && f->capacity < capacity) {
		void *ptr, *ptr2;
		ptr = xmalloc(capacity * sizeof(sval));
		ptr2 = xmalloc(capacity * sizeof(sval));
		if (!ptr || !ptr2) {
			if (ptr) free(ptr);
			if (ptr2) free(ptr2);
			return FALSE;
		}

		if (f->sat) free(f->sat);
		f->sat = (sval *) ptr;

		if (f->sat2) free(f->sat2);
		f->sat2 = (sval *) ptr2;

		f->capacity = capacity;
	}

	capacity = width * height * nbins;
	if (nbins > 0 && f->capacity_hog < capacity) {
		void *ptr;
		ptr = xmalloc(capacity * sizeof(sval));
		if (!ptr) return FALSE;

		if (f->hog) free(f->hog);
		f->hog = (sval *) ptr;

		f->capacity_hog = capacity;
	}

	f->width = width;
	f->height = height;
	f->stride = width;
	f->nbins = nbins;
	return TRUE;
}

int features_precompute(features *f, const image *img)
{
	unsigned int width, height, stride, istride;
	unsigned int row, col, pos, ipos;
	sval *sat, *sat2;

	if (!features_allocate(f, img->width + 1, img->height + 1, 0))
		return FALSE;

	width = f->width;
	height = f->height;
	stride = f->stride;
	istride = img->stride;

	sat = f->sat;
	sat2 = f->sat2;

	memset(sat, 0, width * sizeof(sval));
	memset(sat2, 0, width * sizeof(sval));

	for (row = 1; row < height; row++) {
		ipos = istride * (row - 1);
		pos = stride * row;
		sat[pos] = 0;
		sat2[pos] = 0;
		for (col = 1; col < width; col++) {
			sval val;

			val = (sval) img->pixels[ipos];
			sat[pos + 1] = sat[pos] + val;
			sat2[pos + 1] = sat2[pos] + (val * val);

			ipos++;
			pos++;
		}
	}

	for (col = 1; col < width; col++) {
		pos = col + stride;
		for (row = 2; row < height; row++) {
			sat[pos + stride] += sat[pos];
			sat2[pos + stride] += sat2[pos];
			pos += stride;
		}
	}
	return TRUE;
}

int features_precompute_hog(features *f, const image *img,
                            unsigned int nbins)
{
	unsigned int width, height, stride, istride;
	unsigned int row, col, bin, pos, ipos;
	sval *hog;

	if (!features_allocate(f, img->width + 1, img->height + 1, nbins))
		return FALSE;

	width = f->width;
	height = f->height;
	stride = f->stride;
	istride = img->stride;

	hog = f->hog;

	for (col = 0; col < width; col++) {
		pos = nbins * col;
		for (bin = 0; bin < nbins; bin++) {
			hog[pos + bin] = 0;
		}
	}

	for (row = 0; row < height; row++) {
		pos = nbins * stride * row;
		for (bin = 0; bin < nbins; bin++) {
			hog[pos + bin] = 0;
		}
	}

	for (col = 1; col < width; col++) {
		for (row = 1; row < height; row++) {
			double dx, dy, ang, mag;

			ipos = istride * (row - 1) + (col - 1);
			if (col > 1)
				dx = -img->pixels[ipos - 1];
			else
				dx = -img->pixels[ipos];

			if (col < width - 1)
				dx += img->pixels[ipos + 1];
			else
				dx += img->pixels[ipos];

			if (row > 1)
				dy = -img->pixels[ipos - istride];
			else
				dy = -img->pixels[ipos];

			if (row < height - 1)
				dy += img->pixels[ipos + istride];
			else
				dy += img->pixels[ipos];

			ang = atan2(dx, dy);
			if (ang < 0) ang += M_PI;
			mag = sqrt(dx * dx + dy * dy);

			pos = nbins * (stride * row + col);
			for (bin = 0; bin < nbins; bin++) {
				hog[pos + bin] = 0;
			}
			bin = (unsigned int) floor(ang * nbins / M_PI);
			hog[pos + bin] = (sval) mag;

			for (bin = 0; bin < nbins; bin++) {
				hog[pos] -= hog[pos - nbins * (stride + 1)];
				hog[pos] += hog[pos - nbins * stride];
				hog[pos] += hog[pos - nbins];
				pos++;
			}
		}
	}
	return TRUE;
}

static
sval rectangle_sum(const sval *sat, unsigned int stride,
                   unsigned int left, unsigned int top,
                   unsigned int width, unsigned int height)
{
	unsigned int pos = stride * top + left;
	sval sum;
	sum = sat[pos];
	sum += sat[pos + stride * height + width];
	sum -= sat[pos + width];
	sum -= sat[pos + stride * height];
	return sum;
}

static
void rectangle_sum_hog(const sval *hog, sval *out,
                       unsigned int stride, unsigned int nbins,
                       unsigned int left, unsigned int top,
                       unsigned int width, unsigned int height)
{
	unsigned int bin, pos, off1, off2;
	pos = nbins * (stride * top + left);
	off1 = nbins * (stride * height);
	off2 = nbins * width;

	for (bin = 0; bin < nbins; bin++) {
		out[bin] += hog[pos + bin];
		out[bin] += hog[pos + off1 + off2 + bin];
		out[bin] -= hog[pos + off2 + bin];
		out[bin] -= hog[pos + off1 + bin];
	}
}

double features_stddev(const features *f, const window *w)
{
	unsigned int n, stride;
	double avg, avg2, var;

	stride = f->stride;
	n = w->width * w->height;

	avg = (double) rectangle_sum(f->sat, stride, w->left, w->top,
	                             w->width, w->height);
	avg2 = (double) rectangle_sum(f->sat2, stride, w->left, w->top,
	                              w->width, w->height);
	avg /= n;
	avg2 /= n;
	var = avg2 - (avg * avg);

	if (var < 0) return 0;
	return sqrt(var);
}

void features_crop(const features *f, const window *w, double alpha,
                   sval *sat, unsigned int stride)
{
	unsigned int row, col, pos, tpos;
	unsigned int offset;
	double val;

	offset = w->top * f->stride + w->left;
	for (row = 0; row <= w->height; row++) {
		for (col = 0; col <= w->width; col++) {
			pos = row * f->stride + col;
			tpos = row * stride + col;
			val = f->sat[offset + pos];
			val -= f->sat[offset + pos - col];
			val -= f->sat[offset + col];
			val += f->sat[offset];
			sat[tpos] = (sval) (val * alpha);
		}
	}
}

void features_crop_hog(const features *f, const window *w,
                       sval *hog, unsigned int stride)
{
	unsigned int row, col, pos, tpos;
	unsigned int bin, nbins;
	unsigned int off;
	sval val;

	nbins = f->nbins;
	off = nbins * (w->top * f->stride + w->left);
	for (row = 0; row <= w->height; row++) {
		for (col = 0; col <= w->width; col++) {
			pos = nbins * (row * f->stride + col);
			tpos = nbins * (row * stride + col);
			for (bin = 0; bin < nbins; bin++) {
				val = f->hog[off + pos + bin];
				val -= f->hog[off + pos - col * nbins + bin];
				val -= f->hog[off + col * nbins + bin];
				val += f->hog[off + bin];
				hog[tpos + bin] = val;
			}
		}
	}
}

double features_evaluate(const sval *sat, unsigned int stride,
                         const feature_index *fi)
{
	sval val, sum1, sum2, sum3;
	unsigned int left, top, width, height;

	left = fi->w.left;
	top = fi->w.top;
	width = fi->w.width;
	height = fi->w.height;


	switch (fi->idx) {
	case 0:
		sum1 = rectangle_sum(sat, stride, left, top,
		                     width, height);
		sum2 = rectangle_sum(sat, stride, left + width, top,
		                     width, height);
		val = sum1 - sum2;
		break;
	case 1:
		sum1 = rectangle_sum(sat, stride, left, top,
		                     width, height);
		sum2 = rectangle_sum(sat, stride, left, top + height,
		                     width, height);
		val = sum1 - sum2;
		break;
	case 2:
		sum1 = rectangle_sum(sat, stride, left, top,
		                     3 * width, height);
		sum2 = rectangle_sum(sat, stride, left + width, top,
		                     width, height);
		val = sum1 - 3 * sum2;
		break;
	case 3:
		sum1 = rectangle_sum(sat, stride, left, top,
		                     4 * width, height);
		sum2 = rectangle_sum(sat, stride, left + width, top,
		                     2 * width, height);
		val = sum1 - 2 * sum2;
		break;
	case 4:
		sum1 = rectangle_sum(sat, stride, left, top,
		                     width, 3 * height);
		sum2 = rectangle_sum(sat, stride, left, top + height,
		                     width, height);
		val = sum1 - 3 * sum2;
		break;
	case 5:
		sum1 = rectangle_sum(sat, stride, left, top,
		                     width, 4 * height);
		sum2 = rectangle_sum(sat, stride, left, top + height,
		                     width, 2 * height);
		val = sum1 - 2 * sum2;
		break;
	case 6:
		sum1 = rectangle_sum(sat, stride, left, top,
		                     3 * width, 3 * height);
		sum2 = rectangle_sum(sat, stride, left + width, top + height,
		                     width, height);
		val = sum1 - 9 * sum2;
		break;
	case 7:
		sum1 = rectangle_sum(sat, stride, left, top,
		                     width, height);
		sum2 = rectangle_sum(sat, stride, left + width, top + height,
		                     width, height);
		sum3 = rectangle_sum(sat, stride, left, top,
		                     2 * width, 2 * height);
		val = 2 * (sum1 + sum2) - sum3;
		break;
	default:
		val = 0;
		break;
	}

	return ((double) val);
}

void features_evaluate_hog(const sval *hog, sval *out,
                           unsigned int stride, unsigned int nbins,
                           const feature_index *fi)
{
	unsigned int left, top, width, height;

	left = fi->w.left;
	top = fi->w.top;
	width = fi->w.width;
	height = fi->w.height;

	memset(out, 0, 4 * nbins * sizeof(sval));

	switch (fi->idx) {
	case 0:
		rectangle_sum_hog(hog, out, stride, nbins,
		                  left, top, width, height);
		rectangle_sum_hog(hog, &out[nbins], stride, nbins,
		                  left + width, top, width, height);
		rectangle_sum_hog(hog, &out[2 * nbins], stride, nbins,
		                  left, top + height, width, height);
		rectangle_sum_hog(hog, &out[3 * nbins], stride, nbins,
		                  left + width, top + height, width, height);
		break;
	}
}

int features_emit_rectangle(const feature_index *fi, features_rect_cb cb,
                            void *arg)
{
	unsigned int left, top, width, height;
	left = fi->w.left;
	top = fi->w.top;
	width = fi->w.width;
	height = fi->w.height;

	switch (fi->idx) {
	case 0:
		cb(arg,  1, left, top, width, height);
		cb(arg, -1, left + width, top, width, height);
		break;
	case 1:
		cb(arg,  1, left, top, width, height);
		cb(arg, -1, left, top + height, width, height);
		break;
	case 2:
		cb(arg,  1, left, top, 3 * width, height);
		cb(arg, -3, left + width, top, width, height);
		break;
	case 3:
		cb(arg,  1, left, top, 4 * width, height);
		cb(arg, -2, left + width, top, 2 * width, height);
		break;
	case 4:
		cb(arg,  1, left, top, width, 3 * height);
		cb(arg, -3, left, top + height, width, height);
		break;
	case 5:
		cb(arg,  1, left, top, width, 4 * height);
		cb(arg, -2, left, top + height, width, 2 * height);
		break;
	case 6:
		cb(arg,  1, left, top, 3 * width, 3 * height);
		cb(arg, -9, left + width, top + height, width, height);
		break;
	case 7:
		cb(arg,  2, left, top, width, height);
		cb(arg,  2, left + width, top + height, width, height);
		cb(arg, -1, left, top, 2 * width, 2 * height);
		break;
	default:
		return FALSE;
	}

	return TRUE;
}

struct emit_sat_st {
	void *arg;
	unsigned int stride;
	features_sat_cb cb;
};

static
void emit_sat_aux(void *arg, sval weight,
                  unsigned int left, unsigned int top,
                  unsigned int width, unsigned int height)
{
	unsigned int pos, stride;
	struct emit_sat_st *info = (struct emit_sat_st *) arg;
	stride = info->stride;

	pos = top * stride + left;
	info->cb(info->arg, weight, pos);
	info->cb(info->arg, weight, pos + height * stride + width);
	info->cb(info->arg, -weight, pos + height * stride);
	info->cb(info->arg, -weight, pos + width);
}

int features_emit_sat(const feature_index *fi, unsigned int stride,
                      features_sat_cb cb, void *arg)
{
	struct emit_sat_st info;
	info.arg = arg;
	info.cb = cb;
	info.stride = stride;

	return features_emit_rectangle(fi, &emit_sat_aux, &info);
}

static
void optimize_aux(void *arg, sval weight, unsigned int pos)
{
	feature_index_opt *fo = (feature_index_opt *) arg;
	fo->point[fo->num_opt_points] = pos;
	fo->weight[fo->num_opt_points] = weight;
	fo->num_opt_points++;
}

int features_optimize(const feature_index *fi, feature_index_opt *fo,
                      unsigned int stride)
{
	fo->num_opt_points = 0;
	return features_emit_sat(fi, stride, &optimize_aux, fo);
}

double features_evaluate_fast(const sval *sat, const feature_index_opt *fo)
{
	unsigned int i;
	sval val;

	val = 0;
	for (i = 0; i < fo->num_opt_points; i++) {
		val += fo->weight[i] * sat[fo->point[i]];
	}
	return ((double) val);
}

void feature_enumerator_start(feature_enumerator *fe, unsigned int width,
                              unsigned int height, int use_hog)
{

	fe->width = width;
	fe->height = height;
	fe->use_hog = use_hog;

	fe->fi.idx = 0;
	fe->fi.w.left = 0;
	fe->fi.w.top = 0;
	fe->fi.w.width = 1;
	fe->fi.w.height = 1;
	fe->mx = 2;
	fe->my = (use_hog) ? 2 : 1;
	fe->max_w.left = fe->width - fe->mx * fe->fi.w.width;
	fe->max_w.top = fe->height - fe->my * fe->fi.w.height;
	fe->max_w.width = fe->width / fe->mx;
	fe->max_w.height = fe->height / fe->my;

	fe->count = 0;
}

int feature_enumerator_next(feature_enumerator *fe)
{
	fe->count++;

	if (++(fe->fi.w.left) <= fe->max_w.left)
		return TRUE;
	fe->fi.w.left = 0;

	if (++(fe->fi.w.top) <= fe->max_w.top)
		return TRUE;
	fe->fi.w.top = 0;


	if (++(fe->fi.w.width) <= fe->max_w.width) {
		fe->max_w.left = fe->width - fe->mx * fe->fi.w.width;
		return TRUE;
	}
	fe->fi.w.width = 1;
	fe->max_w.left = fe->width - fe->mx * fe->fi.w.width;

	if (++(fe->fi.w.height) <= fe->max_w.height) {
		fe->max_w.top = fe->height - fe->my * fe->fi.w.height;
		return TRUE;
	}
	fe->fi.w.height = 1;
	fe->max_w.top = fe->height - fe->my * fe->fi.w.height;

	switch (++(fe->fi.idx)) {
	case 1:
		if (fe->use_hog) {
			fe->count--;
			return FALSE;
		}
		fe->mx = 1;
		fe->my = 2;
		break;
	case 2:
		fe->mx = 3;
		fe->my = 1;
		break;
	case 3:
		fe->mx = 4;
		fe->my = 1;
		break;
	case 4:
		fe->mx = 1;
		fe->my = 3;
		break;
	case 5:
		fe->mx = 1;
		fe->my = 4;
		break;
	case 6:
		fe->mx = 3;
		fe->my = 3;
		break;
	case 7:
		fe->mx = 2;
		fe->my = 2;
		break;
	default:
		fe->count--;
		return FALSE;
	}

	fe->max_w.left = fe->width - fe->mx * fe->fi.w.width;
	fe->max_w.top = fe->height - fe->my * fe->fi.w.height;
	fe->max_w.width = fe->width / fe->mx;
	fe->max_w.height = fe->height / fe->my;
	return TRUE;
}

int feature_enumerator_advance(feature_enumerator *fe, unsigned int count)
{
	unsigned int i;
	for (i = 0; i < count; i++)
		if (!feature_enumerator_next(fe))
			return FALSE;
	return TRUE;
}

unsigned int feature_enumerator_count(feature_enumerator *fe)
{
	unsigned int count = 0;
	while (feature_enumerator_next(fe))
		count++;
	return count;
}
