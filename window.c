
#include "window.h"
#include "utils.h"

unsigned int window_area(const window *w)
{
	return w->width * w->height;
}

void window_intersect(const window *w1, const window *w2, window *w)
{
	w->top = MAX(w1->top, w2->top);
	w->left = MAX(w1->left, w2->left);

	w->width = MIN(w1->left + w1->width, w2->left + w2->width);
	if (w->width >= w->left)
		w->width -= w->left;
	else
		w->width = 0;

	w->height = MIN(w1->top + w1->height, w2->top + w2->height);
	if (w->height >= w->top)
		w->height -= w->top;
	else
		w->height = 0;
}

void window_add(const window *w1, const window *w2, window *w)
{
	w->top = MIN(w1->top, w2->top);
	w->left = MIN(w1->left, w2->left);

	w->width = MAX(w1->left + w1->width, w2->left + w2->width);
	w->width -= w->left;

	w->height = MAX(w1->top + w1->height, w2->top + w2->height);
	w->height -= w->top;
}

int window_overlap(const window *w1, const window *w2,
                   double match_thresh, double overlap_thresh)
{
	unsigned int area1, area2, areaI, areaU;
	window w;

	window_intersect(w1, w2, &w);
	areaI = window_area(&w);
	area1 = window_area(w1);
	if (areaI >= match_thresh * area1)
		return TRUE;

	area2 = window_area(w2);
	if (areaI >= match_thresh * area2)
		return TRUE;
	areaU = area1 + area2 - areaI;

	return (areaI >= overlap_thresh * areaU);
}

void window_compute_overlap(const window *w1, const window *w2,
                            double *match_thresh, double *overlap_thresh)
{
	unsigned int area1, area2;
	double areaI, areaU;
	window w;

	window_intersect(w1, w2, &w);
	areaI = (double) window_area(&w);
	area1 = window_area(w1);
	area2 = window_area(w2);
	areaU = area1 + area2 - areaI;
	*match_thresh = MAX(*match_thresh, areaI / area1);
	*match_thresh = MAX(*match_thresh, areaI / area2);
	*overlap_thresh = MAX(*overlap_thresh, areaI / areaU);
}

double window_similarity(const window *w1, const window *w2)
{
	double area1, area2;
	double areaI, areaU;
	window w;

	window_intersect(w1, w2, &w);
	areaI = (double) window_area(&w);
	area1 = (double) window_area(w1);
	area2 = (double) window_area(w2);
	areaU = area1 + area2 - areaI;

	return areaI / areaU;
}
