
#ifndef __WINDOW_H
#define __WINDOW_H

/* Data structures */
typedef
struct window_st {
	unsigned int left, top;
	unsigned int width, height;
} window;

/* Functions */
unsigned int window_area(const window *w);
void window_intersect(const window *w1, const window *w2, window *w);
void window_add(const window *w1, const window *w2, window *w);
int window_overlap(const window *w1, const window *w2,
                   double match_thresh, double overlap_thresh);
void window_compute_overlap(const window *w1, const window *w2,
                            double *match_thresh, double *overlap_thresh);
double window_similarity(const window *w1, const window *w2);

#endif /* __WINDOW_H */
