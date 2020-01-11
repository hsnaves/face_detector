
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "cpa.h"
#include "utils.h"

#define MAX_ITERATIONS 50000000

void cpa_reset(cpa *c)
{
	c->unused_count = NULL;
	c->pl_intercept = NULL;
	c->pl_coef = NULL;
	c->kernel = NULL;
	c->lambda = NULL;
	c->df = NULL;
	c->w = NULL;
	c->best_w = NULL;
	c->dividers = NULL;
}

int cpa_init(cpa *c, unsigned int max_dim, unsigned int max_samples,
             unsigned int max_planes)
{
	size_t size;
	cpa_reset(c);

	c->max_dim = max_dim;
	c->max_samples = max_samples;
	c->max_planes = max_planes;

	size = max_planes * sizeof(unsigned int);
	c->unused_count = (unsigned int *) xmalloc(size);
	if (!c->unused_count) goto error_init;

	size = max_planes * sizeof(double);
	c->pl_intercept = (double *) xmalloc(size);
	if (!c->pl_intercept) goto error_init;

	size = max_planes * (max_dim + 1) * sizeof(double);
	c->pl_coef = (double *) xmalloc(size);
	if (!c->pl_coef) goto error_init;

	size = max_planes * max_planes * sizeof(double);
	c->kernel = (double *) xmalloc(size);
	if (!c->kernel) goto error_init;

	size = max_planes * sizeof(double);
	c->lambda = (double *) xmalloc(size);
	if (!c->lambda) goto error_init;

	size = max_planes * sizeof(double);
	c->df = (double *) xmalloc(size);
	if (!c->df) goto error_init;

	size = (max_dim + 1) * sizeof(double);
	c->w = (double *) xmalloc(size);
	if (!c->w) goto error_init;

	size = (max_dim + 1) * sizeof(double);
	c->best_w = (double *) xmalloc(size);
	if (!c->best_w) goto error_init;

	size = 2 * max_samples * sizeof(double);
	c->dividers = (double *) xmalloc(size);
	if (!c->dividers) goto error_init;

	return TRUE;

error_init:
	cpa_cleanup(c);
	return FALSE;
}

void cpa_cleanup(cpa *c)
{
	if (c->unused_count) {
		free(c->unused_count);
		c->unused_count = NULL;
	}

	if (c->pl_intercept) {
		free(c->pl_intercept);
		c->pl_intercept = NULL;
	}

	if (c->pl_coef) {
		free(c->pl_coef);
		c->pl_coef = NULL;
	}

	if (c->kernel) {
		free(c->kernel);
		c->kernel = NULL;
	}

	if (c->lambda) {
		free(c->lambda);
		c->lambda = NULL;
	}

	if (c->df) {
		free(c->df);
		c->df = NULL;
	}

	if (c->w) {
		free(c->w);
		c->w = NULL;
	}

	if (c->best_w) {
		free(c->best_w);
		c->best_w = NULL;
	}

	if (c->dividers) {
		free(c->dividers);
		c->dividers = NULL;
	}
}

int cpa_set_params(cpa *c, double Cp, double Cn, double eps, double eps_qp,
                   double mu, unsigned int max_unused)
{
	c->Cp = Cp;
	c->Cn = Cn;
	c->eps = eps;
	c->eps_qp = eps_qp;
	c->mu = mu;
	c->max_unused = max_unused;
	return TRUE;
}

static
void cpa_remove_plane(cpa *c, unsigned int i)
{
	unsigned int st1, st2, j, k;

	j = --(c->planes);
	if (i == j) return;

	st1 = c->max_dim  + 1;
	st2 = c->max_planes;
	c->pl_intercept[i] = c->pl_intercept[j];
	c->unused_count[i] = c->unused_count[j];
	c->lambda[i] = c->lambda[j];
	memcpy(&c->pl_coef[i * st1], &c->pl_coef[j * st1],
	       (c->dim + 1) * sizeof(double));

	memcpy(&c->kernel[i * st2], &c->kernel[j * st2],
	       (c->planes + 1) * sizeof(double));
	for (k = 0; k < c->planes; k++) {
		c->kernel[k * st2 + i] = c->kernel[k * st2 + j];
	}
}

static
double dot_product(const double *v1, const double *v2, unsigned int dim)
{
	unsigned int i;
	double sum = 0;

	for (i = 0; i < dim; i++)
		sum += v1[i] * v2[i];
	return sum;
}

static
void add_vectors(double *v1, const double *v2, double alpha, unsigned int dim)
{
	unsigned int i;
	for (i = 0; i < dim; i++) {
		v1[i] += v2[i] * alpha;
	}
}

static
void combine_vectors(double *v1, const double *v2, double alpha,
                     unsigned int dim)
{
	unsigned int i;
	for (i = 0; i < dim; i++) {
		v1[i] = v2[i] + (v1[i] - v2[i]) * alpha;
	}
}

static
void cpa_compute_kernel(cpa *c, int only_last)
{
	unsigned int i, j, st1, st2;
	double v;

	st1 = c->max_dim + 1;
	st2 = c->max_planes;
	i = (only_last) ? c->planes - 1 : 0;
	for (; i < c->planes; i++) {
		for (j = 0; j <= i; j++) {
			v = dot_product(&c->pl_coef[st1 * i],
			                &c->pl_coef[st1 * j], c->dim + 1);
			c->kernel[i * st2 + j] = v;
			c->kernel[j * st2 + i] = v;
		}
	}
}

static
void cpa_compute_df(cpa *c)
{
	unsigned int i, st1;

	st1 = c->max_planes;
	for (i = 0; i < c->planes; i++) {
		c->df[i] = dot_product(&c->kernel[i * st1], c->lambda,
		                       c->planes);
		c->df[i] -= c->pl_intercept[i];
	}
}

static
unsigned int cpa_solve_dual(cpa *c)
{
	double big, little, gap, tau;
	double tmp_b, tmp_l, quad_coef, delta;
	unsigned int i, b, l, iter, st1;

	tau = 1e-10;
	st1 = c->max_planes;
	cpa_compute_df(c);

	gap = c->eps_qp;
	big = little = 0;
	l = b = 0;

	for (iter = 0; iter < MAX_ITERATIONS; iter++) {
		big = -INFINITY;
		little = INFINITY;
		l = b = 0;
		for (i = 0; i < c->planes; i++) {
			if (c->df[i] > big && c->lambda[i] > 0) {
				big = c->df[i];
				b = i;
			}
			if (c->df[i] < little) {
				little = c->df[i];
				l = i;
			}
		}
		gap = dot_product(c->lambda, c->df, c->planes) - little;
		if (gap < c->eps_qp) break;

		tmp_b = c->lambda[b];
		tmp_l = c->lambda[l];
		quad_coef = c->kernel[b * st1 + b] + c->kernel[l * st1 + l];
		quad_coef -= 2 * c->kernel[b * st1 + l];
		quad_coef = MAX(quad_coef, tau);
		delta = (big - little) / quad_coef;
		c->lambda[b] -= delta;
		c->lambda[l] += delta;

		if (c->lambda[b] < 0) {
			c->lambda[b] = 0;
			c->lambda[l] = tmp_b + tmp_l;
		}

		if ((iter % 300) == 299) {
			cpa_compute_df(c);
		} else {
			tmp_b = c->lambda[b] - tmp_b;
			tmp_l = c->lambda[l] - tmp_l;

			for (i = 0; i < c->planes; i++) {
				c->df[i] += c->kernel[i * st1 + b] * tmp_b;
				c->df[i] += c->kernel[i * st1 + l] * tmp_l;
			}
		}
	}

	if (gap >= c->eps_qp) {
		printf("gap = %g, little = %g, l = %u, big = %g, b = %u\n",
		       gap, little, l, big, b);
	}

	return iter + 1;
}

static
double cpa_risk_approximation(const cpa *c, const double *w)
{
	unsigned int i, st;
	double dot, max_pl;

	st = c->max_dim + 1;
	max_pl = -INFINITY;
	for (i = 0; i < c->planes; i++) {
		dot = dot_product(w, &c->pl_coef[i * st], c->dim + 1);
		max_pl = MAX(max_pl, dot + c->pl_intercept[i]);
	}
	return max_pl;
}

static
double cpa_compute_new_plane(cpa *c, const double **feat_vals, int *y,
                             unsigned int num_samples)
{
	unsigned int i;
	double *pl_coef, mult, dot, risk;

	risk = 0;
	pl_coef = &c->pl_coef[c->planes * (c->max_dim + 1)];
	memset(pl_coef, 0, (c->dim + 1) * sizeof(double));
	for (i = 0; i < num_samples; i++) {
		mult = (y[i] > 0) ? c->Cn : c->Cp;
		dot = dot_product(&c->w[1], feat_vals[i], c->dim);
		dot += c->w[0];
		dot *= y[i];
		if (dot <= 1) {
			risk += mult * (1 - dot);
			add_vectors(&pl_coef[1], feat_vals[i],
			            -y[i] * mult, c->dim);
			pl_coef[0] -= y[i] * mult;
		}
	}
	return risk;
}

static
void cpa_compute_primal(const cpa *c, double *w)
{
	unsigned int i, st;

	st = c->max_dim + 1;
	memset(w, 0, (c->dim + 1) * sizeof(double));

	for (i = 0; i < c->planes; i++) {
		add_vectors(w, &c->pl_coef[i * st],
		            -c->lambda[i], c->dim + 1);
	}
}

static
void cpa_remove_unused_planes(cpa *c)
{
	unsigned int i;

	i = 0;
	while (i < c->planes) {
		if (c->lambda[i] == 0) {
			c->unused_count[i]++;
		} else {
			c->unused_count[i] = 0;
		}

		if (c->unused_count[i] >= c->max_unused) {
			printf("Removing plane %u\n", i);
			cpa_remove_plane(c, i);
		} else {
			i++;
		}
	}
}

static
int cmp_pair_dbl(const void *p1, const void *p2)
{
	double d1 = ((const double *) p1)[0];
	double d2 = ((const double *) p2)[0];

	if (d1 < d2) return -1;
	if (d1 > d2) return 1;
	return 0;
}

static
double cpa_line_search(cpa *c, double *best_w, double *w,
                       const double **feat_vals, int *y,
                       unsigned int num_samples)
{
	unsigned int i, count;
	double x, nx, val, nval, mult, dot1, dot2;
	double A0, B0, B, C, K;

	dot1 = dot_product(w, best_w, c->dim + 1);
	dot2 = dot_product(best_w, best_w, c->dim + 1);
	B0 = dot1 - dot2;
	A0 = dot_product(w, w, c->dim + 1);
	A0 = A0 + dot2 - 2 * dot1;

	val = B0;
	count = 0;
	for (i = 0; i < num_samples; i++) {
		dot1 = dot_product(&w[1], feat_vals[i], c->dim);
		dot1 += w[0];

		dot2 = dot_product(&best_w[1], feat_vals[i], c->dim);
		dot2 += best_w[0];

		mult = (y[i] > 0) ? c->Cn : c->Cp;
		B = mult * y[i] * (dot2 - dot1);
		C = mult * (1 - y[i] * dot2);
		K = -C / B;

		if ((B < 0 && K > 0) || (B > 0 && K <= 0)) {
			val += B;
		}

		if (K > 0) {
			c->dividers[2 * count] = K;
			c->dividers[2 * count + 1] = B;
			count++;
		}
	}

	if (val > 0) return 0;
	qsort(c->dividers, count, 2 * sizeof(double), &cmp_pair_dbl);

	x = 0;
	for (i = 0; i < count; i++) {
		nx = c->dividers[2 * i];
		nval = val + A0 * (nx - x);
		if (nval >= 0) {
			return x - (val / A0);
		}
		nval += fabs(c->dividers[2 * i + 1]);
		if (nval >= 0) {
			return nx;
		}
		nx = x;
		val = nval;
	}
	return x;
}

void cpa_solve(cpa *c, const double **feat_vals, int *y,
               unsigned int dim, unsigned int num_samples)
{
	unsigned int st;
	double risk, approx, pl_inter;
	double old_val;

	st = c->max_dim + 1;

	c->dim = dim;
	c->planes = 0;
	memset(c->w, 0, (c->dim + 1) * sizeof(double));
	memset(c->best_w, 0, (c->dim + 1) * sizeof(double));

	memset(c->pl_coef, 0, (c->dim + 1) * sizeof(double));
	c->unused_count[0] = 0;
	c->pl_intercept[0] = 0;
	c->lambda[0] = 1;
	c->planes = 1;
	cpa_compute_kernel(c, TRUE);
	old_val = 0;

	while (c->planes < c->max_planes) {
		double dot, alpha, sq;

		sq = 0.5 * dot_product(c->w, c->w, c->dim + 1);
		approx = cpa_risk_approximation(c, c->w);
		risk = cpa_compute_new_plane(c, feat_vals, y, num_samples);

		printf("Plane: %u, risk difference = %g, risk = %g, "
		       "approx = %g, approx val=%g\n", c->planes,
		       risk - approx, risk, approx, approx + sq);

		if (risk - approx < c->eps) break;
		if (approx + sq > old_val) {
			error("value increased");
			return;
		}

		dot = dot_product(c->w, &c->pl_coef[c->planes * st],
		                  c->dim + 1);
		pl_inter = risk - dot;
		approx = MAX(approx, dot + pl_inter);
		printf("Inter = %g, new approx val = %g\n",
		       pl_inter, approx + sq);

		old_val = approx + sq;
		c->pl_intercept[c->planes] = pl_inter;
		c->lambda[c->planes] = 0;
		c->unused_count[c->planes] = 0;
		c->planes++;

		cpa_compute_kernel(c, TRUE);
		cpa_solve_dual(c);

		cpa_compute_primal(c, c->w);
		alpha = cpa_line_search(c, c->best_w, c->w,
		                        feat_vals, y, num_samples);
		printf("alpha = %g\n", alpha);
		combine_vectors(c->best_w, c->w, 1 - alpha, c->dim + 1);
		combine_vectors(c->w, c->best_w, c->mu, c->dim + 1);
		cpa_remove_unused_planes(c);
	}
}
