
#ifndef __CPA_H
#define __CPA_H

/* Data structures and types */
typedef
struct cpa_st {
	unsigned int dim, planes;
	unsigned int max_dim, max_samples, max_planes;

	double Cp, Cn;
	unsigned int max_unused;
	double eps, eps_qp, mu;

	unsigned int *unused_count;
	double *pl_coef, *pl_intercept;
	double *lambda, *df, *w, *best_w;
	double *dividers;
	double *kernel;

} cpa;

/* Functions */
void cpa_reset(cpa *c);
int cpa_init(cpa *c, unsigned int max_dim, unsigned int max_samples,
             unsigned int max_planes);
void cpa_cleanup(cpa *c);

int cpa_set_params(cpa *c, double Cp, double Cn, double eps, double eps_qp,
                   double mu, unsigned int max_unused);
void cpa_solve(cpa *c, const double **feat_vals, int *y,
               unsigned int dim, unsigned int num_samples);

#endif /* __CPA_H */


