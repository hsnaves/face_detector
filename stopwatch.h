
#ifndef __STOPWATCH_H
#define __STOPWATCH_H

#include <time.h>

/* Data structures */
typedef
struct stopwatch_st {
	clock_t clk_start, clk_end;
	struct timespec time_start, time_end;
} stopwatch;

/* Functions */
void stopwatch_start(stopwatch *sw);
void stopwatch_stop(stopwatch *sw, double *elapsed, double *cpu_time);

#endif /* __STOPWATCH_H */
