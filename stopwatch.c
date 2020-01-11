
#include <time.h>
#include "stopwatch.h"

void stopwatch_start(stopwatch *sw)
{
	clock_gettime(CLOCK_MONOTONIC, &sw->time_start);
	sw->clk_start = clock();
}

void stopwatch_stop(stopwatch *sw, double *elapsed, double *cpu_time)
{
	sw->clk_end = clock();
	clock_gettime(CLOCK_MONOTONIC, &sw->time_end);

	*elapsed = (double) (sw->time_end.tv_sec - sw->time_start.tv_sec);
	*elapsed += ((double) (sw->time_end.tv_nsec - sw->time_start.tv_nsec))
	             / 1.0e+9;
	*cpu_time = ((double) (sw->clk_end - sw->clk_start)) / CLOCKS_PER_SEC;
}

