
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "trainer.h"
#include "detector.h"
#include "cascade.h"
#include "samples.h"
#include "features.h"
#include "image.h"
#include "window.h"
#include "random.h"
#include "utils.h"

#define ARG_FLAG_REQ       1
#define ARG_FLAG_POS       2
#define ARG_FLAG_BIGGER1   4
#define ARG_FLAG_PROB      8
#define ARG_FLAG_NEEDFILE 16
#define ARG_FLAG_DEF      32

enum argument_type {
	ARG_CMD, ARG_FILE, ARG_DIR, ARG_DBL, ARG_INT, ARG_UINT, ARG_BOOL
};

union argument_value {
	unsigned int uint_val;
	int int_val;
	double dbl_val;
	char *str_val;
};

struct argument_definition {
	const char *arg_name;
	enum argument_type arg_type;
	int flags;
	const char *default_value;
	const char *arg_help;
	const char *arg_extra_help;
	union argument_value value;
	int arg_set;
};

static
struct argument_definition arguments[] = {
	{ "--help", ARG_BOOL, 0, NULL,
	  "Print this help" },
	{ "train", ARG_CMD, ARG_FLAG_NEEDFILE, NULL,
	  "Train command", "file..." },
	{ "--datadir", ARG_DIR, ARG_FLAG_REQ, "data",
	  "Specify directory with training images" },
	{ "--width", ARG_UINT, ARG_FLAG_REQ, "24",
	  "Specify width of image for the classifier" },
	{ "--height", ARG_UINT, ARG_FLAG_REQ, "24",
	  "Specify height of image for the classifier" },
	{ "--match_thresh", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_PROB, "0.75",
	  "Coefficient used to determine if two boxes match" },
	{ "--overlap_thresh", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_PROB, "0.33",
	  "Coefficient used to determine if two boxes overlap" },
	{ "--learn_overlap", ARG_BOOL, 0, NULL,
	  "To lean the match_thresh and overlap_thresh from the data" },
	{ "--min_similarity", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_PROB, "0.9",
	  "Minimum similarity when finding positive windows" },
	{ "--Cp", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_POS, "1",
	  "Penalty constant for false positives" },
	{ "--Cn", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_POS, "1",
	  "Penalty constant for false negatives" },
	{ "--max_stages", ARG_UINT, ARG_FLAG_REQ, "25",
	  "Maximum number of stages for the cascade" },
	{ "--max_classifiers", ARG_UINT, ARG_FLAG_REQ, "200",
	  "Maximum number of classifiers per stage" },
	{ "--num_parallels", ARG_UINT, ARG_FLAG_REQ, "1",
	  "Number of parallel classifiers" },
	{ "--cycle_parallels", ARG_BOOL, 0, NULL,
	  "Force traininig to cycle through the parallels" },
	{ "--buckets", ARG_UINT, ARG_FLAG_REQ, "1001",
	  "Number of buckets for boosting" },
	{ "--bucket_max", ARG_DBL, ARG_FLAG_REQ,
	  "300.0", "Maximum value of buckets" },
	{ "--bucket_min", ARG_DBL, ARG_FLAG_REQ,
	  "-300.0", "Minimum value of buckets" },
	{ "--max_false_positive", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_PROB,
	  "0.25", "Maximum false positive rate per stage" },
	{ "--max_false_negative", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_PROB,
	  "0.005", "Maximum false negative rate per stage" },
	{ "--feature_prob", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_PROB,
	  "1", "Choose a random subset of features with this probability" },
	{ "--min_jumbled", ARG_UINT, ARG_FLAG_REQ,
	  "100", "Use all jumbled images if it is less than this parameter" },
	{ "--min_negative_samples", ARG_UINT, ARG_FLAG_REQ, "10",
	  "Minimum number of negative samples" },
	{ "--positive_samples", ARG_UINT, ARG_FLAG_REQ, "0",
	  "Number of positive samples" },
	{ "--negative_samples", ARG_UINT, ARG_FLAG_REQ, "0",
	  "Number of negative samples" },
	{ "--scale", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_BIGGER1, "1.1",
	  "How much to scale images in detection" },
	{ "--min_stddev", ARG_DBL, ARG_FLAG_REQ , "10.0",
	  "Minimum standard deviation for a detected object" },
	{ "--step", ARG_UINT, ARG_FLAG_REQ, "1",
	  "How many pixels should the detection window move per step" },
	{ "--multi_exit", ARG_BOOL, 0, NULL,
	  "Set the cascade to multi-exit" },
	{ "--cascade", ARG_FILE, ARG_FLAG_REQ, "cascade.txt",
	  "Name of the output cascade file" },
	{ "--num_threads", ARG_UINT, ARG_FLAG_REQ, "1",
	  "Number of threads used to train" },
	{ "--max_planes", ARG_UINT, ARG_FLAG_REQ, "1000",
	  "Maximum number of planes for the cutting plane algorithm (CPA)" },
	{ "--eps", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_POS, "1e-6",
	  "To determine convergence of CPA based on risk difference" },
	{ "--eps_qp", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_POS, "1e-6",
	  "To determine convergence of the dual quadratic programming" },
	{ "--mu", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_PROB, "0.1",
	  "Coefficient for the position of the next plane" },
	{ "--max_unused", ARG_UINT, ARG_FLAG_REQ, "200",
	  "Remove plane after not being used for many iterations" },
	{ "--help", ARG_BOOL, 0, NULL,
	  "Print this help" },
	{ "resize", ARG_CMD, ARG_FLAG_NEEDFILE, NULL,
	  "Resize an image file", "file..." },
	{ "--output", ARG_FILE, ARG_FLAG_REQ, NULL,
	  "Name of the output BMP image file" },
	{ "--width", ARG_UINT, ARG_FLAG_REQ | ARG_FLAG_DEF, "resize",
	  "New width of the image" },
	{ "--height", ARG_UINT, ARG_FLAG_REQ | ARG_FLAG_DEF, "resize",
	  "New height of the image" },
	{ "--help", ARG_BOOL, 0, NULL,
	  "Print this help" },
	{ "detect", ARG_CMD, ARG_FLAG_NEEDFILE, NULL,
	  "Detect objects in a picture", "file..." },
	{ "--cascade", ARG_FILE, ARG_FLAG_REQ, "cascade.txt",
	  "Name of the input cascade file" },
	{ "--scale", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_DEF
	             | ARG_FLAG_BIGGER1, "cascade",
	  "How much to scale images in detection" },
	{ "--min_stddev", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_DEF, "cascade",
	  "Minimum standard deviation for a detected object" },
	{ "--step", ARG_UINT, ARG_FLAG_REQ | ARG_FLAG_DEF, "cascade",
	  "How many pixels should the detection window move per step" },
	{ "--match_thresh", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_DEF
	                    | ARG_FLAG_PROB, "cascade",
	  "Coefficient used to determine if two boxes match" },
	{ "--overlap_thresh", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_DEF
	                      | ARG_FLAG_PROB, "cascade",
	  "Coefficient used to determine if two boxes overlap" },
	{ "--min_width", ARG_UINT, ARG_FLAG_REQ, "0",
	  "Minimum detection window width" },
	{ "--max_width", ARG_UINT, ARG_FLAG_REQ, "0",
	  "Maximum detection window width" },
	{ "--min_height", ARG_UINT, ARG_FLAG_REQ, "0",
	  "Minimum detection window height" },
	{ "--max_height", ARG_UINT, ARG_FLAG_REQ, "0",
	  "Maximum detection window height" },
	{ "--output", ARG_FILE, ARG_FLAG_REQ, NULL,
	  "Name of the output image file" },
	{ "--help", ARG_BOOL, 0, NULL,
	  "Print this help" },
	{ "evaluate", ARG_CMD, ARG_FLAG_NEEDFILE, NULL,
	  "Evaluate cascade file", "file..." },
	{ "--cascade", ARG_FILE, ARG_FLAG_REQ, "cascade.txt",
	  "Name of the input cascade file" },
	{ "--datadir", ARG_DIR, ARG_FLAG_REQ, "data",
	  "Specify directory with testing images" },
	{ "--scale", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_DEF
	             | ARG_FLAG_BIGGER1, "cascade",
	  "How much to scale images in detection" },
	{ "--min_stddev", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_DEF, "cascade",
	  "Minimum standard deviation for a detected object" },
	{ "--step", ARG_UINT, ARG_FLAG_REQ | ARG_FLAG_DEF, "cascade",
	  "How many pixels should the detection window move per step" },
	{ "--match_thresh", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_DEF
	                    | ARG_FLAG_PROB, "cascade",
	  "Coefficient used to determine if two boxes match" },
	{ "--overlap_thresh", ARG_DBL, ARG_FLAG_REQ | ARG_FLAG_DEF
	                      | ARG_FLAG_PROB, "cascade",
	  "Coefficient used to determine if two boxes overlap" },
	{ "--min_width", ARG_UINT, ARG_FLAG_REQ, "0",
	  "Minimum detection window width" },
	{ "--max_width", ARG_UINT, ARG_FLAG_REQ, "0",
	  "Maximum detection window width" },
	{ "--min_height", ARG_UINT, ARG_FLAG_REQ, "0",
	  "Minimum detection window height" },
	{ "--max_height", ARG_UINT, ARG_FLAG_REQ, "0",
	  "Maximum detection window height" },
	{ "--num_cascades", ARG_UINT, ARG_FLAG_REQ, "1",
	  "Number of cascades used to evaluate" },
	{ "--num_threads", ARG_UINT, ARG_FLAG_REQ, "1",
	  "Number of threads used to evaluate" },
	{ "--help", ARG_BOOL, 0, NULL,
	  "Print this help" },
};
#define ARGUMENTS_SIZE \
  (sizeof(arguments) / sizeof(struct argument_definition))

static
const char *get_specifier(enum argument_type arg_type)
{
	const char *specifier;
	switch (arg_type) {
	case ARG_FILE:
		specifier = "<file>";
		break;
	case ARG_DIR:
		specifier = "<dir>";
		break;
	case ARG_UINT:
		specifier = "<uint>";
		break;
	case ARG_DBL:
		specifier = "<dbl>";
		break;
	case ARG_INT:
		specifier = "<int>";
		break;
	default:
		specifier = "";
		break;
	}
	return specifier;
}

static
int parse_argument(enum argument_type arg_type, const char *str,
                   union argument_value *val)
{
	switch (arg_type) {
	case ARG_FILE:
	case ARG_DIR:
		val->str_val = (char *) str;
		break;
	case ARG_UINT:
		val->uint_val = (unsigned int) atoi(str);
		break;
	case ARG_DBL:
		val->dbl_val = atof(str);
		break;
	case ARG_INT:
		val->int_val = atoi(str);
		break;
	case ARG_BOOL:
		val->int_val = TRUE;
		break;
	default:
		return FALSE;
	}
	return TRUE;
}

static
void show_help(unsigned int cmd, char *prog_name)
{
	unsigned int j, len, max_len = 0;
	const char *specifier;

	if (cmd == 0) {
		printf("Usage: %s [options] cmd...\n", prog_name);

		max_len = 0;
		printf("Possible commands:\n");
		for (j = 0; j < ARGUMENTS_SIZE; j++) {
			if (arguments[j].arg_type != ARG_CMD) continue;
			len = (unsigned int) strlen(arguments[j].arg_name);
			if (len > max_len) max_len = len;
		}
		for (j = 0; j < ARGUMENTS_SIZE; j++) {
			if (arguments[j].arg_type != ARG_CMD) continue;
			printf("  %s", arguments[j].arg_name);
			len = (unsigned int) strlen(arguments[j].arg_name);
			while (len++ <= max_len) printf(" ");
			printf("        %s", arguments[j].arg_help);
			if (arguments[j].flags & ARG_FLAG_DEF)
				printf("[default from %s]",
				       arguments[j].default_value);
			else if (arguments[j].default_value)
				printf("[default argument: %s]",
				       arguments[j].default_value);
			printf("\n");
		}
		printf("\n");
	} else {
		printf("Usage: %s %s [options] %s\n", prog_name,
		       arguments[cmd - 1].arg_name,
		       arguments[cmd - 1].arg_extra_help);
	}

	printf("Options:\n");
	for (j = cmd; j < ARGUMENTS_SIZE; j++) {
		if (arguments[j].arg_type == ARG_CMD) break;
		len = 2;
		len += (unsigned int) strlen(arguments[j].arg_name);

		specifier = get_specifier(arguments[j].arg_type);
		len += (unsigned int) strlen(specifier);
		if (max_len < len) max_len = len;
	}

	for (j = cmd; j < ARGUMENTS_SIZE; j++) {
		if (arguments[j].arg_type == ARG_CMD) break;

		specifier = get_specifier(arguments[j].arg_type);
		printf("  %s %s", arguments[j].arg_name, specifier);
		len = 2;
		len += (unsigned int) strlen(arguments[j].arg_name);
		len += (unsigned int) strlen(specifier);
		while (len++ <= max_len) printf(" ");
		printf("        %s", arguments[j].arg_help);
		if (arguments[j].flags & ARG_FLAG_DEF)
			printf("[default from %s]",
			       arguments[j].default_value);
		else if (arguments[j].default_value)
			printf("[default: %s]", arguments[j].default_value);
		printf("\n");
	}
}

static
int get_argument(unsigned int cmd, const char *arg_name,
                 union argument_value *val)
{
	unsigned int j;

	if (arg_name == NULL && cmd != 0) {
		if (arguments[cmd - 1].arg_set) {
			*val = arguments[cmd - 1].value;
			return TRUE;
		}
		if (arguments[cmd - 1].default_value &&
		    !(arguments[cmd - 1].flags & ARG_FLAG_DEF)) {
			val->str_val = (char *)
			   arguments[cmd - 1].default_value;
			return TRUE;
		}
		return FALSE;
	}

	for (j = cmd; j < ARGUMENTS_SIZE; j++) {
		if (arguments[j].arg_type == ARG_CMD) break;
		if (strcmp(arguments[j].arg_name, arg_name) == 0) {
			if (arguments[j].arg_set) {
				*val = arguments[j].value;
				return TRUE;
			}
			if (arguments[j].default_value &&
			    !(arguments[j].flags & ARG_FLAG_DEF)) {
				parse_argument(arguments[j].arg_type,
				               arguments[j].default_value,
				               val);
				return TRUE;
			}
			return FALSE;
		}
	}
	error("wrong argument name `%s'", arg_name);
	return FALSE;
}

static
int process_arguments(int argc, char **argv, unsigned int *pcmd)
{
	unsigned int i, j, cmd;
	char *cmd_extra = NULL, *str;
	union argument_value val;
	int err;

	cmd = 0;
	for (i = 1; i < argc; i++) {
		if (cmd > 0) {
			for (j = cmd; j < ARGUMENTS_SIZE; j++) {
				if (arguments[j].arg_type == ARG_CMD)
					break;
				if (strcmp(arguments[j].arg_name,
				           argv[i]) == 0)
					goto consume_argument;
			}
			if (!cmd_extra && argv[i][0] != '-' &&
			    (arguments[cmd - 1].flags & ARG_FLAG_NEEDFILE)) {
				cmd_extra = argv[i];
				arguments[cmd - 1].value.str_val = cmd_extra;
				arguments[cmd - 1].arg_set = TRUE;
				continue;
			}
		} else {
			for (j = 0; j < ARGUMENTS_SIZE; j++) {
				if (arguments[j].arg_type == ARG_CMD)
					break;
				if (strcmp(arguments[j].arg_name,
				           argv[i]) == 0)
					goto consume_argument;
			}
			for (; j < ARGUMENTS_SIZE; j++) {
				if (arguments[j].arg_type != ARG_CMD)
					continue;
				if (strcmp(arguments[j].arg_name,
				           argv[i]) == 0)
					goto consume_command;
			}

			if (argv[i][0] != '-') {
				error("invalid command `%s'", argv[i]);
				return FALSE;
			}
		}
		error("invalid argument `%s'", argv[i]);
		return FALSE;

consume_argument:
		if (i == argc - 1 && arguments[j].arg_type != ARG_BOOL) {
			error("argument `%s' needs to be specified", argv[i]);
			return FALSE;
		}

		str = (arguments[j].arg_type == ARG_BOOL) ? NULL : argv[++i];
		parse_argument(arguments[j].arg_type, str,
		               &arguments[j].value);

		if (arguments[j].flags & ARG_FLAG_POS) {
			if (arguments[j].value.dbl_val <= 0) {
				error("argument to `%s' should be positive",
				      argv[i - 1]);
				return FALSE;
			}
		}
		if (arguments[j].flags & ARG_FLAG_PROB) {
			if (arguments[j].value.dbl_val < 0
			    || arguments[j].value.dbl_val > 1) {
				error("argument to `%s' should be between",
				      " zero and one", argv[i - 1]);
				return FALSE;
			}
		}
		if (arguments[j].flags & ARG_FLAG_BIGGER1) {
			if (arguments[j].value.dbl_val <= 1) {
				error("argument to `%s' should be at "
				      "least one", argv[i - 1]);
				return FALSE;
			}
		}

		arguments[j].arg_set = TRUE;
		continue;

consume_command:
		cmd = j + 1;
		continue;
	}

	*pcmd = 0;
	if (get_argument(0, "--help", &val)) {
		show_help(0, argv[0]);
		return TRUE;
	}
	if (get_argument(cmd, "--help", &val)) {
		show_help(cmd, argv[0]);
		return TRUE;
	}

	if (cmd == 0) {
		error("please specify a command");
		return FALSE;
	}

	err = FALSE;
	if ((arguments[cmd - 1].flags & ARG_FLAG_NEEDFILE)
	    && (!arguments[cmd - 1].arg_set)) {
		error("command `%s' needs a filename to be specified",
		       arguments[cmd - 1].arg_name);
		err = TRUE;
	}

	for (j = cmd; j < ARGUMENTS_SIZE; j++) {
		if (arguments[j].arg_type == ARG_CMD) break;
		if (arguments[j].flags & ARG_FLAG_REQ) {
			if (!arguments[j].arg_set
			    && !arguments[j].default_value) {
				error("please specify option `%s'",
				      arguments[j].arg_name);
				err = TRUE;
			}
		}
	}
	if (err) return FALSE;

	*pcmd = cmd;
	return TRUE;
}

static
int resize_image(unsigned int cmd)
{
	unsigned int width, height;
	const char *img_filename, *output_filename;
	union argument_value val;
	image img, out;

	if (!get_argument(cmd, NULL, &val))
		return FALSE;
	img_filename = val.str_val;

	image_init(&img);
	image_init(&out);
	if (!image_read(&img, img_filename))
		goto error_resize;

	if (!get_argument(cmd, "--output", &val))
		goto error_resize;
	output_filename = val.str_val;

	width = img.width;
	height = img.height;

	if (get_argument(cmd, "--width", &val))
		width = val.uint_val;

	if (get_argument(cmd, "--height", &val))
		height = val.uint_val;

	if (!image_resize(&img, &out, width, height))
		goto error_resize;

	if (!image_write(&out, output_filename))
		goto error_resize;

	image_cleanup(&img);
	image_cleanup(&out);
	return TRUE;

error_resize:
	image_cleanup(&img);
	image_cleanup(&out);
	return FALSE;
}

static
int detect_objects(unsigned int cmd)
{
	unsigned int i, step;
	const char *img_filename, *cascade_filename, *output_filename;
	double scale, min_stddev, match_thresh, overlap_thresh;
	unsigned int min_width, min_height, max_width, max_height;
	union argument_value val;
	int multi_exit;
	cascade c;
	image img;

	if (!get_argument(cmd, NULL, &val))
		return FALSE;
	img_filename = val.str_val;

	if (!get_argument(cmd, "--cascade", &val))
		return FALSE;
	cascade_filename = val.str_val;

	if (!get_argument(cmd, "--min_width", &val))
		return FALSE;
	min_width = val.uint_val;

	if (!get_argument(cmd, "--max_width", &val))
		return FALSE;
	max_width = val.uint_val;

	if (!get_argument(cmd, "--min_height", &val))
		return FALSE;
	min_height = val.uint_val;

	if (!get_argument(cmd, "--max_height", &val))
		return FALSE;
	max_height = val.uint_val;

	image_init(&img);
	if (!cascade_load(&c, cascade_filename, TRUE))
		goto error_detect;

	cascade_get_params(&c, &scale, &min_stddev, &step,
	                   &match_thresh, &overlap_thresh, &multi_exit);

	if (!get_argument(cmd, "--output", &val))
		goto error_detect;
	output_filename = val.str_val;

	if (get_argument(cmd, "--scale", &val))
		scale = val.dbl_val;

	if (get_argument(cmd, "--min_stddev", &val))
		min_stddev = val.dbl_val;

	if (get_argument(cmd, "--step", &val))
		step = val.uint_val;

	if (get_argument(cmd, "--match_thresh", &val))
		match_thresh = val.dbl_val;

	if (get_argument(cmd, "--overlap_thresh", &val))
		overlap_thresh = val.dbl_val;

	if (!image_read(&img, img_filename))
		goto error_detect;

	printf("scale = %g, min_stddev = %g, step = %u, "
	       "match_thresh = %g, overlap_thresh = %g\n",
	       scale, min_stddev, step, match_thresh, overlap_thresh);
	cascade_set_params(&c, scale, min_stddev, step,
	                   match_thresh, overlap_thresh, multi_exit);

	printf("min_width = %u, max_width = %u, "
	       "min_height = %u, max_height = %u\n",
	       min_width, max_width, min_height, max_height);
	cascade_set_scan(&c, min_width, min_height,
	                 max_width, max_height);

	cascade_set_image(&c, &img);

	if (!cascade_detect(&c, TRUE))
		goto error_detect;

	for (i = 0; i < c.num_detected_objects; i++) {
		detected_object *obj;

		obj = &c.detected_objects[i];
		printf("Object at (%u, %u, %u, %u)\n",
		       obj->w.left, obj->w.top, obj->w.width, obj->w.height);

		image_draw_window(&img, &obj->w, 255, 4);
	}
	printf("Num jumbled = %u\n", c.num_jumbled_objects);

	if (!image_write(&img, output_filename))
		goto error_detect;

	cascade_cleanup(&c);
	image_cleanup(&img);
	return TRUE;

error_detect:
	cascade_cleanup(&c);
	image_cleanup(&img);
	return FALSE;
}

static
int evaluate_cascade(unsigned int cmd)
{
	unsigned int step, num_cascades, num_threads;
	const char *cascade_filename, *test_filename;
	const char *testing_directory;
	double scale, min_stddev, match_thresh, overlap_thresh;
	unsigned int min_width, min_height, max_width, max_height;
	union argument_value val;
	int multi_exit;
	detector dt;
	samples smp;

	detector_reset(&dt);
	samples_reset(&smp);

	if (!get_argument(cmd, NULL, &val))
		goto error_evaluate;
	test_filename = val.str_val;

	if (!get_argument(cmd, "--cascade", &val))
		goto error_evaluate;
	cascade_filename = val.str_val;

	if (!get_argument(cmd, "--num_cascades", &val))
		goto error_evaluate;
	num_cascades = val.uint_val;

	if (!get_argument(cmd, "--num_threads", &val))
		goto error_evaluate;
	num_threads = val.uint_val;

	num_threads = MAX(1, num_threads);
	num_cascades = MAX(num_threads, num_cascades);

	if (!detector_load(&dt, cascade_filename, TRUE,
	                   num_cascades, num_threads))
		goto error_evaluate;

	detector_get_params(&dt, &scale, &min_stddev, &step,
	                    &match_thresh, &overlap_thresh, &multi_exit);

	if (!get_argument(cmd, "--datadir", &val))
		goto error_evaluate;
	testing_directory = val.str_val;

	if (get_argument(cmd, "--scale", &val))
		scale = val.dbl_val;

	if (get_argument(cmd, "--min_stddev", &val))
		min_stddev = val.dbl_val;

	if (get_argument(cmd, "--step", &val))
		step = val.uint_val;

	if (get_argument(cmd, "--match_thresh", &val))
		match_thresh = val.dbl_val;

	if (get_argument(cmd, "--overlap_thresh", &val))
		overlap_thresh = val.dbl_val;

	if (!get_argument(cmd, "--min_width", &val))
		goto error_evaluate;
	min_width = val.uint_val;

	if (!get_argument(cmd, "--max_width", &val))
		goto error_evaluate;
	max_width = val.uint_val;

	if (!get_argument(cmd, "--min_height", &val))
		goto error_evaluate;
	min_height = val.uint_val;

	if (!get_argument(cmd, "--max_height", &val))
		goto error_evaluate;
	max_height = val.uint_val;

	printf("scale = %g, min_stddev = %g, step = %u, "
	       "match_thresh = %g, overlap_thresh = %g\n",
	       scale, min_stddev, step, match_thresh, overlap_thresh);
	detector_set_params(&dt, scale, min_stddev, step,
	                    match_thresh, overlap_thresh, multi_exit);

	printf("min_width = %u, max_width = %u, "
	       "min_height = %u, max_height = %u\n",
	       min_width, max_width, min_height, max_height);
	detector_set_scan(&dt, min_width, min_height,
	                  max_width, max_height);

	if (!samples_read(&smp, test_filename))
		goto error_evaluate;

	if (!detector_evaluate(&dt, &smp, testing_directory))
		goto error_evaluate;

	detector_cleanup(&dt);
	samples_cleanup(&smp);
	return TRUE;

error_evaluate:
	detector_cleanup(&dt);
	samples_cleanup(&smp);
	return FALSE;
}

static
int train(unsigned int cmd)
{
	unsigned int max_stages;
	unsigned int max_classifiers;
	unsigned int positive_samples;
	unsigned int negative_samples;
	unsigned int width, height;
	unsigned int buckets;
	unsigned int num_parallels;
	unsigned int min_jumbled;
	unsigned int min_negative;
	unsigned int step;
	unsigned int num_threads;
	unsigned int max_planes;
	unsigned int max_unused;
	double match_thresh, overlap_thresh;
	double scale, min_stddev;
	double Cp, Cn;
	double bucket_min, bucket_max;
	double max_false_positive;
	double max_false_negative;
	double feature_prob, min_similarity;
	double eps, eps_qp, mu;
	int learn_overlap;
	int multi_exit;
	int cycle_parallels;
	char *filename;
	char *training_directory;
	char *cascade_filename;
	union argument_value val;
	trainer_data td;

	if (!get_argument(cmd, NULL, &val))
		return FALSE;
	filename = val.str_val;

	if (!get_argument(cmd, "--cascade", &val))
		return FALSE;
	cascade_filename = val.str_val;

	if (!get_argument(cmd, "--datadir", &val))
		return FALSE;
	training_directory = val.str_val;

	if (!get_argument(cmd, "--max_stages", &val))
		return FALSE;
	max_stages = val.uint_val;

	if (!get_argument(cmd, "--max_classifiers", &val))
		return FALSE;
	max_classifiers = val.uint_val;

	if (!get_argument(cmd, "--positive_samples", &val))
		return FALSE;
	positive_samples = val.uint_val;

	if (!get_argument(cmd, "--negative_samples", &val))
		return FALSE;
	negative_samples = val.uint_val;

	if (!get_argument(cmd, "--min_jumbled", &val))
		return FALSE;
	min_jumbled = val.uint_val;

	if (!get_argument(cmd, "--feature_prob", &val))
		return FALSE;
	feature_prob = val.dbl_val;

	if (!get_argument(cmd, "--width", &val))
		return FALSE;
	width = val.uint_val;

	if (!get_argument(cmd, "--height", &val))
		return FALSE;
	height = val.uint_val;

	if (!get_argument(cmd, "--match_thresh", &val))
		return FALSE;
	match_thresh = val.dbl_val;

	if (!get_argument(cmd, "--overlap_thresh", &val))
		return FALSE;
	overlap_thresh = val.dbl_val;

	learn_overlap = FALSE;
	if (get_argument(cmd, "--learn_overlap", &val))
		learn_overlap = TRUE;

	if (!get_argument(cmd, "--min_similarity", &val))
		return FALSE;
	min_similarity = val.dbl_val;

	if (!get_argument(cmd, "--num_parallels", &val))
		return FALSE;
	num_parallels = val.uint_val;

	cycle_parallels = FALSE;
	if (get_argument(cmd, "--cycle_parallels", &val))
		cycle_parallels = TRUE;

	if (!get_argument(cmd, "--buckets", &val))
		return FALSE;
	buckets = val.uint_val;

	if (!get_argument(cmd, "--bucket_min", &val))
		return FALSE;
	bucket_min = val.dbl_val;

	if (!get_argument(cmd, "--bucket_max", &val))
		return FALSE;
	bucket_max = val.dbl_val;

	if (!get_argument(cmd, "--min_negative_samples", &val))
		return FALSE;
	min_negative = val.uint_val;

	if (!get_argument(cmd, "--Cp", &val))
		return FALSE;
	Cp = val.dbl_val;

	if (!get_argument(cmd, "--Cn", &val))
		return FALSE;
	Cn = val.dbl_val;

	if (!get_argument(cmd, "--scale", &val))
		return FALSE;
	scale = val.dbl_val;

	if (!get_argument(cmd, "--min_stddev", &val))
		return FALSE;
	min_stddev = val.dbl_val;

	if (!get_argument(cmd, "--step", &val))
		return FALSE;
	step = val.uint_val;

	multi_exit = FALSE;
	if (get_argument(cmd, "--multi_exit", &val))
		multi_exit = TRUE;

	if (!get_argument(cmd, "--max_false_positive", &val))
		return FALSE;
	max_false_positive = val.dbl_val;

	if (!get_argument(cmd, "--max_false_negative", &val))
		return FALSE;
	max_false_negative = val.dbl_val;

	if (!get_argument(cmd, "--num_threads", &val))
		return FALSE;
	num_threads = val.uint_val;

	if (!get_argument(cmd, "--max_planes", &val))
		return FALSE;
	max_planes = val.uint_val;

	if (!get_argument(cmd, "--eps", &val))
		return FALSE;
	eps = val.dbl_val;

	if (!get_argument(cmd, "--eps_qp", &val))
		return FALSE;
	eps_qp = val.dbl_val;

	if (!get_argument(cmd, "--mu", &val))
		return FALSE;
	mu = val.dbl_val;

	if (!get_argument(cmd, "--max_unused", &val))
		return FALSE;
	max_unused = val.uint_val;

	if (!trainer_init(&td, filename, width, height,
	                  positive_samples, negative_samples,
	                  buckets, num_parallels, num_threads,
	                  0, max_planes))
		goto error_train;

	trainer_boost_params(&td, Cp, Cn, bucket_min, bucket_max);
	trainer_cascade_params(&td, multi_exit, scale, min_stddev, step,
	                       match_thresh, overlap_thresh, learn_overlap);
	trainer_cpa_params(&td, Cp, Cn, eps, eps_qp, mu, max_unused);
	trainer_params(&td, max_stages, max_classifiers, min_jumbled,
	               min_negative, max_false_positive, max_false_negative,
	               feature_prob, min_similarity, cycle_parallels);

	if (!trainer_train(&td, cascade_filename, training_directory))
		goto error_train;

	trainer_cleanup(&td);
	return TRUE;

error_train:
	trainer_cleanup(&td);
	return FALSE;
}

int main(int argc, char **argv)
{
	unsigned int cmd;
	const char *cmd_name;
#ifdef CONSOLE_UNBUFFERED
	setvbuf(stdout, 0, _IONBF, 0);
	setvbuf(stderr, 0, _IONBF, 0);
#endif
	genrand_randomize();

	if (!process_arguments(argc, argv, &cmd))
		return 1;

	if (cmd == 0)
		return 0;

	cmd_name = arguments[cmd - 1].arg_name;
	if (strcmp("resize", cmd_name) == 0) {
		if (!resize_image(cmd))
			return 1;
	} else if (strcmp("detect", cmd_name) == 0) {
		if (!detect_objects(cmd))
			return 1;
	} else if (strcmp("evaluate", cmd_name) == 0) {
		if (!evaluate_cascade(cmd))
			return 1;
	} else {
		if (!train(cmd))
			return 1;
	}

	return 0;
}

