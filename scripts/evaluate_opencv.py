
import detect
import utils
import argparse

def evaluate_opencv(cascade_file, test_file, match_thresh, overlap_thresh):
	samples = utils.read_samples_file(test_file, "y")
	pictures = samples.keys()
	candidate = detect.ocv_detect_objects(pictures, cascade_file)
	scores = detect.compute_scores(samples, candidate,
	                               match_thresh, overlap_thresh)
	detect.print_scores(scores)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("cascade", help = "Name of the cascade file")
	parser.add_argument("test", help = "Width of the test file")
	parser.add_argument("match_thresh", help = "parameter for match")
	parser.add_argument("overlap_thresh", help = "parameter for overlap")
	args = parser.parse_args()
	match_thresh = float(args.match_thresh)
	overlap_thresh = float(args.overlap_thresh)
	evaluate_opencv(args.cascade, args.test, match_thresh, overlap_thresh)
