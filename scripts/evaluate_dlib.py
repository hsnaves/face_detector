
import utils
import detect
import argparse


def evaluate_dlib(test_file, match_thresh, overlap_thresh):
	samples = utils.read_samples_file(test_file, "y")
	pictures = samples.keys()
	candidate = detect.dlib_detect_faces(pictures)
	scores = detect.compute_scores(samples, candidate,
	                               match_thresh, overlap_thresh)
	detect.print_scores(scores)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("test", help = "Width of the test file")
	parser.add_argument("match_thresh", help = "parameter for match")
	parser.add_argument("overlap_thresh", help = "parameter for overlap")
	args = parser.parse_args()
	match_thresh = float(args.match_thresh)
	overlap_thresh = float(args.overlap_thresh)
	evaluate_dlib(args.test, match_thresh, overlap_thresh)
