import utils
from random import shuffle
import argparse

def generate_folds(positive_fn, negative_fn, num_samples, train_fn, test_fn):
	answers = utils.read_samples_file(positive_fn, "y")
	negative_answers = utils.read_samples_file(negative_fn, "n")

	images = answers.keys()
	negative_images = negative_answers.keys()

	shuffle(images)
	train = images[:num_samples]
	test = images[num_samples:]

	with open(train_fn, "w") as f:
		f.write("positive,left,top,width,height,filename\n")
		utils.write_samples_file(f, train, answers, "y")
		utils.write_samples_file(f, negative_images, negative_answers,
		                         "n")

	with open(test_fn, "w") as f:
		f.write("positive,left,top,width,height,filename\n")
		utils.write_samples_file(f, test, answers, "y")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("num_folds", help = "Number of folds")
	parser.add_argument("positive", help = "Name of file containing " + \
	                    "positive examples")
	parser.add_argument("negative", help = "Name of file containing " + \
	                    "negative examples")
	parser.add_argument("num_samples", help = "Number of positives " + \
	                    "samples for training")
	args = parser.parse_args()
	num_folds = int(args.num_folds)
	num_samples = int(args.num_samples)
	for fold in xrange(1, num_folds + 1):
		train_fn = "train%d.csv" % fold
		test_fn = "test%d.csv" % fold
		generate_folds(args.positive, args.negative, num_samples,
		               train_fn, test_fn)
