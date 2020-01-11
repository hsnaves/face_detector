import utils
import argparse

def generate_info_file(filename, output_filename):
	answers = utils.read_samples_file(filename, "y")

	with open(output_filename, "w") as f:
		for imgfile, boxes in answers.iteritems():
			f.write("%s %d" % (imgfile, len(boxes)))
			for box in boxes:
				f.write(" %d %d %d %d" % box)
			f.write("\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("train", help = "Name of the CSV training file")
	parser.add_argument("info", help = "Name of the output INFO file")
	args = parser.parse_args()
	generate_info_file(args.train, args.info)
