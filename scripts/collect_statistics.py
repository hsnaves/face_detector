import re
import argparse

def process_file(filename):
	total_fp = 0
	total_fn = 0
	total_objects = 0
	prog = re.compile(r"^total fp = (\d+), total fn = (\d+), " \
	                  + "total objects = (\d+)")
	with open(filename, "rb") as f:
		for line in f:
			result = prog.match(line)
			if result:
				fp = int(result.group(1))
				fn = int(result.group(2))
				num_objects = int(result.group(3))
				total_fp += fp
				total_fn += fn
				total_objects += num_objects

	print "total fp =", total_fp, "total fn =", total_fn, \
	      "total objects =", total_objects

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("logfile", help = "Name of the log file")
	args = parser.parse_args()
	process_file(args.logfile)
