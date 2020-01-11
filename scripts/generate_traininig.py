
import csv
import detect
import utils
import os.path
import argparse


def read_pictures(filename, input_dir = ''):
	pictures = []
	with open(filename, "rb") as f:
		csvreader = csv.DictReader(f, delimiter=',', quotechar='"')
		for row in csvreader:
			imgfile = row['filename']
			filename = os.path.join(input_dir, imgfile)
			pictures.append(filename)

	return pictures


def process_pictures(filename, output_filename, cascade_xml, input_dir = ''):
	pictures = read_pictures(filename, input_dir)
	if cascade_xml is None:
		answers = detect.dlib_detect_faces(pictures)
	else:
		answers = detect.ocv_detect_objects(pictures, cascade_xml)

	with open(output_filename, "w") as f:
		f.write("positive,left,top,width,height,filename\n")
		utils.write_samples_file(f, pictures, answers, "y")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("csv_file", help = "Name of the CSV file")
	parser.add_argument("output_file", help = "Name of the output file")
	parser.add_argument("data_dir", help = "Name of the data directory")
	parser.add_argument("--cascade_xml", help = "Name of the opencv " + \
	                    "cascade file")
	args = parser.parse_args()
	process_pictures(args.csv_file, args.output_file,
	                 args.cascade_xml, args.data_dir)

