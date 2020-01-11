import csv

def read_samples_file(filename, keep_positive = "y"):
	answers = dict()
	with open(filename, "rb") as f:
		csvreader = csv.DictReader(f, delimiter=',', quotechar='"')
		for row in csvreader:
			imgfile = row['filename']
			if row['positive'] != keep_positive:
				continue

			if not imgfile in answers:
				answers[imgfile] = list()

			left = int(row['left'])
			top = int(row['top'])
			width = int(row['width'])
			height = int(row['height'])
			box = (left, top, width, height)
			answers[imgfile].append(box)

	return answers

def write_samples_file(f, images, answers, keep_positive = "y"):
	for image in images:
		for box in answers[image]:
			f.write("%s,%d,%d,%d,%d,%s\n" % (
			  keep_positive, box[0], box[1], box[2], box[3], image
			))
