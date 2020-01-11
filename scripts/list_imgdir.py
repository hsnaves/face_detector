import cv2
import os.path
import os
import argparse


def list_directory(dirname, positive = "y"):
	print "positive,left,top,width,height,filename"
	for f in os.listdir(dirname):
		filename = os.path.join(dirname, f)
		if not os.path.isfile(filename):
			continue

		basename, extension = os.path.splitext(f)
		if not extension in [".png", ".jpg", ".jpeg"]:
			continue

		img = cv2.imread(filename)
		width = img.shape[1]
		height = img.shape[0]
		print "%s,0,0,%d,%d,%s" % (positive, width, height, filename)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("dirname", help = "Name of the directory " + \
	                    "with images")
	parser.add_argument("positive", help = "If positive")
	args = parser.parse_args()
	list_directory(args.dirname, args.positive)



