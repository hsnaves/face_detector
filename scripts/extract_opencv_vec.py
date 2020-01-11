import struct, array

import cv2
import numpy as np
import argparse

def extractvec(fn, width = 24, height = 24):
	f = open(fn,'rb')
	HEADERTYP = '<iihh' # img count, img size, min, max

	# read header
	imgcount, imgsize, _, _ = struct.unpack(HEADERTYP, f.read(12))

	print "Image count: ", imgcount
	for i in range(imgcount):
		img  = np.zeros((height, width), np.uint8)
		f.read(1) # read gap byte

		data = array.array('h')
		###  buf = f.read(imgsize*2)
		###  data.fromstring(buf)

		data.fromfile(f, imgsize)

		for r in range(height):
			for c in range(width):
				img[r,c] = data[r * width + c]

		cv2.imwrite("image%d.png" % i, img)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("vec", help = "Name of the VEC file")
	parser.add_argument("width", help = "Width of each image")
	parser.add_argument("height", help = "Height of each image")
	args = parser.parse_args()
	width = int(args.width)
	height = int(args.height)
	extractvec(args.vec, width, height)
