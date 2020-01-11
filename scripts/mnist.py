import numpy as np
import cv2

def convert(imgf, width, height, n):
	f = open(imgf, "rb")
	f.read(16)
	for i in xrange(n):
		image = np.zeros((height, width), dtype = np.uint8)
		for row in xrange(height):
			for col in xrange(width):
				image[row, col] = ord(f.read(1))
		print "Writing image", i + 1
		cv2.imwrite("mnist/image%d.png" % i, image)

	f.close()

convert("train-images-idx3-ubyte", 28, 28, 60000)


