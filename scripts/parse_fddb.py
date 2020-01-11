
import math
import re
import sys
import cv2

def process_file(filename):
	result = dict()
	with open(filename, "r") as f:
		state = 0
		filename = ""
		objects = []
		for line in f:
			if state == 0:
				filename = "FDDB/" + line.strip() + ".jpg"
				im = cv2.imread(filename)
				im_width = im.shape[1]
				im_height = im.shape[0]
				state = 1
			elif state == 1:
				num_objects = int(line.strip())
				objects = []
				state = 2
			elif state == 2:
				args = re.split("[ ]+", line.strip())
				major_r = float(args[0])
				minor_r = float(args[1])
				angle = float(args[2])
				center_x = float(args[3])
				center_y = float(args[4])
				v1_x = major_r * math.cos(angle)
				v1_y = major_r * math.sin(angle)
				v2_x = -minor_r * math.sin(angle)
				v2_y = minor_r * math.cos(angle)
				max_x = max(v1_x, -v1_x, v2_x, -v2_x)
				min_x = min(v1_x, -v1_x, v2_x, -v2_x)
				max_y = max(v1_y, -v1_y, v2_y, -v2_y)
				min_y = min(v1_y, -v1_y, v2_y, -v2_y)
				left = int(center_x + min_x)
				left = max(left, 0)
				right = int(center_x + max_x)
				right = min(right, im_width - 1)
				top = int(center_y + min_y)
				top = max(top, 0)
				bottom = int(center_y + max_y)
				bottom = min(bottom, im_height - 1)
				width = right - left
				height = bottom - top
				if width < 0 or height < 0:
					print major_r, minor_r, angle, center_x, center_y
					print width, height
					print left, top, right, bottom
					print im_width, im_height
					sys.exit(1)

				obj = (left, top, width, height)
				objects.append(obj)
				if len(objects) >= num_objects:
					result[filename] = objects
					state = 0
	return result

if __name__ == "__main__":
	results = []
	for i in xrange(10):
		filename = "FDDB-fold-%02d-ellipseList.txt" % (i + 1)
		results.append(process_file(filename))
	consolidated = dict()
	for r in results:
		for key, val in r.iteritems():
			if key in consolidated:
				if val != consolidated[key]:
					print "Problem in ", key
			consolidated[key] = val
	print "positive,left,top,width,height,filename"
	for key, val in consolidated.iteritems():
		filename = key
		for obj in val:
			print "y,%d,%d,%d,%d,%s" % (
			    obj[0], obj[1], obj[2], obj[3], filename
			)
