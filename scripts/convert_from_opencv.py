import xml.etree.ElementTree
import argparse

def find_feature(rectangles):
	l = len(rectangles)
	left0 = int(rectangles[0][0])
	top0 = int(rectangles[0][1])
	width0 = int(rectangles[0][2])
	height0 = int(rectangles[0][3])
	weight0 = float(rectangles[0][4])

	left1 = int(rectangles[1][0])
	top1 = int(rectangles[1][1])
	width1 = int(rectangles[1][2])
	height1 = int(rectangles[1][3])
	weight1 = float(rectangles[1][4])

	if l == 2:
		if width0 == 2 * width1 and height0 == height1 \
		   and top0 == top1 and left1 == left0 + width1 \
		   and weight0 == -1 and weight1 == 2:
			return (0, left0, top0, width1, height0, -1)

		if width0 == 2 * width1 and height0 == height1 \
		   and top0 == top1 and left1 == left0 \
		   and weight0 == -1 and weight1 == 2:
			return (0, left0, top0, width1, height0, +1)

		if width0 == width1 and height0 == 2 * height1 \
		   and top1 == top0 + height1 and left1 == left0 \
		   and weight0 == -1 and weight1 == 2:
			return (1, left0, top0, width0, height1, -1)

		if width0 == width1 and height0 == 2 * height1 \
		   and top0 == top1 and left1 == left0 \
		   and weight0 == -1 and weight1 == 2:
			return (1, left0, top0, width1, height0, +1)

		if width0 == 3 * width1 and height0 == height1\
		   and left1 == left0 + width1 and top1 == top0\
		   and weight0 == -1 and weight1 == 3:
			return (2, left0, top0, width1, height0, -1)

		if width0 == 2 * width1 and height0 == height1\
		   and left1 == left0 + (width1 / 2) and top1 == top0\
		   and weight0 == -1 and weight1 == 2:
			return (3, left0, top0, width1 / 2, height0, -1)

		if width0 == width1 and height0 == 3 * height1\
		   and left0 == left1 and top1 == top0 + height1\
		   and weight0 == -1 and weight1 == 3:
			return (4, left0, top0, width0, height1, -1)

		if width0 == width1 and height0 == 2 * height1\
		   and left1 == left0 and top1 == top0 + (height1 / 2)\
		   and weight0 == -1 and weight1 == 2:
			return (5, left0, top0, width1 / 2, height0, -1)

	if l == 3:
		left2 = int(rectangles[2][0])
		top2 = int(rectangles[2][1])
		width2 = int(rectangles[2][2])
		height2 = int(rectangles[2][3])
		weight2 = float(rectangles[2][4])

		if left0 == left1 and top0 == top1\
		   and width0 == 2 * width1 and height0 == 2 * height1\
		   and left2 == left1 + width1 and top2 == top1 + height1\
		   and width1 == width2 and height1 == height2\
		   and weight0 == -1 and weight1 == 2 and weight2 == 2:
			return (7, left0, top0, width1, height1, +1)

		if left1 == left0 + width1 and top0 == top1\
		   and width0 == 2 * width1 and height0 == 2 * height1\
		   and left2 == left0 and top2 == top0 + height2\
		   and width1 == width2 and height1 == height2\
		   and weight0 == -1 and weight1 == 2 and weight2 == 2:
			return (7, left0, top0, width1, height1, -1)

	print rectangles
	return None

def process_file(filename, output_filename):
	tree = xml.etree.ElementTree.parse(filename)
	root = tree.getroot()

	cascade = root.find("cascade")
	width = int(cascade.find("width").text)
	height = int(cascade.find("height").text)
	area = (width - 1) * (height - 1)

	list_features = []
	features = cascade.find("features")
	for feature in features.findall("_"):
		rects = feature.find("rects")
		rectangles = []
		for rect in rects.findall("_"):
			coordinates = rect.text.strip().split(" ")
			rectangles.append(coordinates)
		feature = find_feature(rectangles)
		list_features.append(feature)

	stage_list = []
	stages = cascade.find("stages")
	for stage in stages.findall("_"):
		stage_intercept = -float(stage.find("stageThreshold").text)
		classifier_list = []
		classifiers = stage.find("weakClassifiers")
		for classifier in classifiers.findall("_"):
			internal = classifier.find("internalNodes").text
			leaf = classifier.find("leafValues").text
			internal = internal.strip().split(" ")
			leaf = leaf.strip().split(" ")
			coef0 = float(leaf[0])
			coef1 = float(leaf[1])
			feat_idx = int(internal[2])
			thresh = area * float(internal[3])
			feature = list_features[feat_idx]
			if feature[5] < 0:
				thresh = -thresh
				coef0, coef1 = coef1, coef0

			stage_intercept += coef0
			weak = (coef1 - coef0, thresh, feature)
			classifier_list.append(weak)
			#print internal, leaf
		stage_list.append((stage_intercept, classifier_list))

	with open(output_filename, "w") as f:
		f.write("%d %d 1\n" % (width, height))
		f.write("1.1 10.0 1 0.75 0.33\n")
		f.write("0\n")
		f.write("%d\n" % len(stage_list))
		for stage in stage_list:
			f.write("%d\n" % len(stage[1]))
			for i, weak in enumerate(stage[1]):
				coef = weak[0]
				intercept = 0
				if i == 0:
					intercept = stage[0]
				thresh = weak[1]
				f.write("%g %g %g %d %d %d %d %d\n" % (
				   coef, thresh, intercept, weak[2][0],
				   weak[2][1], weak[2][2], weak[2][3],
				   weak[2][4]
				))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("xmlfile", help = "Name of the openCV XML file")
	parser.add_argument("output", help = "Name of the output file")
	args = parser.parse_args()
	process_file(args.xmlfile, args.output)

