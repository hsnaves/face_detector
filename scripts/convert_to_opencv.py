import argparse

def print_rectangle(f, weight, left, top, width, height):
	f.write("          ")
	f.write("<_>%d %d %d %d %g</_>\n" %\
	        (left, top, width, height, weight))

def print_feature(f, feat_idx, left, top, width, height):
	if feat_idx == 0:
		print_rectangle(f,  1, left, top, width, height)
		print_rectangle(f, -1, left + width, top, width, height)
	elif feat_idx == 1:
		print_rectangle(f,  1, left, top, width, height)
		print_rectangle(f, -1, left, top + height, width, height)
	elif feat_idx == 2:
		print_rectangle(f,  1, left, top, 3 * width, height)
		print_rectangle(f, -3, left + width, top, width, height)
	elif feat_idx == 3:
		print_rectangle(f,  1, left, top, 4 * width, height)
		print_rectangle(f, -2, left + width, top, 2 * width, height)
	elif feat_idx == 4:
		print_rectangle(f,  1, left, top, width, 3 * height)
		print_rectangle(f, -3, left, top + height, width, height)
	elif feat_idx == 5:
		print_rectangle(f,  1, left, top, width, 4 * height);
		print_rectangle(f, -2, left, top + height, width, 2 * height)
	elif feat_idx == 6:
		print_rectangle(f,  1, left, top, 3 * width, 3 * height)
		print_rectangle(f, -9, left + width, top + height,
		                width, height)
	elif feat_idx == 7:
		print_rectangle(f,  2, left, top, width, height)
		print_rectangle(f,  2, left + width, top + height,
		                width, height)
		print_rectangle(f, -1, left, top, 2 * width, 2 * height)

def process_file(filename, output_filename, parallel = 0):
	with open(filename, "r") as f:
		stage_list = []
		max_weak = 0

		line = f.readline()
		parts = line.split(" ")
		width = int(parts[0])
		height = int(parts[1])
		num_parallels = int(parts[2])

		line = f.readline()
		# discard these parameters
		line = f.readline()
		# discard multi-exit

		line = f.readline()
		num_stages = int(line)
		for i in xrange(num_stages):
			line = f.readline()
			stage = []
			num_weak = int(line)
			for j in xrange(num_weak):
				line = f.readline()
				parts = line.split(" ")
				coef = float(parts[0])
				thresh = float(parts[1])
				intercept = float(parts[2])
				feat_idx = int(parts[3])
				f_left = int(parts[4])
				f_top = int(parts[5])
				f_width = int(parts[6])
				f_height = int(parts[7])
				weak = (coef, thresh, intercept, feat_idx,
				        f_left, f_top, f_width, f_height)
				stage.append(weak)

			if i % num_parallels == parallel:
				max_weak = max(max_weak, num_weak)
				stage_list.append(stage)

	with open(output_filename, "w") as f:
		f.write("<?xml version=\"1.0\"?>\n")
		f.write("<opencv_storage>\n")
		f.write("  <cascade type_id=\"opencv-cascade-classifier\">\n")
		f.write("    <stageType>BOOST</stageType>\n")
		f.write("    <featureType>HAAR</featureType>\n")
		f.write("    <height>%d</height>\n" % height)
		f.write("    <width>%d</width>\n" % width)
		f.write("    <stageParams>\n")
		f.write("      <maxWeakCount>%d</maxWeakCount>\n" % max_weak)
		f.write("    </stageParams>\n")
		f.write("    <featureParams>\n")
		f.write("      <maxCatCount>0</maxCatCount>\n")
		f.write("    </featureParams>\n")
		f.write("    <stageNum>%d</stageNum>\n" % num_stages);
		f.write("    <stages>\n");

		rect_count = 0
		area = width * height
		for stage in stage_list:
			f.write("      <_>\n")
			f.write("        ")
			f.write("<maxWeakCount>%d</maxWeakCount>\n" % \
			        len(stage))
			intercepts = map(lambda x: x[2], stage)
			total_intercept = reduce(lambda x,y: x+y, intercepts)
			f.write("        ")
			f.write("<stageThreshold>%g</stageThreshold>\n" %\
			        -total_intercept)
			f.write("        <weakClassifiers>\n")
			for weak in stage:
				f.write("          <_>\n")
				f.write("            ")
				f.write("<internalNodes>")
				f.write("0 -1 %d %g</internalNodes>\n" %\
				        (rect_count, weak[1] / area))
				f.write("            ")
				f.write("<leafValues>%g %g</leafValues>\n" %\
				        (0.0, weak[0]))
				f.write("          </_>\n")
				rect_count += 1

			f.write("        </weakClassifiers>\n")
			f.write("      </_>\n")

		f.write("    </stages>\n")
		f.write("    <features>\n")

		for stage in stage_list:
			for weak in stage:
				f.write("      <_>\n")
				f.write("        <rects>\n")
				print_feature(f, weak[3], weak[4], weak[5],
				              weak[6], weak[7])
				f.write("        </rects>\n")
				f.write("      </_>\n")

		f.write("    </features>\n")
		f.write("  </cascade>\n")
		f.write("</opencv_storage>\n\n")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("cascadefile", help = "Name of the cascade file")
	parser.add_argument("output", help = "Name of the output file")
	args = parser.parse_args()
	process_file(args.cascadefile, args.output)

