
import cv2
import dlib
import os.path

def rectangles_overlap(rect1, rect2, match_thresh, overlap_thresh):
	left = max(rect1[0], rect2[0])
	right = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
	if left >= right:
		return False
	top = max(rect1[1], rect2[1])
	bottom = min(rect1[1] + rect1[3], rect2[1] + rect2[3])
	if top >= bottom:
		return False

	areaI = (right - left) * (bottom - top)
	area1 = rect1[2] * rect1[3]
	if areaI >= match_thresh * area1:
		return True
	area2 = rect2[2] * rect2[3]
	if areaI >= match_thresh * area2:
		return True
	areaU = area1 + area2 - areaI
	return (areaI >= overlap_thresh * areaU)

def ocv_detect_objects(pictures, cascade_file = 'default.xml', input_dir = '',
                       scale = 1.1, width = 24, height = 24):
	csc = cv2.CascadeClassifier(cascade_file)
	answers = dict()
	for picture in pictures:
		filename = os.path.join(input_dir, picture)
		print "Processing", filename, "..."
		img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
		objs = csc.detectMultiScale(img, scaleFactor = scale,
		                            minNeighbors = 0,
		                            minSize = (height, width),
		                            flags = cv2.CASCADE_SCALE_IMAGE)
		rectangles = []
		for rect in objs:
			overlap = False
			for other in rectangles:
				if rectangles_overlap(rect, other, 0.75, 0.33):
					overlap = True
					break
			if not overlap:
				rectangles.append(rect)
		answers[picture] = rectangles

	return answers

def dlib_detect_faces(pictures, input_dir = ''):
	detector = dlib.get_frontal_face_detector()
	answers = dict()

	for picture in pictures:
		filename = os.path.join(input_dir, picture)
		print "Processing", filename, "..."
		img = cv2.imread(filename)
		faces = detector(img, 1)
		rectangles = list()
		for face in faces:
			rectangles.append((face.left(), face.top(),
			                   face.right() - face.left(),
			                   face.bottom() - face.top()))
		answers[filename] = rectangles
	return answers

def compute_scores(answers, candidate, match_thresh, overlap_thresh):
	total_fn = 0
	total_fp = 0
	total_objs = 0
	scores = dict()
	for filename, ans in answers.iteritems():
		cand = list(candidate[filename])
		fn = 0
		for rect1 in ans:
			found = False
			for j in xrange(len(cand)):
				rect2 = cand[j]
				if rectangles_overlap(rect1, rect2,
				                      match_thresh,
				                      overlap_thresh):
					found = True
					cand.pop(j)
					break
			if not found:
				fn += 1
		objs = len(cand)
		fp = len(cand)
		scores[filename] = (fp, fn, objs)

	return scores

def print_scores(scores):
	fp = 0
	fn = 0
	objs = 0
	for filename, score in scores.iteritems():
		print "%s: fp = %d, fn = %d, objs = %d" % \
		  (filename, score[0], score[1], score[2])
		fp += score[0]
		fn += score[1]
		objs += score[2]
	print "Total: fp =", fp, "fn =", fn, "objs = ", objs

