
import utils
import argparse

def generate_xml_file(filename, dataset_name, output_filename):
	answers = utils.read_samples_file(filename, "y")
	negative_answers = utils.read_samples_file(filename, "n")

	with open(output_filename, "w") as f:
		f.write("<?xml version='1.0' encoding='ISO-8859-1'?>\n")
		f.write("<?xml-stylesheet type='text/xsl' href='image_metadata_stylesheet.xsl'?>\n")
		f.write("<dataset>\n")
		f.write("<name>%s</name>\n" % dataset_name)
		f.write("<comment></comment>\n")
		f.write("<images>\n")
		for imgfile, boxes in answers.iteritems():
			f.write("  <image file='%s'>\n" % imgfile)
			for box in boxes:
				f.write("    <box left='%d' top='%d' width='%d' height='%d' />\n" % box)
			f.write("  </image>\n")
		for imgfile, boxes in negative_answers.iteritems():
			f.write("  <image file='%s'>\n" % imgfile)
			f.write("  </image>\n")
		f.write("</images>\n")
		f.write("</dataset>\n")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("train", help = "Name of the CSV training file")
	parser.add_argument("dataset", help = "Name of the dataset")
	parser.add_argument("xml", help = "Name of output XML file")
	args = parser.parse_args()
	generate_xml_file(args.train, args.dataset, args.xml)
