# Optional file used to get custom XML file with specified points
# Change LANDMARKS variable and run script

# importing packages
import argparse
import re

# Setting argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
                help="path to iBug 300-W data split XML file")
ap.add_argument("-t", "--output", required=True,
                help="path output data split XML file")
args = vars(ap.parse_args())

# Used to open and write file using:
# python parse_xml.py -i {input} -t {ouput}
# 	args: --input ../data/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml \
# 	args: --output ../data/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train_eyes.xml
# ex: python .\parse_xml.py -i ../data/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train.xml -t ../data/ibug_300W_large_face_landmark_dataset/labels_ibug_300W_train_eyes.xml


# Setting landmark range (Currently for all points)
LANDMARKS = set(list(range(0, 67)))

# utilize regular expression to determine if there is a part on a line
PART = re.compile("part name='[0-9]+'")

# load the contents of the original XML file
print("[INFO] parsing data split XML file...")
rows = open(args["input"]).read().strip().split("\n")
output = open(args["output"], "w")

# loop over the rows of the data split file
for row in rows:
    # check to see if the current line has the (x, y)-coordinates for
    # the facial landmarks we are interested in
    parts = re.findall(PART, row)

    # if there is no information related to the (x, y)-coordinates of
    # the facial landmarks, write the current line out to disk
    if len(parts) == 0:
        output.write("{}\n".format(row))

    # otherwise, there is annotation information
    else:
        # parse out the name of the attribute from the row
        attr = "name='"
        i = row.find(attr)
        j = row.find("'", i + len(attr) + 1)
        name = int(row[i + len(attr):j])
        # if the facial landmark name exists within the range of targetted
        # indexes, write it to output file
        if name in LANDMARKS:
            output.write("{}\n".format(row))
# close the output file
output.close()
