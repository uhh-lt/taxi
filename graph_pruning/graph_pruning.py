
import csv
import sys
import random


hypo_to_hyper = {}


def add_line(node):
    id = node[0]
    hypo = node[1]
    hyper = node[2]

    print("Read line with ID '%s': %s --> %s" % (id, hypo, hyper))

    if hypo not in hypo_to_hyper:
        hypo_to_hyper[hypo] = hyper
    elif random.randint(0, 1) == 0:
        print("Replace hypernym of hyponym '%s' with: %s" % (hypo, hyper))
        hypo_to_hyper[hypo] = hyper
    else:
        print("Keep hypernym '%s' of hyponym '%s'." % (hypo_to_hyper[hypo], hypo))


########################################################################################################################
#
# Main
#
########################################################################################################################
filename_in = None
filename_out = None
delimiter = '\t'

if len(sys.argv) >= 2:
    filename_in = sys.argv[1]
    
if len(sys.argv) >= 3:
    filename_out = sys.argv[2]

if len(sys.argv) >= 4:
    delimiter = sys.argv[3]


if filename_in is None:
    raise Exception("No CSV-file provided")

if filename_out is None:
    raise Exception("No output provided")


print("Reading file: %s" % filename_in)
print("Delimiter: %s" % delimiter)

with open(filename_in, "r") as f:
    reader = csv.reader(f, delimiter=delimiter)

    for i, line in enumerate(reader):
        add_line(line)

print("Relations created (hypo_to_hyper):")
print(hypo_to_hyper)

print("Write result.")

with open(filename_out, "w+") as f:
    writer = csv.writer(f, delimiter=delimiter)
    id = 0

    for hypo in hypo_to_hyper:
        row = [id, hypo, hypo_to_hyper[hypo]]

        print("   %s" % row)
        writer.writerow(row)

        id += 1





