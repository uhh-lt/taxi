
import csv
import sys
import random
from tarjan import tarjan

hypo_to_hyper = {}
hyper_to_hypo = {}


def prepare_tarjan(node):
    id = node[0]
    hypo = node[1]
    hyper = node[2]

    print("Read line with ID '%s': %s --> %s" % (id, hypo, hyper))

    if hyper not in hyper_to_hypo:
        hyper_to_hypo[hyper] = []

    if hypo not in hyper_to_hypo:
        hyper_to_hypo[hypo] = []

    hyper_to_hypo[hyper].append(hypo)



def do_tarjan():
    cycles_removed = 0

    cycle = []  # Initialize with value to trigger the while loop. Python has no do-while...


    while(cycle is not None):
        print("Running tarjan...")
        t = tarjan(hyper_to_hypo)

        # ['plant pathology','pathology'],

        i = 0
        cycle = None

        while(cycle is None and i < len(t)):
            if len(t[i]) > 1:
                # Do pruning
                print("Cycle detected: %s" % t[i])
                cycle = t[i]

            i += 1

        if cycle is not None:
            hypernym_index_removed_from = random.randint(0, len(cycle) - 1)
            hypernym_removed_from = cycle[hypernym_index_removed_from]

            for c in cycle:
                if c in hyper_to_hypo[hypernym_removed_from]:
                    print("Remove hyponym '%s' from hypernym '%s'." % (c, hypernym_removed_from))
                    hyper_to_hypo[hypernym_removed_from].remove(c)
                    cycles_removed += 1
                    break


    print("Removed %s cycles." % cycles_removed)
    return hyper_to_hypo




def add_line(node):
    id = node[0]
    hypo = node[1]
    hyper = node[2]

    #print("Read line with ID '%s': %s --> %s" % (id, hypo, hyper))

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
mode = "old"
root = None

supported_modes = ["tarjan", "old"]

if len(sys.argv) >= 2:
    filename_in = sys.argv[1]

if len(sys.argv) >= 3:
    filename_out = sys.argv[2]

if len(sys.argv) >= 4:
    mode = sys.argv[3]

if len(sys.argv) >= 5:
    root = sys.argv[4]

if len(sys.argv) >= 6:
    delimiter = sys.argv[5]

if filename_in is None:
    raise Exception("No CSV-file provided")

if filename_out is None:
    raise Exception("No output provided")

if mode not in supported_modes:
    raise Exception("Mode '%s' unknown. Supported: %s" % (mode, supported_modes))


print("Reading file: %s" % filename_in)
print("Delimiter: %s" % delimiter)
print("Mode: %s" % mode)

if root is not None:
    print("Root: %s" % root)

with open(filename_in, "r") as f:
    reader = csv.reader(f, delimiter=delimiter)

    for i, line in enumerate(reader):
        if mode == "old":
            add_line(line)
        elif mode == "tarjan":
            prepare_tarjan(line)

result = None

if mode == "old":
    print("Relations created (hypo_to_hyper):")
    print(hypo_to_hyper)

    print("Write result.")

    with open(filename_out, "w+") as f:
        writer = csv.writer(f, delimiter=delimiter)
        id = 0

        for hypo in hypo_to_hyper:
            row = [id, hypo, hypo_to_hyper[hypo]]

            #print("   %s" % row)
            writer.writerow(row)

            id += 1
elif mode == "tarjan":
    result = do_tarjan()

    print("Relations created (tarjan):")
    print(result)

    print("Write result.")

    # digraph G {
    #   "Welcome" -> "To"
    #   "To" -> "Web"
    #   "To" -> "GraphViz!"
    # }

    graphviz = "digraph G {\n"

    with open(filename_out, "w+") as f:
        writer = csv.writer(f, delimiter=delimiter)
        id = 0

        for hyper in result:
            if root is not None and len(result[hyper]) == 0:
                result[hyper].append(root)

            for hypo in result[hyper]:
                if hypo == hyper:
                    print("Found hypernym %s equivalent to hyponym %s." % (hyper, hypo))
                else:
                    row = [id, hypo, hyper]

                    print("   %s" % row)
                    writer.writerow(row)

                    graphviz += "\"%s\" -> \"%s\"\n" % (hyper, hypo)

                    id += 1

    graphviz += "}"


    with open(filename_out + "_graphviz", "w+") as f:
        f.write(graphviz)

    graphviz += "}"


    print(graphviz)

    with open(filename_out + "_gephi.csv", "w+") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(["id", "source", "target"])
        id = 0

        for hyper in result:
            if root is not None and len(result[hyper]) == 0:
                result[hyper].append(root)

            for hypo in result[hyper]:
                if hypo == hyper:
                    print("Found hypernym %s equivalent to hyponym %s." % (hyper, hypo))
                else:
                    row = [id, hypo, hyper]

                    print("   %s" % row)
                    writer.writerow(row)

                    graphviz += "\"%s\" -> \"%s\"\n" % (hyper, hypo)

                    id += 1
