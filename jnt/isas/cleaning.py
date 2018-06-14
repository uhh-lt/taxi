import os
import csv


def connected_to_root(element, list_data, root):
    parent = element
    parent_last = None
    while parent != parent_last:
        parent_last = parent
        for relation in list_data:
            if parent == relation[0]:
                parent = relation[1]
                #print parent
                #break
    #print '\n'
    return [parent == root, parent]



def zero_out_nodes(elements):
    zero_out = set([])
    for element in elements:
        parents = [relation[1] for relation in list_data]
        if not element in parents:
            zero_out.add(element)
    return zero_out


list_data = set([])
with open(os.path.join(os.path.dirname(__file__), '../../', 'vocabularies', 'science_en.csv-space-relations.csv-taxo.csv-LinearSVC.csv'), 'rb') as f:
    reader = csv.reader(f, delimiter = '\t')
    for i, line in enumerate(reader):
        list_data.add((line[1], line[2]))
    ROOT = "science"

    elements = set([])
    for relation in list_data:
        elements.add(relation[0])
        elements.add(relation[1])
    elements_connected = set([])
    for element in elements:
        ele_root = connected_to_root(element, list_data, ROOT)
        if ele_root[0]:
            print ele_root[1]
            elements_connected.add(element)
        else:
            print ele_root[1]

    zero_out = zero_out_nodes(list_data)
    #print zero_out
    for element in zero_out:
        list_data.add((element, ROOT))
    #print list_data
