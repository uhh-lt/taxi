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


def all_parent_rel(element, list_data):
    parent_root = element
    remove_relations = []
    for relation in list_data:
        if relation[0] == parent_root:
            remove_relations.append(relation)
    for ele in remove_relations:
        remove_relations += all_parent_rel(ele[1], list_data)
    return remove_relations


list_data = []
with open(os.path.join(os.path.dirname(__file__), '../../', 'vocabularies', 'science_en.csv-space-relations.csv-taxo.csv-LinearSVC.csv'), 'rb') as f:
    reader = csv.reader(f, delimiter = '\t')
    for i, line in enumerate(reader):
        list_data.append((line[1], line[2]))
ROOT = "science"
all_parent_root = all_parent_rel(ROOT, list_data)
for el in all_parent_root:
    list_data.remove(el)

elements = set([])
for relation in list_data:
    elements.add(relation[0])
    elements.add(relation[1])
elements_connected = set([])
for element in elements:
    ele_root = connected_to_root(element, list_data, ROOT)
    if not ele_root[0]:
        list_data.append((ele_root[1], ROOT))

#find_cycles(ROOT, list_data)
list_all_childs(ROOT, list_data)

# with open(os.path.join(os.path.dirname(__file__), '../../', 'vocabularies', 'science_en.taxo'), 'wb') as f:
#     for element in list_data:
#         f.write(element[0] + '\t' + element[1]  + '\n')
