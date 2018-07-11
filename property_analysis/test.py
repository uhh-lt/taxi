import csv
from tarjan import tarjan
list_data = []
with open('/home/rami/Documents/taxi_server/science_en.csv-relations.csv-taxo.csv-SVC_small.csv', 'rb') as f:
    reader = csv.reader(f, delimiter = '\t')
    for i, line in enumerate(reader):
        list_data.append((line[0], line[1], line[2]))

index_map = {}
index_map_inv = {}
index = 0
for element in list_data:
	if element[1] not in index_map:
		index_map[element[1]] = index
		index_map_inv[index] = element[1]
		index+=1
	if element[2] not in index_map:
		index_map[element[2]] = index
		index_map_inv[index] = element[2]
		index+=1
hypo_to_hyper = {}
for element in list_data:
	id = element[0]
	hypo = index_map[element[2]]
	hyper = index_map[element[1]]

	print("Read line with ID '%s': %s --> %s" % (id, hypo, hyper))

	if hypo not in hypo_to_hyper:
	    hypo_to_hyper[hypo] = [hyper]
	else:
	    hypo_to_hyper[hypo].append(hyper)

for key,value in index_map.iteritems():
	print key + str(value)


comp = tarjan(hypo_to_hyper)

new_nodes = set([])
for co in comp:
	if len(co) > 1 :
		print "OH NO A CIRCLE"
		print co
	else:
		new_nodes.add(index_map_inv[co[0]])

new_list_data = []

for rel in list_data:
	if rel[1] in new_nodes and rel[2] in new_nodes:
		new_list_data.append(rel)
	

print hypo_to_hyper

print len(new_list_data)
print comp
print len(comp)
print len(index_map)
  
  

