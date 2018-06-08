import csv
import numpy

list_data = set([])
with open('en_science.csv', 'rb') as f:
    reader = csv.reader(f, delimiter = '\t')
    for i, line in enumerate(reader):
        list_data.add((line[0], line[1]))
        if not (i + 1) % 100000:
            print i
            #break

with open('isas-commoncrawl.csv', 'rb') as f:
    reader = csv.reader(f, delimiter = '\t')
    for i, line in enumerate(reader):
        list_data.add((line[0], line[1]))
        if not (i + 1) % 100000:
            print i
            #break

print list_data
final_list = []
with open('science_en_taxo.csv', 'rb') as f:
    reader = csv.reader(f, delimiter = '\t')
    for i, line in enumerate(reader):
        final_list.append((line[1], line[2]))
        #print list_temp_test
# final_list = []
# for element in list_temp_test:
#     list_test_part = [tuple(element + y) for y in list_temp_test]
#     final_list += list_test_part

count = 0.0
print list(list_data)[2]
print final_list[1]
for tupel in final_list:
    if tupel in list_data:
        count+=1.0

print str(count / float(len(final_list)))
    #your_list = list(reader)
    #print len(your_list)
    #print type(your_list[0])
