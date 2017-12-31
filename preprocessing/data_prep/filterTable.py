#!/usr/bin/python
import sys
weird_set = set()
for line in open('./conceptIDCounts.dict.weird.txt'):
    if line.strip():
        weird_set.add(line.strip().split()[0])

for line in open('./lemConceptIDCounts.dict.weird.txt'):
    if line.strip():
        weird_set.add(line.strip().split()[0])

output_wf = open(sys.argv[2], 'w')
for line in open(sys.argv[1]):
    if line.strip():
        fields = line.strip().split(' #### ')
        word = fields[0].strip()
        if word in weird_set:
            print word
        else:
            print >>output_wf, line.strip()
output_wf.close()
