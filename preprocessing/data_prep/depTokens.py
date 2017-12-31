#!/usr/bin/python
import sys, os

sent_index = 0

output_dir = sys.argv[2]
token_wf = open(os.path.join(sys.argv[2], 'dep.token'), 'w')
pos_wf = open(os.path.join(sys.argv[2], 'dep.pos'), 'w')
toks = []
poss = []
for line in open(sys.argv[1]):
    fields = line.strip().split()
    if len(fields) < 2: #A new sent
        sent_index += 1
        print >> token_wf, (' '.join(toks))
        print >> pos_wf, (' '.join(poss))
        toks = []
        poss = []
        continue

    toks.append(fields[1].strip())
    poss.append(fields[3].strip())
    assert fields[3].strip() == fields[4].strip()

token_wf.close()
pos_wf.close()

