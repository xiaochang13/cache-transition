from collections import defaultdict
class ConceptLabel(object):
    def __init__(self, label=None):
        self.value = label
        self.alignments = []
        self.rels = []
        self.rel_map = {}

        self.tail_ids = []
        self.parent_rels = []
        self.parent_ids = []
        self.aligned = False
        self.isVar = False

    def setVarType(self, v):
        self.isVar = v

    def getRelStr(self, idx):
        assert idx in self.rel_map
        return self.rel_map[idx]

    def buildRelMap(self):
        n_rels = len(self.rels)
        for i in range(n_rels):
            curr_idx = self.tail_ids[i]
            self.rel_map[curr_idx] = self.rels[i]
        n_rels = len(self.parent_rels)
        for i in range(n_rels):
            curr_idx = self.parent_ids[i]
            self.rel_map[curr_idx] = self.parent_rels[i]

    def setValue(self, s):
        self.value = s

    def getValue(self):
        return self.value

    def addAlignment(self, word_idx):
        self.alignments.append(word_idx)

    def getArc(self, k):
        if k >= len(self.rels):
            return None
        return self.rels[k]

class AMRGraph(object):
    def __init__(self):
        self.n = 0
        self.concepts = []
        self.counter = 0
        self.widToConceptID = {}
        self.toks = None
        self.headToTail = defaultdict(set)
        self.tailToHead = defaultdict(set)

    def incomingArcs(self, v):
        if v < 0:
            return None
        concept = self.concepts[v]
        return concept.parent_rels

    def outgoingArcs(self, v):
        if v < 0:
            return None
        concept = self.concepts[v]
        return concept.rels

    def setRoot(self, v):
        self.root = v

    def count(self):
        self.counter += 1

    def isAligned(self, v):
        if v >= len(self.concepts):
            return False
        return self.concepts[v].aligned

    def initTokens(self, toks):
        self.toks = toks

    def initLemma(self, lems):
        self.lemmas = lems

    def nextConceptIDX(self):
        return self.counter

    def conceptLabel(self, idx):
        concept = self.concepts[idx]
        return concept.getValue()

    def getConcept(self, idx):
        return self.concepts[idx]

    def buildWordToConceptIDX(self):
        """
        Assume multiple-to-one alignments
        :return:
        """
        for (i, concept) in enumerate(self.concepts):
            for word_idx in concept.alignments:
                self.widToConceptID[word_idx] = i

    def __str__(self):
        ret = ""
        for (i, concept) in enumerate(self.concepts):
            ret += ("Current concept %d: %s\n" % (i, concept.getValue()))
            concept.buildRelMap()
            rel_repr = ""
            for tail_v in concept.tail_ids:
                rel_repr += concept.rel_map[tail_v]
                rel_repr += (":" + self.concepts[tail_v].getValue()+ " ")
            ret += ("Tail concepts: %s\n" % rel_repr)
        return ret

    def buildEdgeMap(self):
        for (i, concept) in enumerate(self.concepts):
            for tail_v in concept.tail_ids:
                self.headToTail[i].add(tail_v)
                self.tailToHead[tail_v].add(i)

    def addConcept(self, c):
        self.concepts.append(c)
        self.n += 1
