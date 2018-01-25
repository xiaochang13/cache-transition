from collections import deque
from dependency import *
from AMRGraph import *
import utils
# from utils import Tokentype
class CacheConfiguration(object):
    def __init__(self, size, length):
        """
        :param size: number of elems of the fixed-sized cache.
        :param length: number of words in the buffer.
        """
        self.stack = []
        self.buffer = deque(range(length))

        # Each cache elem is a (word idx, concept idx) pair.
        self.cache = [(-1, -1) for _ in range(size)]
        self.cache_size = size

        self.candidate = None   # A register for the newly generated vertex.

        self.hypothesis = AMRGraph()  # The AMR graph being built.
        self.gold = None  # The reference AMR graph.

        self.actionSeq = []
        self.wordSeq = []
        self.lemSeq = []
        self.posSeq = []
        self.conceptSeq = []

        self.start_word = True # Whether start processing a new word.
        self.tree = DependencyTree()

        self.cand_vertex = None

        self.last_action = None
        self.pop_buff = True

    def setGold(self, graph_):
        self.gold = graph_

    def getTokenFeats(self, idx, type):
        if idx < 0:
            return type.name + ":" + utils.NULL
        prefix = type.name + ":"
        if type == utils.Tokentype.WORD:
            return prefix + self.wordSeq[idx]
        if type == utils.Tokentype.LEM:
            return prefix + self.lemSeq[idx]
        if type == utils.Tokentype.POS:
            return prefix + self.posSeq[idx]
        if type == utils.Tokentype.CONCEPT:
            return prefix + self.conceptSeq[idx]

    def getArcFeats(self, concept_idx, idx, prefix, outgoing=True):
        arc_label = self.hypothesis.getConceptArc(concept_idx, idx, outgoing)
        return prefix + arc_label

    def getNumArcFeats(self, concept_idx, prefix, outgoing=True):
        arc_num = self.hypothesis.getConceptArcNum(concept_idx, outgoing)
        return "%s%d" % (prefix, arc_num)

    def getDepDistFeats(self, idx1, idx2):
        prefix = "DepDist="
        if idx1 < 0 or idx2 < 0:
            return prefix + utils.NULL
        dep_dist = self.tree.getDepDist(idx1, idx2)
        return "%s%d" % (prefix, dep_dist)

    def getTokenDistFeats(self, idx1, idx2, upper, prefix):
        if idx1 < 0 or idx2 < 0:
            return prefix + utils.NULL
        assert idx1 < idx2, "Left token index not smaller than right"
        token_dist = idx2 - idx1
        if token_dist > upper:
            token_dist = upper
        return "%s%d" % (prefix, token_dist)

    def getTokenTypeFeatures(self, word_idx, concept_idx, feats, prefix=""):
        word_repr = prefix + self.getTokenFeats(word_idx, utils.Tokentype.WORD)
        lem_repr = prefix + self.getTokenFeats(word_idx, utils.Tokentype.LEM)
        pos_repr = prefix + self.getTokenFeats(word_idx, utils.Tokentype.POS)
        concept_repr = prefix + self.getTokenFeats(concept_idx, utils.Tokentype.CONCEPT)
        feats.append(word_repr)
        feats.append(lem_repr)
        feats.append(pos_repr)
        feats.append(concept_repr)

    def getConceptRelationFeatures(self, concept_idx, feats):
        first_concept_arc = self.getArcFeats(concept_idx, 0, "ARC=")
        second_concept_arc = self.getArcFeats(concept_idx, 1, "ARC=")
        parent_concept_arc = self.getArcFeats(concept_idx, 0, "PARC=", False)
        concept_parrel_num = self.getNumArcFeats(concept_idx, "#PARC=", False)
        feats.append(first_concept_arc)
        feats.append(second_concept_arc)
        feats.append(parent_concept_arc)
        feats.append(concept_parrel_num)

    def getCacheFeat(self, word_idx=-1, concept_idx=-1, idx=-1):
        if idx == -1:
            return ["NONE"] * utils.cache_feat_num

        feats = []
        cache_word_idx, cache_concept_idx = self.getCache(idx)

        # Candidate token features.
        self.getTokenTypeFeatures(word_idx, concept_idx, feats)

        # Cache token features.
        self.getTokenTypeFeatures(cache_word_idx, cache_concept_idx, feats)

        # Distance features
        word_dist_repr = self.getTokenDistFeats(cache_word_idx, word_idx, 4, "WordDist=")
        concept_dist_repr = self.getTokenDistFeats(cache_concept_idx, concept_idx, 4, "ConceptDist=")
        dep_dist_repr = self.getDepDistFeats(cache_word_idx, word_idx)

        # Dependency label
        dep_label_repr = "DepLabel=" + self.tree.getDepLabel(cache_word_idx, word_idx)

        feats.append(word_dist_repr)
        feats.append(concept_dist_repr)
        feats.append(dep_dist_repr)
        feats.append(dep_label_repr)

        # Get arc information for the current concept
        self.getConceptRelationFeatures(concept_idx, feats)

        self.getConceptRelationFeatures(cache_concept_idx, feats)

        assert len(feats) == utils.cache_feat_num
        return feats

    def pushIDXFeatures(self, word_idx=-1, concept_idx=-1):
        if concept_idx == -1:
            return ["NONE"] * utils.pushidx_feat_num
        feats = []

        # Candidate vertex features.
        self.getTokenTypeFeatures(word_idx, concept_idx, feats)

        # Cache vertex features.
        for cache_idx in range(self.cache_size):
            cache_word_idx, cache_concept_idx = self.cache[cache_idx]
            prefix = "cache%d_" % cache_idx
            self.getTokenTypeFeatures(cache_word_idx, cache_concept_idx, feats, prefix)

        assert len(feats) == utils.pushidx_feat_num
        return feats

    def shiftPopFeatures(self, word_idx=-1, concept_idx=-1, active=False):
        if not active:
            return ["NONE"] * utils.shiftpop_feat_num
        feats = []
        rst_word_idx, rst_concept_idx = self.cache[self.cache_size-1]
        # Right most cache token features
        self.getTokenTypeFeatures(rst_word_idx, rst_concept_idx, feats, "rst_")

        # Buffer token features
        self.getTokenTypeFeatures(word_idx, concept_idx, feats, "buf_")

        # Then get the dependency links to right words
        dep_list = self.bufferDepConnections(rst_word_idx)

        dep_num = len(dep_list)
        if dep_num > 4:
            dep_num = 4
        dep_num_repr = "depnum=%d" % dep_num
        feats.append(dep_num_repr)
        for i in range(3):
            if i >= dep_num:
                feats.append("dep=" + utils.NULL)
            else:
                feats.append("dep=" + dep_list[i])

        assert len(feats) == utils.shiftpop_feat_num
        return feats

    def popBuffer(self):
        self.buffer.popleft()

    def nextBufferElem(self):
        if len(self.buffer) == 0:
            return -1
        # assert len(self.buffer) > 0, "Fetch word from empty buffer."
        return self.buffer[0]

    def bufferSize(self):
        return len(self.buffer)

    def getCache(self, idx):
        if idx < 0 or idx > self.cache_size:
            return None
        return self.cache[idx]

    def moveToCache(self, elem):
        self.cache.append(elem)

    def popStack(self):
        stack_size = len(self.stack)
        if stack_size < 1:
            return None
        top_elem = self.stack.pop()
        return top_elem

    def getStack(self, idx):
        stack_size = len(self.stack)
        if idx < 0 or idx >= stack_size:
            return None
        return self.stack[idx]

    def stackSize(self):
        return len(self.stack)

    # Whether the next operation should be processing a new word
    # or a vertex from the cache.
    def needsPop(self):
        """
        Whether the next operation should be processing a new word
        or a vertex from the cache.
        :return:
        """
        last_cache_word, last_cache_concept = self.cache[self.cache_size-1]
        right_edges = self.gold.right_edges
        # print "Current last cache word %d, cache concept %d" % (last_cache_word, last_cache_concept)
        # ($, $) at last cache position.
        if last_cache_concept == -1:
            return False

        next_buffer_concept_idx = self.hypothesis.nextConceptIDX()
        num_concepts = self.gold.n
        if next_buffer_concept_idx >= num_concepts:
            return True

        if last_cache_concept not in right_edges:
            return False

        assert next_buffer_concept_idx > last_cache_concept and num_concepts > last_cache_concept



        return right_edges[last_cache_concept][-1] < next_buffer_concept_idx

    def shiftBuffer(self):
        if len(self.buffer) == 0:
            return False
        self.popBuffer()
        return True

    def rightmostCache(self):
        return self.cache[self.cache_size-1]

    def pop(self):
        stack_size = len(self.stack)
        if stack_size < 1:
            return False
        cache_idx, vertex = self.stack.pop()

        # Insert a vertex to a certain cache position.
        # Then pop the last cache vertex.
        self.cache.insert(cache_idx, vertex)
        self.cache.pop()
        return True

    def connectArc(self, cand_vertex_idx, cache_vertex_idx, direction, arc_label):
        """
        Make a directed labeled arc between a cache vertex and the candidate vertex.
        :param cand_vertex: the newly generated concept from the buffer.
        :param cache_vertex: a certain vertex in the cache.
        :param direction: the direction of the connected arc.
        :param arc_label: the label of the arc.
        :return: None
        """
        cand_c = self.hypothesis.concepts[cand_vertex_idx]
        cache_c = self.hypothesis.concepts[cache_vertex_idx]
        if direction == 0: # an L-edge, from candidate to cache.
            cand_c.tail_ids.append(cache_vertex_idx)
            cand_c.rels.append(arc_label)
            cache_c.parent_ids.append(cand_vertex_idx)
            cache_c.parent_rels.append(arc_label)
        else:
            cand_c.parent_ids.append(cache_vertex_idx)
            cand_c.parent_rels.append(arc_label)
            cache_c.tail_ids.append(cand_vertex_idx)
            cache_c.rels.append(arc_label)

    def bufferDepConnections(self, word_idx, thre=20):
        ret_set = set()
        start = self.tree.n
        if len(self.buffer) > 0:
            start = self.buffer[0]
        end = self.tree.n
        if end - start > thre:
            end = start + thre
        for idx in range(start, end):
            if self.tree.getHead(idx) == word_idx:
                ret_set.add("R="+self.tree.getLabel(idx))
            elif self.tree.getHead(word_idx) == idx:
                ret_set.add("L="+self.tree.getLabel(word_idx))
        return list(ret_set)

    def bufferDepConnectionNum(self, word_idx, thre=20):
        ret_set = self.bufferDepConnections(word_idx, thre)
        return len(ret_set)

    def pushToStack(self, cache_idx):
        """
        Push a certain cache vertex onto the stack.
        :param cache_idx:
        :return:
        """
        cache_word_idx, cache_concept_idx = self.cache[cache_idx]
        del self.cache[cache_idx]
        self.stack.append((cache_idx, (cache_word_idx, cache_concept_idx)))

    def __str__(self):
        ret = "Buffer: %s" % str(self.buffer)
        ret += "  Cache: %s" % str(self.cache)
        ret += "  Stack: %s" % str(self.stack)
        return ret
