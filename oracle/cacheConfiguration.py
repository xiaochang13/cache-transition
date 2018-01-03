from collections import deque
from dependency import *
from AMRGraph import *
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

        self.start_word = True # Whether start processing a new word.
        self.tree = DependencyTree()

        self.cand_vertex = None

        self.last_action = None

    def setGold(self, graph_):
        self.gold = graph_

    def popBuffer(self):
        self.buffer.popleft()

    def nextBufferElem(self):
        assert len(self.buffer) > 0, "Fetch word from empty buffer."
        return self.buffer[0]

    def bufferSize(self):
        return len(self.buffer)

    # In oracle extraction, which cache index to push onto stack.
    def chooseCacheIndex(self):
        """
        Oracle extraction for PushIDX: choose a cache vertex that will
        be least recently used.
        :return: the cache index of the chosen vertex.
        """
        headToTail = self.gold.headToTail
        tailToHead = self.gold.tailToHead
        widToConceptID = self.gold.widToConceptID
        max_dist = -1
        max_idx = -1
        for cache_idx in range(self.cache_size):
            cache_concept_idx = self.cache[cache_idx][1]
            curr_dist = 1000
            tail_set = headToTail[cache_concept_idx]
            head_set = None
            if cache_concept_idx in tailToHead:
                head_set = tailToHead[cache_concept_idx]
            for (buffer_idx, word_idx) in enumerate(self.buffer):
                if word_idx in widToConceptID:
                    buffer_concept_id = widToConceptID[word_idx]
                    if buffer_concept_id in tail_set:
                        curr_dist = buffer_idx
                        break
                    if head_set is not None and buffer_concept_id in head_set:
                        curr_dist = buffer_idx
                        break
            if curr_dist > max_dist:
                max_idx = cache_idx
                max_dist = curr_dist
        return max_idx

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
        headToTail = self.gold.headToTail
        tailToHead = self.gold.tailToHead
        widToConceptID = self.gold.widToConceptID

        last_cache_word, last_cache_concept = self.cache[self.cache_size-1]
        if last_cache_concept == -1: # ($, $) at last cache position.
            return False
        tail_set = headToTail[last_cache_concept]
        head_set = None
        if last_cache_concept in tailToHead:
            head_set = tailToHead[last_cache_concept]

        for (buffer_idx, word_idx) in enumerate(self.buffer):
            if word_idx in widToConceptID:
                buffer_concept_id = widToConceptID[word_idx]
                # If there exists an arc to things in the buffer, no POP operation.
                if (buffer_concept_id in tail_set or
                        (head_set is not None and buffer_concept_id in head_set)):
                    return False

        return True

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
        return ret_set

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
