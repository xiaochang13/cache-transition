#!/usr/bin/python
import sys
import time
import pickle
import os
import re
import cPickle
import random
from amr_graph import *
from amr_utils import *
import logger
import argparse
from re_utils import *
from preprocess import *
from collections import defaultdict
from entities import identify_entities
from constants import *
from date_extraction import *
from utils import *
def removeAligned(spans, aligned_toks):
    new_spans = []
    for (start, end) in spans:
        covered = set(xrange(start, end))
        if len(aligned_toks & covered) > 0:
            continue
        new_spans.append((start, end))
        aligned_toks |= covered
    return new_spans

def build_bimap(tok2frags):
    frag2map = defaultdict(set)
    index2frags = defaultdict(set)
    for index in tok2frags:
        for frag in tok2frags[index]:
            index2frags[index].add(frag)
            frag2map[frag].add(index)

    return (index2frags, frag2map)

#Here we try to make the tok to fragment mapping one to one
def rebuild_fragment_map(tok2frags):
    (index2frags, frag2map) = build_bimap(tok2frags)
    for index in tok2frags:
        if len(tok2frags[index]) > 1:
            new_frag_list = []
            min_frag = None
            min_length = 100
            for frag in tok2frags[index]:
                index_set = frag2map[frag]
                assert index in index_set
                if len(index_set) > 1:
                    if len(index_set) < min_length:
                        min_length = len(index_set)
                        min_frag = frag
                    index_set.remove(index)
                else:
                    new_frag_list.append(frag)
            if len(new_frag_list) == 0:
                assert min_frag is not None
                new_frag_list.append(min_frag)
            tok2frags[index] = new_frag_list
    return tok2frags

def mergeSpans(index_to_spans):
    new_index_to_spans = {}
    for index in index_to_spans:
        span_list = index_to_spans[index]
        span_list = sorted(span_list, key=lambda x:x[1])
        new_span_list = []
        curr_start = None
        curr_end = None

        for (idx, (start, end, _)) in enumerate(span_list):
            if curr_end is not None:
                #assert start >= curr_end, span_list
                if start < curr_end:
                    continue
                if start > curr_end: #There is a gap in between
                    new_span_list.append((curr_start, curr_end, None))
                    curr_start = start
                    curr_end = end
                else: #They equal, so update the end
                    curr_end = end

            else:
                curr_start = start
                curr_end = end

            if idx + 1 == len(span_list): #Have reached the last position
                new_span_list.append((curr_start, curr_end, None))
        new_index_to_spans[index] = new_span_list
    return new_index_to_spans

def collapseTokens(tok_seq, lemma_seq, pos_seq, span_to_type, isTrain=True):
    n_toks = len(tok_seq)
    collapsed = set()

    new_alignment = defaultdict(list)
    collapsed_seq = []
    collapsed_lem = []
    collapsed_pos = []
    #aligned_toks = set()
    for i in xrange(n_toks):
        if i in collapsed:
            continue
        for j in xrange(n_toks+1):
            if (i, j) in span_to_type:
                node_index, node_label, _ = span_to_type[(i, j)]
                collapsed |= set(xrange(i, j))
                curr_index = len(collapsed_seq)
                new_alignment[node_index].append((curr_index, curr_index+1, None))
                if 'NE_' in node_label or 'DATE' in node_label or 'NUMBER' in node_label or node_label == "NE":
                    if "NE_" in node_label:
                        rd = random.random()
                        if rd >= 0.9 and isTrain:
                            node_label = "NE"
                    collapsed_seq.append(node_label)
                    collapsed_lem.append(node_label)
                    if 'NE' in node_label:
                        collapsed_pos.append('NE')
                    elif 'DATE' in node_label:
                        collapsed_pos.append('DATE')
                    else:
                        collapsed_pos.append('NUMBER')
                elif node_label == 'PHRASE':
                    collapsed_seq.append('@'.join(lemma_seq[i:j]).lower())
                    collapsed_lem.append('@'.join(lemma_seq[i:j]).lower())
                    collapsed_pos.append('PHRASE')
                elif 'VB' in node_label:
                    collapsed_seq.append(node_label)
                    collapsed_lem.append(node_label)
                    collapsed_pos.append('VB')
                else:

                    collapsed_seq.append(tok_seq[j-1].lower())
                    collapsed_lem.append(lemma_seq[j-1].lower())
                    collapsed_pos.append(pos_seq[j-1])

        if i not in collapsed:
            if "LRB" in tok_seq[i]:
                tok_seq[i] = '('
                lemma_seq[i] = '('
            elif "RRB" in tok_seq[i]:
                tok_seq[i] = ')'
                lemma_seq[i] = ')'
            collapsed_seq.append(tok_seq[i].lower())
            collapsed_lem.append(lemma_seq[i].lower())
            collapsed_pos.append(pos_seq[i])

    return collapsed_seq, collapsed_lem, collapsed_pos, new_alignment

def getDateAttr(frag):
    date_relations = set(['time', 'year', 'month', 'day', 'weekday', 'century', 'era', 'decade', 'dayperiod', 'season', 'timezone'])
    root_index = frag.root
    amr_graph = frag.graph
    root_node = amr_graph.nodes[root_index]

    index_to_attr = {}

    attr_indices = set()
    for edge_index in root_node.v_edges:
        curr_edge = amr_graph.edges[edge_index]
        if curr_edge.label in date_relations:
            attr_indices.add(curr_edge.tail)
            index_to_attr[curr_edge.tail] = curr_edge.label.upper()
    return (attr_indices, index_to_attr)

#Given an alignment and the fragment, output the covered span
def getSpanSide(toks, alignments, frag, unaligned_toks):
    aligned_set = set()
    amr_graph = frag.graph

    covered_set = set()
    all_date_attrs, index_to_attr = getDateAttr(frag)

    index_to_toks = defaultdict(list)

    for curr_align in reversed(alignments):
        curr_tok = curr_align.split('-')[0]
        curr_frag = curr_align.split('-')[1]

        start = int(curr_tok)
        end = start + 1

        aligned_set.add(start)

        (index_type, index) = amr_graph.get_concept_relation(curr_frag)
        if index_type == 'c':
            if frag.nodes[index] == 1: #Covered current
                covered_set.add(start)
                index_to_toks[index].append(start)
                if index in all_date_attrs:
                    all_date_attrs.remove(index)

        else: #An edge covered span
            if frag.edges[index] == 1:
                covered_set.add(start)

    covered_toks = sorted(list(covered_set))
    non_covered = [amr_graph.nodes[index].node_str() for index in all_date_attrs]
    return covered_toks, non_covered, index_to_toks

def extractNodeMapping(alignments, amr_graph):
    aligned_set = set()

    node_to_span = defaultdict(list)
    edge_to_span = defaultdict(list)

    num_nodes = len(amr_graph.nodes)
    num_edges = len(amr_graph.edges)

    op_toks = []
    role_toks = []
    for curr_align in reversed(alignments):
        curr_tok = curr_align.split('-')[0]
        curr_frag = curr_align.split('-')[1]

        start = int(curr_tok)
        end = start + 1

        aligned_set.add(start)

        (index_type, index) = amr_graph.get_concept_relation(curr_frag)
        if index_type == 'c':
            node_to_span[index].append((start, end, None))
            curr_node = amr_graph.nodes[index]

            #Extract ops for entities
            if len(curr_node.p_edges) == 1:
                par_edge = amr_graph.edges[curr_node.p_edges[0]]
                if 'op' == par_edge.label[:2]:
                    op_toks.append((start, curr_node.c_edge))

            if curr_node.is_named_entity():
                role_toks.append((start, curr_node.c_edge))

        else:
            edge_to_span[index].append((start, end, None))
    new_node_to_span = mergeSpans(node_to_span)
    new_edge_to_span = mergeSpans(edge_to_span)

    return (op_toks, role_toks, new_node_to_span, new_edge_to_span, aligned_set)

def extract_fragments(alignments, amr_graph):
    tok2frags = defaultdict(list)

    num_nodes = len(amr_graph.nodes)
    num_edges = len(amr_graph.edges)

    op_toks = []
    role_toks = []
    for curr_align in reversed(alignments):
        curr_tok = curr_align.split('-')[0]
        curr_frag = curr_align.split('-')[1]

        start = int(curr_tok)
        end = start + 1

        (index_type, index) = amr_graph.get_concept_relation(curr_frag)
        frag = AMRFragment(num_edges, num_nodes, amr_graph)
        if index_type == 'c':
            frag.set_root(index)
            curr_node = amr_graph.nodes[index]

            #Extract ops for entities
            if len(curr_node.p_edges) == 1:
                par_edge = amr_graph.edges[curr_node.p_edges[0]]
                if 'op' == par_edge.label[:2]:
                    op_toks.append((start, curr_node.c_edge))

            if curr_node.is_named_entity():
                role_toks.append((start, curr_node.c_edge))

            frag.set_edge(curr_node.c_edge)

        else:
            frag.set_edge(index)
            curr_edge = amr_graph.edges[index]
            frag.set_root(curr_edge.head)
            frag.set_node(curr_edge.tail)

        frag.build_ext_list()
        frag.build_ext_set()

        tok2frags[start].append(frag)

    for index in tok2frags:
        if len(tok2frags[index]) > 1:
            tok2frags[index] = connect_adjacent(tok2frags[index], logger)

    tok2frags = rebuild_fragment_map(tok2frags)
    for index in tok2frags:
        for frag in tok2frags[index]:
            frag.set_span(index, index+1)

    return (op_toks, role_toks, tok2frags)

#Verify this fragment contains only one edge and return it
def unique_edge(frag):
    #assert frag.edges.count() == 1, 'Not unify edge fragment found'
    amr_graph = frag.graph
    edge_list = []
    n_edges = len(frag.edges)
    for i in xrange(n_edges):
        if frag.edges[i] == 1:
            edge_list.append(i)
    assert len(edge_list) == frag.edges.count()
    return tuple(edge_list)

#For the unaligned concepts in the AMR graph
def unalignedOutTails(amr, all_alignments):
    tail_set = defaultdict(set)
    index_set = set()
    stack = [amr.root]
    visited = set()

    edge_list = []
    edge_map = defaultdict(set)

    while stack:
        curr_node_index = stack.pop()

        curr_node = amr.nodes[curr_node_index]
        curr_var = curr_node.node_str()

        if curr_node_index in visited: #A reentrancy found
            continue

        index_set.add(curr_node_index)
        visited.add(curr_node_index)
        unaligned = False
        if curr_node_index in all_alignments:
            exclude_rels, _, _ = amr.get_symbol(curr_node_index, {}, 0, 0)
        else:
            exclude_rels = []
            unaligned = True
            _ = tail_set[curr_node_index]

        assert len(exclude_rels) == 0

        for edge_index in reversed(curr_node.v_edges):
            curr_edge = amr.edges[edge_index]
            child_index = curr_edge.tail
            stack.append(child_index)
            if unaligned and child_index != curr_node_index: #Avoid self edge
                tail_set[curr_node_index].add(child_index)

            edge_map[curr_node_index].add(child_index)
            edge_map[child_index].add(curr_node_index)
            edge_list.append((curr_node_index, child_index))
    return index_set, tail_set, edge_map, edge_list

def visitUnaligned(mapped_set, amr, index):
    assert index in mapped_set
    stack = [index]
    visited_seq = []
    visited = set()
    while stack:
        to_remove = set()
        curr_index = stack.pop()
        if curr_index in visited:
            continue
        visited.add(curr_index)

        del mapped_set[curr_index]
        visited_seq.append(curr_index)

        curr_node = amr.nodes[index]
        for edge_index in curr_node.p_edges:
            parent_edge = amr.edges[edge_index]
            head_index = parent_edge.head
            if head_index in mapped_set:
                stack.append(head_index)
    return mapped_set, visited_seq

def removeLeaves(mapped_set, amr):
    updated = True
    leaf_seq = []
    while updated and len(mapped_set) > 0:
        to_remove = set()
        for index in mapped_set:
            if len(mapped_set[index]) == 0: #Leaf found
                to_remove.add(index)
        if len(to_remove) == 0:
            updated = False
        else:
            for index in to_remove:
                leaf_seq.append(index)
                del mapped_set[index]
                curr_node = amr.nodes[index]
                for edge_index in curr_node.p_edges:
                    parent_edge = amr.edges[edge_index]
                    head_index = parent_edge.head
                    if head_index in mapped_set and index in mapped_set[head_index]:
                        mapped_set[head_index].remove(index)
    return mapped_set, leaf_seq


def buildPiSeq(amr, tok_seq, all_alignments, sorted_indexes, print_info=False):

    index_set, tail_set, edge_map, edge_list = unalignedOutTails(amr, all_alignments)

    vertex_set = set()

    pi_seq = []
    visited = set()

    for index in sorted_indexes:
        if index in visited:
            continue

        visited.add(index)
        pi_seq.append(index)
        curr_node = amr.nodes[index]
        for edge_index in curr_node.p_edges:
            parent_edge = amr.edges[edge_index]
            head_index = parent_edge.head
            if head_index in tail_set:
                assert index in tail_set[head_index]
                #tail_set[head_index].remove(index)
                #tail_set, leaf_seq = removeLeaves(tail_set, amr)
                tail_set, leaf_seq = visitUnaligned(tail_set, amr, head_index)
                if leaf_seq:
                    pi_seq.extend(leaf_seq)
                    visited |= set(leaf_seq)

    for index in index_set:
        if index not in visited:
            pi_seq.append(index)

    assert len(pi_seq) == len(index_set)

    return pi_seq, edge_map, edge_list

class AMR_stats(object):
    def __init__(self):
        self.num_reentrancy = 0
        self.num_predicates = defaultdict(int)
        self.num_nonpredicate_vals = defaultdict(int)
        self.num_consts = defaultdict(int)
        self.num_named_entities = defaultdict(int)
        self.num_entities = defaultdict(int)
        self.num_relations = defaultdict(int)

    def update(self, local_re, local_pre, local_non, local_con, local_ent, local_ne):
        self.num_reentrancy += local_re
        for s in local_pre:
            self.num_predicates[s] += local_pre[s]

        for s in local_non:
            self.num_nonpredicate_vals[s] += local_non[s]

        for s in local_con:
            self.num_consts[s] += local_con[s]

        for s in local_ent:
            self.num_entities[s] += local_ent[s]

        for s in local_ne:
            self.num_named_entities[s] += local_ne[s]
        #for s in local_rel:
        #    self.num_relations[s] += local_rel[s]

    def collect_stats(self, amr_graphs):
        for amr in amr_graphs:
            (named_entity_nums, entity_nums, predicate_nums, variable_nums, const_nums, reentrancy_nums) = amr.statistics()
            self.update(reentrancy_nums, predicate_nums, variable_nums, const_nums, entity_nums, named_entity_nums)

    def dump2dir(self, dir):
        def dump_file(f, dict):
            sorted_dict = sorted(dict.items(), key=lambda k:(-k[1], k[0]))
            for (item, count) in sorted_dict:
                print >>f, '%s %d' % (item, count)
            f.close()

        pred_f = open(os.path.join(dir, 'pred'), 'w')
        non_pred_f = open(os.path.join(dir, 'non_pred_val'), 'w')
        const_f = open(os.path.join(dir, 'const'), 'w')
        entity_f = open(os.path.join(dir, 'entities'), 'w')
        named_entity_f = open(os.path.join(dir, 'named_entities'), 'w')
        #relation_f = open(os.path.join(dir, 'relations'), 'w')

        dump_file(pred_f, self.num_predicates)
        dump_file(non_pred_f, self.num_nonpredicate_vals)
        dump_file(const_f, self.num_consts)
        dump_file(entity_f, self.num_entities)
        dump_file(named_entity_f, self.num_named_entities)
        #dump_file(relation_f, self.num_relations)

    def loadFromDir(self, dir):
        def load_file(f, dict):
            for line in f:
                item = line.strip().split(' ')[0]
                count = int(line.strip().split(' ')[1])
                dict[item] = count
            f.close()

        pred_f = open(os.path.join(dir, 'pred'), 'r')
        non_pred_f = open(os.path.join(dir, 'non_pred_val'), 'r')
        const_f = open(os.path.join(dir, 'const'), 'r')
        entity_f = open(os.path.join(dir, 'entities'), 'r')
        named_entity_f = open(os.path.join(dir, 'named_entities'), 'r')

        load_file(pred_f, self.num_predicates)
        load_file(non_pred_f, self.num_nonpredicate_vals)
        load_file(const_f, self.num_consts)
        load_file(entity_f, self.num_entities)
        load_file(named_entity_f, self.num_named_entities)

    def __str__(self):
        s = ''
        s += 'Total number of reentrancies: %d\n' % self.num_reentrancy
        s += 'Total number of predicates: %d\n' % len(self.num_predicates)
        s += 'Total number of non predicates variables: %d\n' % len(self.num_nonpredicate_vals)
        s += 'Total number of constants: %d\n' % len(self.num_consts)
        s += 'Total number of entities: %d\n' % len(self.num_entities)
        s += 'Total number of named entities: %d\n' % len(self.num_named_entities)

        return s

def similarTok(curr_var, tok):
    if curr_var == tok:
        return True
    var_len = len(curr_var)
    tok_len = len(tok)
    if var_len > 3 and tok_len > 3 and tok[:4] == curr_var[:4]:
        return True
    if isNum(tok) and tok in curr_var:
        return True
    if isNum(curr_var) and curr_var in tok:
        return True

def getMapSeq(amr, tok_seq, all_alignments, unaligned, verb_map, pred_freq_thre=50, var_freq_thre=50):
    old_depth = -1
    depth = -1
    stack = [(amr.root, TOP, None, 0)] #Start from the root of the AMR
    aux_stack = []
    seq = []

    cate_tok_seq = []
    ret_index = 0

    seq_map = {}   #Map each span to a category
    visited = set()

    covered = set()
    multiple_covered = set()

    node_to_label = {}
    cate_to_index = {}

    while stack:
        old_depth = depth
        curr_node_index, rel, parent, depth = stack.pop()
        curr_node = amr.nodes[curr_node_index]
        curr_var = curr_node.node_str()

        i = 0

        while old_depth - depth >= i:
            if aux_stack == []:
                import pdb
                pdb.set_trace()
            seq.append(aux_stack.pop())
            i+=1

        if curr_node_index in visited: #A reentrancy found
            seq.append((rel+LBR, None))
            seq.append((RET + ('-%d' % ret_index), None))
            ret_index += 1
            aux_stack.append((RBR+rel, None))
            continue

        visited.add(curr_node_index)
        seq.append((rel+LBR, None))

        if curr_node_index in all_alignments:
            exclude_rels, cur_symbol, categorized = amr.get_symbol(curr_node_index, verb_map, pred_freq_thre, var_freq_thre)
        else:
            freq = amr.getFreq(curr_node_index)
            retrieved = False
            if freq and freq < 100:
                #print ' '.join(tok_seq)
                #print 'unseen: %s' % curr_var
                for index in unaligned:
                    if similarTok(curr_var, tok_seq[index]):
                        print 'retrieved: %s, %s' % (curr_var, tok_seq[index])
                        all_alignments[curr_node_index].append((index, index+1, None))

                        exclude_rels, cur_symbol, categorized = amr.get_symbol(curr_node_index, verb_map, pred_freq_thre, var_freq_thre)
                        retrieved = True
                        break

            if not retrieved:
                exclude_rels, cur_symbol, categorized = [], curr_var, False
                print 'unseen: %s' % curr_var

        if categorized:
            node_to_label[curr_node_index] = cur_symbol

        seq.append((cur_symbol, curr_node_index))
        aux_stack.append((RBR+rel, None))

        for edge_index in reversed(curr_node.v_edges):
            curr_edge = amr.edges[edge_index]
            child_index = curr_edge.tail
            if curr_edge.label in exclude_rels:
                visited.add(child_index)
                if 'VERBAL' in cur_symbol: #Might have other relations
                    tail_node = amr.nodes[child_index]
                    for next_edge_index in reversed(tail_node.v_edges):
                        next_edge = amr.edges[next_edge_index]
                        next_child_index = next_edge.tail
                        stack.append((next_child_index, next_edge.label, curr_var, depth+1))
                continue
            stack.append((child_index, curr_edge.label, curr_var, depth+1))

    seq.extend(aux_stack[::-1])
    cate_span_map, end_index_map, covered_toks = categorizedSpans(all_alignments, node_to_label)

    map_seq = []
    nodeindex_to_tokindex = {}  #The mapping
    label_to_index = defaultdict(int)

    for tok_index, tok in enumerate(tok_seq):
        if tok_index not in covered_toks:
            cate_tok_seq.append(tok)
            align_str = '%d-%d++%s++NONE++NONE++NONE' % (tok_index, tok_index+1, tok)
            map_seq.append(align_str)
            continue

        if tok_index in end_index_map: #This span can be mapped to category
            end_index = end_index_map[tok_index]
            assert (tok_index, end_index) in cate_span_map

            node_index, aligned_label, wiki_label = cate_span_map[(tok_index, end_index)]

            if node_index not in nodeindex_to_tokindex:
                nodeindex_to_tokindex[node_index] = defaultdict(int)

            indexed_aligned_label = '%s-%d' % (aligned_label, label_to_index[aligned_label])
            nodeindex_to_tokindex[node_index] = indexed_aligned_label
            label_to_index[aligned_label] += 1

            cate_tok_seq.append(aligned_label)

            align_str = '%d-%d++%s++%s++%s++%s' % (tok_index, end_index, ' '.join(tok_seq[tok_index:end_index]), wiki_label if wiki_label is not None else 'NONE', amr.nodes[node_index].node_str(), aligned_label)
            map_seq.append(align_str)

    #seq = [nodeindex_to_tokindex[node_index] if node_index in nodeindex_to_tokindex else label for (label, node_index) in seq]

    return map_seq
#Traverse AMR from top down, also categorize the sequence in case of alignment existed
def categorizeParallelSequences(amr, tok_seq, all_alignments, unaligned, verb_map, pred_freq_thre=50, var_freq_thre=50):

    old_depth = -1
    depth = -1
    stack = [(amr.root, TOP, None, 0)] #Start from the root of the AMR
    aux_stack = []
    seq = []

    cate_tok_seq = []
    ret_index = 0

    seq_map = {}   #Map each span to a category
    visited = set()

    covered = set()
    multiple_covered = set()

    node_to_label = {}
    cate_to_index = {}

    while stack:
        old_depth = depth
        curr_node_index, rel, parent, depth = stack.pop()
        curr_node = amr.nodes[curr_node_index]
        curr_var = curr_node.node_str()

        i = 0

        while old_depth - depth >= i:
            if aux_stack == []:
                import pdb
                pdb.set_trace()
            seq.append(aux_stack.pop())
            i+=1

        if curr_node_index in visited: #A reentrancy found
            seq.append((rel+LBR, None))
            seq.append((RET + ('-%d' % ret_index), None))
            ret_index += 1
            aux_stack.append((RBR+rel, None))
            continue

        visited.add(curr_node_index)
        seq.append((rel+LBR, None))

        if curr_node_index in all_alignments:
            exclude_rels, cur_symbol, categorized = amr.get_symbol(curr_node_index, verb_map, pred_freq_thre, var_freq_thre)
        else:
            freq = amr.getFreq(curr_node_index)
            retrieved = False
            if freq and freq < 100:
                #print ' '.join(tok_seq)
                #print 'unseen: %s' % curr_var
                for index in unaligned:
                    if similarTok(curr_var, tok_seq[index]):
                        print 'retrieved: %s, %s' % (curr_var, tok_seq[index])
                        all_alignments[curr_node_index].append((index, index+1, None))

                        exclude_rels, cur_symbol, categorized = amr.get_symbol(curr_node_index, verb_map, pred_freq_thre, var_freq_thre)
                        retrieved = True
                        break

            if not retrieved:
                exclude_rels, cur_symbol, categorized = [], curr_var, False
                print 'unseen: %s' % curr_var

        if categorized:
            node_to_label[curr_node_index] = cur_symbol

        seq.append((cur_symbol, curr_node_index))
        aux_stack.append((RBR+rel, None))

        for edge_index in reversed(curr_node.v_edges):
            curr_edge = amr.edges[edge_index]
            child_index = curr_edge.tail
            if curr_edge.label in exclude_rels:
                visited.add(child_index)
                if 'VERBAL' in cur_symbol: #Might have other relations
                    tail_node = amr.nodes[child_index]
                    for next_edge_index in reversed(tail_node.v_edges):
                        next_edge = amr.edges[next_edge_index]
                        next_child_index = next_edge.tail
                        stack.append((next_child_index, next_edge.label, curr_var, depth+1))
                continue
            stack.append((child_index, curr_edge.label, curr_var, depth+1))

    seq.extend(aux_stack[::-1])
    cate_span_map, end_index_map, covered_toks = categorizedSpans(all_alignments, node_to_label)

    map_seq = []
    nodeindex_to_tokindex = {}  #The mapping
    label_to_index = defaultdict(int)

    for tok_index, tok in enumerate(tok_seq):
        if tok_index not in covered_toks:
            cate_tok_seq.append(tok)
            align_str = '%d-%d++%s++NONE++NONE++NONE' % (tok_index, tok_index+1, tok)
            map_seq.append(align_str)
            continue

        if tok_index in end_index_map: #This span can be mapped to category
            end_index = end_index_map[tok_index]
            assert (tok_index, end_index) in cate_span_map

            node_index, aligned_label, wiki_label = cate_span_map[(tok_index, end_index)]

            if node_index not in nodeindex_to_tokindex:
                nodeindex_to_tokindex[node_index] = defaultdict(int)

            indexed_aligned_label = '%s-%d' % (aligned_label, label_to_index[aligned_label])
            nodeindex_to_tokindex[node_index] = indexed_aligned_label
            label_to_index[aligned_label] += 1

            cate_tok_seq.append(aligned_label)

            align_str = '%d-%d++%s++%s++%s++%s' % (tok_index, end_index, ' '.join(tok_seq[tok_index:end_index]), wiki_label if wiki_label is not None else 'NONE', amr.nodes[node_index].node_str(), aligned_label)
            map_seq.append(align_str)

    seq = [nodeindex_to_tokindex[node_index] if node_index in nodeindex_to_tokindex else label for (label, node_index) in seq]

    return map_seq

def categorizedSpans(all_alignments, node_to_label):
    visited = set()

    all_alignments = sorted(all_alignments.items(), key=lambda x:len(x[1]))
    span_map = {}
    end_index_map = {}

    for (node_index, aligned_spans) in all_alignments:
        if node_index in node_to_label:
            aligned_label = node_to_label[node_index]
            for (start, end, wiki_label) in aligned_spans:
                span_set = set(xrange(start, end))
                if len(span_set & visited) != 0:
                    continue

                visited |= span_set

                span_map[(start, end)] = (node_index, aligned_label, wiki_label)
                end_index_map[start] = end

    return span_map, end_index_map, visited

def linearize_amr(args):

    def loadbyfreq(file, threshold):
        curr_f = open(file, 'r')
        concept_set = set()
        for line in curr_f:
            if line.strip():
                fields = line.strip().split()
                concept = fields[0]
                curr_freq = int(fields[1])
                if curr_freq >= threshold:
                    concept_set.add(concept)
        return concept_set

    logger.file = open(os.path.join(args.run_dir, 'logger'), 'w')

    amr_file = os.path.join(args.data_dir, 'amr')
    alignment_file = os.path.join(args.data_dir, 'alignment')
    tok_file = os.path.join(args.data_dir, 'token')
    lem_file = os.path.join(args.data_dir, 'lemmatized_token')
    pos_file = os.path.join(args.data_dir, 'pos')

    amr_graphs = load_amr_graphs(amr_file)
    alignments = [line.strip().split() for line in open(alignment_file, 'r')]
    toks = [line.strip().split() for line in open(tok_file, 'r')]
    lems = [line.strip().split() for line in open(lem_file, 'r')]
    poss = [line.strip().split() for line in open(pos_file, 'r')]
    if args.realign:
        dep_toks = [line.strip().split() for line in open(args.dep_file, 'r')]

    assert len(amr_graphs) == len(alignments) and len(amr_graphs) == len(toks) and len(amr_graphs) == len(poss), '%d %d %d %d %d' % (len(amr_graphs), len(alignments), len(toks), len(poss))

    num_self_cycle = 0
    used_sents = 0

    amr_statistics = AMR_stats()

    if args.use_stats:
        amr_statistics.loadFromDir(args.stats_dir)
    else:
        os.system('mkdir -p %s' % args.stats_dir)
        amr_statistics.collect_stats(amr_graphs)
        amr_statistics.dump2dir(args.stats_dir)

    singleton_num = 0.0
    multiple_num = 0.0
    total_num = 0.0
    empty_num = 0.0

    phrases = set([line.strip().split('###')[0].strip() for line in open('phrases')])

    conll_wf = open(args.conll_file, 'w')
    output_tok = os.path.join(args.run_dir, "token")
    output_lem = os.path.join(args.run_dir, "lemma")
    output_pos = os.path.join(args.run_dir, "pos")

    tok_wf = open(output_tok, 'w')
    lemma_wf = open(output_lem, 'w')
    pos_wf = open(output_pos, 'w')

    quantity_times = defaultdict(int)
    phrase_to_concept = {}
    phrase_to_count = defaultdict(int)
    matched_verb_num = 0

    in_verblist_num = 0
    verbalization_map = {}

    mle_map = {}
    mleLemmaMap = {}

    tok2counts = {}
    lemma2counts = {}

    dep_sent_index = 0
    dep_tok_seq = None
    random.seed(0)

    conceptToOutGo = {}
    conceptToIncome = {}

    #op_concepts = set()
    #labelToRels = {}
    #labelToIncoming = {}
    concept_counts = defaultdict(int)
    relcounts = defaultdict(int)

    op_concepts = loadbyfreq('./ops_concepts.txt', 30)
    frequent_outgo = loadbyfreq('./concept_rels.txt', 100)
    frequent_income = loadbyfreq('./concept_income_rels.txt', 100)

    frequent_set = frequent_outgo & frequent_income
    print 'frequent set:', frequent_set

    for (sent_index, (tok_seq, lemma_seq, pos_seq, alignment_seq, amr)) in enumerate(zip(toks, lems, poss, alignments, amr_graphs)):

        logger.writeln('Sentence #%d' % (sent_index+1))
        logger.writeln(' '.join(tok_seq))

        amr.setStats(amr_statistics)

        #if sent_index > 3:
        #    sys.exit(1)

        if args.realign:
            dep_tok_seq = dep_toks[dep_sent_index]

        has_cycle = False
        if amr.check_self_cycle():
            num_self_cycle += 1
            has_cycle = True

        #if args.realign:

        amr.set_sentence(lemma_seq) ##Here we consider the lemmas for amr graph
        amr.set_poss(pos_seq)

        aligned_fragments = []
        reentrancies = {}  #Map multiple spans as reentrancies, keeping only one as original, others as connections

        has_multiple = False
        no_alignment = False

        aligned_set = set()

        (opt_toks, role_toks, node_to_span, edge_to_span, temp_aligned) = extractNodeMapping(alignment_seq, amr)

        temp_unaligned = set(xrange(len(pos_seq))) - temp_aligned

        aligned_toks = set()

        all_frags = []
        all_alignments = defaultdict(list)
        nodeid_to_frag = {}

        entity_toks = set()
        entity_not_align = False
        ####Extract named entities#####
        for (frag, wiki_label) in amr.extract_entities():
            root_index = frag.root
            if len(opt_toks) == 0:
                logger.writeln("No alignment for the entity found")

            (aligned_indexes, entity_spans) = all_aligned_spans(frag, opt_toks, role_toks, temp_unaligned)
            root_node = amr.nodes[frag.root]

            entity_mention_toks = root_node.namedEntityMention()

            total_num += 1.0

            if entity_spans:
                try:
                    entity_spans = removeAligned(entity_spans, aligned_toks)
                except:
                    print entity_spans
                    sys.exit(1)

            if entity_spans:

                nodeid_to_frag[root_index] = frag
                entity_spans = removeRedundant(tok_seq, entity_spans, entity_mention_toks)

                for (start, end) in entity_spans:
                    entity_toks |= set(xrange(start, end))
                    aligned_toks |= set(xrange(start, end))
                    logger.writeln(' '.join(tok_seq[start:end]))
                    all_alignments[frag.root].append((start, end, wiki_label))

                    tok_s = ' '.join(tok_seq)
                    lemma_s = ' '.join(lemma_seq)

                if len(entity_spans) == 1:
                    singleton_num += 1.0
                    logger.writeln('Single fragment')

                else:
                    multiple_num += 1.0
            else:
                entity_not_align = True
                empty_num += 1.0

        if entity_not_align:
            continue

        ####Process date entities
        date_entity_frags = amr.extract_all_dates()
        for frag in date_entity_frags:
            all_date_indices, index_to_attr = getDateAttr(frag)
            covered_toks, non_covered, index_to_toks = getSpanSide(tok_seq, alignment_seq, frag, temp_unaligned)

            covered_set = set(covered_toks)
            root_index = frag.root

            all_spans = getContinuousSpans(covered_toks, temp_unaligned, covered_set)
            all_spans = removeAligned(all_spans, aligned_toks)
            if all_spans:
                temp_spans = []
                for start, end in all_spans:
                    if start > 0 and (start-1) in temp_unaligned:
                        if tok_seq[start-1] in str(frag) and tok_seq[0] in '0123456789':
                            temp_spans.append((start-1, end))
                        else:
                            temp_spans.append((start, end))
                    else:
                        temp_spans.append((start, end))
                all_spans = temp_spans
                all_spans = removeDateRedundant(all_spans)
                for start, end in all_spans:
                    nodeid_to_frag[root_index] = frag
                    all_alignments[frag.root].append((start, end, None))
                    aligned_toks |= set(xrange(start, end))
                    entity_toks |= set(xrange(start, end))
            else:
                for index in temp_unaligned:
                    curr_tok = tok_seq[index]
                    found = False
                    for un_tok in non_covered:
                        if curr_tok[0] in '0123456789' and curr_tok in un_tok:
                            #print 'recovered: %s' % curr_tok
                            found = True
                            break
                    if found:
                        nodeid_to_frag[root_index] = frag
                        all_alignments[frag.root].append((index, index+1, None))
                        aligned_toks.add(index)
                        entity_toks.add(index)
                        #print 'Date: %s' % tok_seq[index]

        #Verbalization list
        tok2tuples = defaultdict(set)
        verb_map = defaultdict(set)

        matched_tuples = set()

        for (index, curr_tok) in enumerate(tok_seq):
            curr_lem = lemma_seq[index]
            if not curr_tok in VERB_LIST:
                curr_tok = lemma_seq[index]
            if curr_tok in VERB_LIST:
                in_verblist_num += 1

                matched = False
                for subgraph in VERB_LIST[curr_tok]:

                    matched_frags = amr.matchSubgraph(subgraph)
                    if matched_frags:
                        matched = True
                        matched_verb_num += 1
                        for frag_tuples in matched_frags:
                            valid = True
                            for (head, rel, tail) in frag_tuples:
                                if (head, rel, tail) in matched_tuples:
                                    valid = False
                                    break
                                matched_tuples.add((head, rel, tail))
                            if valid:
                                for (head, rel, tail) in frag_tuples:
                                    tok2tuples[index].add((head, rel, tail))
                                    verb_map[head].add((head, rel, tail))
                                    all_alignments[head].append((index, index+1, None))
                if curr_lem not in verbalization_map:
                    verbalization_map[curr_lem] = defaultdict(int)
                if matched:
                    verbalization_map[curr_lem]["y"] += 1
                else:
                    verbalization_map[curr_lem]["n"] += 1

        has_wrong_align = False
        for node_index in node_to_span:
            if node_index in all_alignments:
                continue

            curr_node = amr.nodes[node_index]
            spans = [(x,y) for (x, y,_) in node_to_span[node_index]]
            all_spans = removeAligned(spans, aligned_toks)

            if all_spans:
                node_repr = amr.nodes[node_index].node_str()

                for start, end in all_spans:

                    aligned_toks |= set(xrange(start, end))
                    all_alignments[node_index].append((start, end, None))
                    if end -start - len(node_repr.split('-')) > 1:
                        has_wrong_align = True
                        print '_'.join(tok_seq[start:end]), ":", node_repr
                        break

                    if end - start > 1:
                        phrase_repr = '@'.join(lemma_seq[start:end])

                        if isNumber(node_repr):
                            template = []
                            for k in xrange(start, end):
                                if not isNumber(lemma_seq[k]):
                                    template.append(lemma_seq[k])
                                    #quantity_times[tok_seq[k]] += 1
                                else:
                                    template.append("NUM")
                            quantity_times[' '.join(template)] += 1
                        else:
                            phrase_to_count[phrase_repr] += 1

                            if not phrase_repr in phrase_to_concept:
                                phrase_to_concept[phrase_repr] = defaultdict(int)
                            phrase_to_concept[phrase_repr][node_repr] += 1

        if has_wrong_align:
            continue

        ##Based on the alignment from node index to spans in the string
        unaligned_toks = set(xrange(len(pos_seq))) - aligned_toks

        assert len(tok_seq) == len(pos_seq)

        try:
            new_amr, _, span_to_type = AMRGraph.collapsedAmr(amr, all_alignments, tok_seq, lemma_seq, pos_seq, unaligned_toks, verb_map, phrases, nodeid_to_frag, 0, 0)   #Do not use any kind of trade off
        except:
            print 'problematic case here:', sent_index
            continue

        new_amr.update_stats(conceptToOutGo, conceptToIncome, frequent_set, concept_counts, relcounts)
        for (start, end) in span_to_type:
            (node_index, curr_symbol, wiki_label) = span_to_type[(start, end)]

            tok_s = ' '.join(tok_seq[start:end])
            lemma_s = ' '.join(lemma_seq[start:end])
            if not tok_s in tok2counts:
                tok2counts[tok_s] = defaultdict(int)

            tok2counts[tok_s][(curr_symbol, wiki_label)] += 1
            if not ((curr_symbol == "NE") or ("NE_" in curr_symbol) or (curr_symbol == "DATE")):
                if not lemma_s in lemma2counts:
                    lemma2counts[lemma_s] = defaultdict(int)

                lemma2counts[lemma_s][(curr_symbol, wiki_label)] += 1

        for index in unaligned_toks:
            tok_s = tok_seq[index]
            lemma_s = lemma_seq[index]
            if not tok_s in tok2counts:
                tok2counts[tok_s] = defaultdict(int)
            if not lemma_s in lemma2counts:
                lemma2counts[lemma_s] = defaultdict(int)
            tok2counts[tok_s][("NONE", "NONE")] += 1
            lemma2counts[lemma_s][("NONE", "NONE")] += 1

        if entity_not_align:
            continue

        dep_sent_index += 1
        collapsed_toks, collapsed_lem, collapsed_pos, new_alignment = collapseTokens(tok_seq, lemma_seq, pos_seq, span_to_type, True)
        new_amr.set_sentence(collapsed_lem)
        #new_amr.set_sentence(collapsed_toks)
        #new_amr.set_to(collapsed_toks)
        new_amr.set_poss(collapsed_pos)
        new_amr.setStats(amr_statistics)

        if args.realign:
            new_alignment = mergeToks(collapsed_toks, dep_tok_seq, new_alignment, sent_index)
            if new_alignment is None:
                continue

        span_indexes = []
        for node_index in new_alignment:
            for (start, end, _) in new_alignment[node_index]:
                span_indexes.append((start, end, node_index))
        sorted_indexes = sorted(span_indexes, key=lambda x: (x[0], x[1]))

        print_info = False
        sorted_indexes = [z for (x, y, z) in sorted_indexes]

        print_info = False

        if args.realign:
            pi_seq, edge_map, edge_list = buildPiSeq(new_amr, dep_tok_seq, new_alignment, sorted_indexes, print_info)
        else:
            pi_seq, edge_map, edge_list = buildPiSeq(new_amr, collapsed_toks, new_alignment, sorted_indexes, print_info)

        root_num = 0

        print >> conll_wf, 'sentence %d' % sent_index
        print >> tok_wf, (" ".join(collapsed_toks))
        print >> lemma_wf, (" ".join(collapsed_lem))
        print >> pos_wf, (" ".join(collapsed_pos))
        print str(new_amr)

        origToNew = {}
        for (i, index) in enumerate(pi_seq):
            assert index not in origToNew
            origToNew[index] = i

        #Output the graph file in a conll format
        for (i, index) in enumerate(pi_seq):
            line_reps = []
            curr_node = new_amr.nodes[index]
            line_reps.append(str(i))
            word_indices = []
            if index in new_alignment:
                for (start, end, _) in new_alignment[index]:
                    for tok_id in xrange(start, end):
                        word_indices.append(tok_id)
            var_bit = '1' if curr_node.is_var_node() else '0'
            line_reps.append(var_bit)
            concept_repr = new_amr.nodes[index].node_str()
            line_reps.append(concept_repr)
            word_repr = '#'.join([str(tok_id) for tok_id in word_indices]) if word_indices else 'NONE'
            line_reps.append(word_repr)
            child_triples = curr_node.childTriples()
            child_repr = '#'.join(['%s:%d' % (label, origToNew[tail_index]) for (label, tail_index) \
                    in child_triples]) if child_triples else 'NONE'
            line_reps.append(child_repr)
            parent_triples = curr_node.parentTriples()
            parent_repr = '#'.join(['%s:%d' % (label, origToNew[head_index]) for (label, head_index) \
                    in parent_triples]) if parent_triples else 'NONE'
            line_reps.append(parent_repr)
            print >> conll_wf, (' '.join(line_reps))
        print >> conll_wf, ''

    outgo_f = open('outgoing_edges.txt', 'w')
    for concept in conceptToOutGo:
        rels = sorted(conceptToOutGo[concept].items(), key=lambda x: -x[1])
        rel_str = ';'.join(['%s:%d' % (l, c) for (l, c) in rels])
        print >>outgo_f, ('%s  %s' % (concept, rel_str))
    outgo_f.close()

    concepts_wf = open('concept_counts.txt', 'w')
    sorted_conceptcounts = sorted(concept_counts.items(), key=lambda x: -x[1])
    for (concept, count) in sorted_conceptcounts:
        print >>concepts_wf, ("%s %d" % (concept, count))
    concepts_wf.close()

    relation_wf = open('relation_counts.txt', 'w')
    sorted_relcounts = sorted(relcounts.items(), key=lambda x: -x[1])
    for (rel, count) in sorted_relcounts:
        print >>relation_wf, ("%s %d" % (rel, count))
    relation_wf.close()

    #or_f = open('ops_concepts.txt', 'w')
    #sorted_opcounts = sorted(op_concepts.items(), key=lambda x: -x[1])
    #for concept, count in sorted_opcounts:
    #    if count >= 30:
    #        print >> or_f, ('%s %s' % (concept, count))
    #or_f.close()
    #conceptFreq = defaultdict(int)
    #for concept in labelToRels:
    #    for (rel, count) in labelToRels[concept].items():
    #        conceptFreq[concept] += count

    #sorted_freq = sorted(conceptFreq.items(), key=lambda x: -x[1])
    #concept_rels_wf = open('concept_rels.txt', 'w')
    #for (concept, count) in sorted_freq:
    #    rels = sorted(labelToRels[concept].items(), key=lambda x:-x[1])
    #    rel_str = ';'.join(['%s:%d' % (l, c) for (l, c) in rels])
    #    print >>concept_rels_wf, ("%s %d %s" % (concept, count, rel_str))
    #concept_rels_wf.close()

    #incomeFreq = defaultdict(int)
    #for concept in labelToIncoming:
    #    for (rel, count) in labelToIncoming[concept].items():
    #        incomeFreq[concept] += count

    #sorted_freq = sorted(incomeFreq.items(), key=lambda x: -x[1])
    #concept_rels_wf = open('concept_income_rels.txt', 'w')
    #for (concept, count) in sorted_freq:
    #    rels = sorted(labelToIncoming[concept].items(), key=lambda x:-x[1])
    #    rel_str = ';'.join(['%s:%d' % (l, c) for (l, c) in rels])
    #    print >>concept_rels_wf, ("%s %d %s" % (concept, count, rel_str))
    #concept_rels_wf.close()

    income_f = open('incoming_edges.txt', 'w')
    for concept in conceptToIncome:
        rels = sorted(conceptToIncome[concept].items(), key=lambda x: -x[1])
        rel_str = ';'.join(['%s:%d' % (l, c) for (l, c) in rels])
        print >>income_f, ('%s  %s' % (concept, rel_str))
    income_f.close()

    quant_wf = open('quantities', 'w')
    sorted_quantities = sorted(quantity_times.items(), key=lambda x: -x[1])
    for (quantity_term, count) in sorted_quantities:
        print>> quant_wf, quantity_term, count
    quant_wf.close()

    print 'total matched verbalization', matched_verb_num
    print 'total in verblist num', in_verblist_num

    phrase_wf = open('phrase_table', 'w')
    for phrase in phrase_to_concept:
        sorted_items = sorted(phrase_to_concept[phrase].items(), key=lambda x: -x[1])
        if sorted_items[0][1] > 3:
            print>> phrase_wf, phrase,  '###', sorted_items[0][0], '###', sorted_items[0][1]
    phrase_wf.close()

    conll_wf.close()
    tok_wf.close()
    lemma_wf.close()
    pos_wf.close()

    for tok_s in tok2counts:
        sorted_map_counts = sorted(tok2counts[tok_s].items(), key=lambda x:-x[1])
        mle_map[tok_s] = sorted_map_counts[0][0]

    print '######NUMBERS#######'
    for lemma_s in lemma2counts:
        sorted_map_counts = sorted(lemma2counts[lemma_s].items(), key=lambda x:-x[1])
        mleLemmaMap[lemma_s] = sorted_map_counts[0][0]
        if mleLemmaMap[lemma_s][0] == 'NUMBER':
            print lemma_s

    #assert 'in fact' in mleLemmaMap
    #loadMLECount('./conceptIDCounts.dict')
    #loadMLECount('./lemConceptIDCounts.dict')

    mle_map = filterNoise(mle_map, './conceptIDCounts.dict.weird.txt')
    mleLemmaMap = filterNoise(mleLemmaMap, './lemConceptIDCounts.dict.weird.txt')

    assert 'billion' not in mle_map

    dumpMap(mle_map, 'mleConceptID.dict')
    dumpMap(mleLemmaMap, 'lemmaMLEConceptID.dict')

    if not args.realign:
        linearizeData(mle_map, mleLemmaMap, phrases, args.dev_dir, args.dev_output)
        linearizeData(mle_map, mleLemmaMap, phrases, args.test_dir, args.test_output)

def filterNoise(curr_map, filter_file):
    weird_set = set()
    for line in open(filter_file):
        if line.strip():
            weird_set.add(line.strip().split()[0])

    new_map = {}
    for l in curr_map:
        if curr_map[l][0] == 'NONE' and l in weird_set and (not '@' in l):
            continue
        new_map[l] = curr_map[l]
    return new_map

def loadMLECount(file):
    mleCounts = {}
    count_f = open(file, 'r')
    weird_wf = open(('%s.weird.txt' % file), 'w')
    for line in count_f:
        if line.strip():
            fields = line.strip().split(' #### ')
            word = fields[0].strip()
            choices = fields[1].split()
            for curr in choices:
                concept = curr.split(':')[0]
                count = int(curr.split(':')[1])
                if concept == '-NULL-' and count < 20:
                    print >> weird_wf, word, count
                else:
                    if word not in mleCounts:
                        mleCounts[word] = defaultdict(int)
                    mleCounts[word][concept] = count
    weird_wf.close()
    return mleCounts



#Given the original text, generate the categorized text
def linearizeData(mle_map, mleLemmaMap, phrases, data_dir, output_dir):

    def identifyNumber(seq, mle_map):
        quantities = set(['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'billion', 'tenth', 'million', 'thousand', 'hundred', 'viii', 'eleven', 'twelve', 'thirteen', 'iv'])
        for tok in seq:
            if tok in mle_map and mle_map[tok][0] != 'NUMBER':
                #print 'not number:', tok
                return False
            elif not (isNumber(tok) or tok in quantities):
                return False
        return True

    def isCategory(cate):
        categories = set(["NUMBER", "DATE"])
        return cate in categories or "VB" in cate or "NE_" in cate

    def replaceSymbol(s):
        return s.replace("@ - @", "@-@").replace("@ :@", "@:@").replace("@ / @", "@/@")

    def allUnaligned(seq, mleLemmaMap):
        for tok in seq:
            if not tok in mleLemmaMap:
                return False
            aligned_c = mleLemmaMap[tok][0]
            if not (aligned_c.lower() == "none"):
                return False
        return True

    tok_file = os.path.join(data_dir, 'token')
    lem_file = os.path.join(data_dir, 'lemma')
    pos_file = os.path.join(data_dir, 'pos')
    ner_file = os.path.join(data_dir, 'ner')
    date_file = os.path.join(data_dir, 'date')

    toks = [line.strip().split() for line in open(tok_file, 'r')]
    lems = [line.strip().split() for line in open(lem_file, 'r')]
    poss = [line.strip().split() for line in open(pos_file, 'r')]

    all_entities, entity_map = identify_entities(tok_file, ner_file, mle_map) #tok_file or lem_file?
    all_dates = dateMap(date_file)

    tokseq_result = os.path.join(output_dir, 'token')
    lemseq_result = os.path.join(output_dir, 'lemma')
    posseq_result = os.path.join(output_dir, 'pos')
    map_file = os.path.join(output_dir, 'cate_map')

    tok_wf = open(tokseq_result, 'w')
    lem_wf = open(lemseq_result, 'w')
    pos_wf = open(posseq_result, 'w')
    map_wf = open(map_file, 'w')

    filtered_phrase = set()

    for (sent_index, (tok_seq, lem_seq, pos_seq, entities_in_sent)) in enumerate(zip(toks, lems, poss, all_entities)):
        #if sent_index != 197 and sent_index != 196:
        #    continue
        print 'snt: %d' % sent_index
        n_toks = len(tok_seq)
        aligned_set = set()

        all_spans = []
        date_spans = all_dates[sent_index]
        date_set = set()

        span_to_type = {}

        #Align dates
        for (start, end) in date_spans:
            if end - start > 1:
                new_aligned = set(xrange(start, end))
                aligned_set |= new_aligned
                entity_name = ' '.join(tok_seq[start:end])
                if entity_name in mle_map:
                    entity_typ = mle_map[entity_name]
                else:
                    entity_typ = ('DATE', "NONE")
                all_spans.append((start, end, entity_typ))
                span_to_type[(start, end)] = ("NONE", "DATE", "NONE")
                print 'Date:', start, end
            else:
                date_set.add(start)


        #First align multi tokens
        for (start, end, entity_typ) in entities_in_sent:
            print 'entity:', ' '.join(tok_seq[start:end]), entity_typ
            #if end - start > 1:
            new_aligned = set(xrange(start, end))
            if len(aligned_set & new_aligned) != 0:
                continue
            aligned_set |= new_aligned
            entity_name = ' '.join(tok_seq[start:end])
            if replaceSymbol(entity_name) in mle_map:
                new_name = replaceSymbol(entity_name)
                if new_name != entity_name:
                    entity_name = new_name
                    print "Replaced:", entity_name

            if entity_name in mle_map:
                entity_typ = mle_map[entity_name]
                if not "NE_" in entity_typ[0] and entity_name.lower() in mle_map:
                    entity_typ = mle_map[entity_name.lower()]
                    if not "NE_" in entity_typ[0]:
                        print "Removed discovered entity:", entity_name
                        continue

            elif start == 0 and entity_name.lower() in mle_map:
                entity_typ = mle_map[entity_name.lower()]
                if not "NE_" in entity_typ[0]:
                    print "Removed discovered entity:", entity_name
                    continue

            elif entity_typ == "PER":
                entity_typ = ('NE_person', '-')
            else:
                entity_typ = ('NE', '-')
            all_spans.append((start, end, entity_typ))
            span_to_type[(start, end)] = ("NONE", entity_typ[0], "NONE")

        #Align the first token, @someone
        if (0, 1) not in span_to_type:
            first_tok = tok_seq[0]
            if len(first_tok) > 2 and first_tok[0] == '@':
                per_name = first_tok[1:]
                entity_typ = ('NE_person', '-')
                aligned_set.add(0)
                span_to_type[(0, 1)] = ("NONE", entity_typ[0], "NONE")
                all_spans.append((0, 1, entity_typ))

        #Align things that are in the ner align map
        for start in xrange(n_toks):
            for end in xrange(n_toks, start, -1):
                new_aligned = set(xrange(start, end))
                if len(aligned_set & new_aligned) != 0:
                    continue

                phrase = ' '.join(tok_seq[start:end])
                if phrase in entity_map:
                    print "Discovered entity name:", phrase, entity_map[phrase]
                    entity_typ = entity_map[phrase]
                    if entity_typ == "PER":
                        entity_typ = ('NE_person', '-')
                    else:
                        entity_typ = ('NE', '-')
                    all_spans.append((start, end, entity_typ))
                    span_to_type[(start, end)] = ("NONE", entity_typ[0], "NONE")

        #Then phrase in sentence
        for start in xrange(n_toks):
            for end in xrange(n_toks, start+1, -1):
                assert end - start > 1
                new_aligned = set(xrange(start, end))
                if len(aligned_set & new_aligned) != 0:
                    continue

                lem_phrase = ' '.join(lem_seq[start:end]).lower()
                real_p = "@".join(lem_seq[start:end]).lower()

                is_phrase = False

                if lem_phrase in mleLemmaMap and real_p in phrases:

                    print 'phrase:', (' '.join(lem_seq[start:end]))
                    entity_typ = mleLemmaMap[lem_phrase]
                    #print entity_map
                    all_spans.append((start, end, entity_typ))
                    span_to_type[(start, end)] = ("NONE", entity_typ[0], "NONE")
                    is_phrase = True

                if is_phrase:
                    aligned_set |= new_aligned

        #Numbers
        for start in xrange(n_toks):
            for end in xrange(n_toks, start, -1):
                new_aligned = set(xrange(start, end))
                if len(aligned_set & new_aligned) != 0:
                    continue

                if identifyNumber(tok_seq[start:end], mle_map):
                    entity_typ = ("NUMBER", "NONE")
                    all_spans.append((start, end, entity_typ))
                    span_to_type[(start, end)] = ("NONE", entity_typ[0], "NONE")
                    aligned_set |= new_aligned

        #Single token
        for (index, curr_tok) in enumerate(tok_seq):
            if index in aligned_set:
                continue

            curr_lem = lem_seq[index]
            curr_pos = pos_seq[index]
            aligned_set.add(index)

            if curr_tok in mle_map or curr_lem in mleLemmaMap:
                if curr_tok in mle_map:
                    (category, wiki_label) = mle_map[curr_tok]
                else:
                    (category, wiki_label) = mleLemmaMap[curr_lem]
                    if "NE" == category or "NE_" in category:
                        if curr_lem != curr_tok:
                            category = "NONE"

                if isCategory(category):
                    if category.lower() == 'none':
                        all_spans.append((index, index+1, (curr_tok, "NONE")))
                    else:
                        all_spans.append((index, index+1, (category, wiki_label)))
                        span_to_type[(index, index+1)] = ("NONE", category, "NONE")
            elif curr_lem in VERB_LIST:
                subgraph = VERB_LIST[curr_lem][0]
                root_label = subgraph.keys()[0]
                if re.match('.*-[0-9]+', root_label) is not None:
                    print "VERBALIZED:", curr_tok, curr_lem, root_label
                    all_spans.append((index, index+1, ("VB_VERB", "NONE")))
                    span_to_type[(index, index+1)] = ("NONE", "VB_VERB", "NONE")
                else:
                    var_name = "VB_" + root_label
                    print "VERBALIZED:", curr_tok, curr_lem, var_name
                    all_spans.append((index, index+1, (var_name, "NONE")))
                    span_to_type[(index, index+1)] = ("NONE", var_name, "NONE")
            else:
                if curr_tok[0] in '\"\'.':
                    print 'weird token: %s, %s' % (curr_tok, curr_pos)
                    continue
                if index in date_set:
                    entity_typ = ('DATE', "NONE")
                    all_spans.append((index, index+1, entity_typ))
                    span_to_type[(index, index+1)] = ("NONE", "DATE", "NONE")

        all_spans = sorted(all_spans, key=lambda span: (span[0], span[1]))
        map_repr_seq = mapRepr(all_spans, tok_seq, lem_seq)

        print >> map_wf, '##'.join(map_repr_seq)

        collapsed_toks, collapsed_lem, collapsed_pos, _ = collapseTokens(tok_seq, lem_seq, pos_seq, span_to_type, False)

        print >> tok_wf, (" ".join(collapsed_toks))
        print >> lem_wf, (" ".join(collapsed_lem))
        print >> pos_wf, (" ".join(collapsed_pos))

    print 'filtered phrases:'
    for phrase in filtered_phrase:
        print phrase
    tok_wf.close()
    lem_wf.close()
    pos_wf.close()
    map_wf.close()

def buildLinearEnt(entity_name, ops):
    ops_strs = ['op%d( %s )op%d' % (index, s, index) for (index, s) in enumerate(ops, 1)]
    ent_repr = '%s name( name %s )name' % (entity_name, ' '.join(ops_strs))
    return ent_repr



def mapRepr(linearized_tokseq, tok_seq, lem_seq):
    map_repr = []
    for (start, end, (tok, wiki_label)) in linearized_tokseq:
        if isSpecial(tok):
            if "VB_" in tok and lem_seq[start] in VERB_LIST:
                map_repr.append('%s++%s++%s' % (tok, "@@".join(lem_seq[start:end]), wiki_label))
            else:
                map_repr.append('%s++%s++%s' % (tok, "@@".join(tok_seq[start:end]), wiki_label))
    return map_repr

#Given dev or test data, build the linearized token sequence
#Based on entity mapping from training, NER tagger
def conceptID(args):
    return

def dateMap(dateFile):
    dates_in_lines = []
    for line in open(dateFile):
        date_spans = []
        if line.strip():
            spans = line.strip().split()
            for sp in spans:
                start = int(sp.split('-')[0])
                end = int(sp.split('-')[1])
                date_spans.append((start, end))
        dates_in_lines.append(date_spans)
    return dates_in_lines

#Build the entity map for concept identification
#Choose either the most probable category or the most probable node repr
#In the current setting we only care about NE and DATE
def loadMap(map_file):
    span_to_cate = {}

    #First load all possible mappings each span has
    with open(map_file, 'r') as map_f:
        for line in map_f:
            if line.strip():
                spans = line.strip().split('##')
                for s in spans:
                    try:
                        fields = s.split('++')
                        toks = fields[1]
                        wiki_label = fields[2]
                        node_repr = fields[3]
                        category = fields[-1]
                    except:
                        print spans, line
                        print fields
                        sys.exit(1)
                    if toks not in span_to_cate:
                        span_to_cate[toks] = defaultdict(int)
                    span_to_cate[toks][(category, node_repr, wiki_label)] += 1

    mle_map = {}
    for toks in span_to_cate:
        sorted_types = sorted(span_to_cate[toks].items(), key=lambda x:-x[1])
        curr_type = sorted_types[0][0][0]
        if curr_type[:2] == 'NE' or curr_type[:4] == 'DATE':
            mle_map[toks] = sorted_types[0][0]
    return mle_map

def dumpMap(mle_map, result_file):
    with open(result_file, 'w') as wf:
        for toks in mle_map:
            print >>wf, ('%s####%s##%s' % (toks, mle_map[toks][0], mle_map[toks][1]))

#For each sentence, rebuild the map from categorized form to graph side nodes
def rebuildMap(args):
    return

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--amr_file", type=str, help="the original AMR graph files", required=False)
    argparser.add_argument("--conll_file", type=str, help="output the AMR graph in conll format", required=False)
    argparser.add_argument("--dev_dir", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--test_dir", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--dev_output", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--test_output", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--stop", type=str, help="stop words file", required=False)
    argparser.add_argument("--lemma", type=str, help="lemma file", required=False)
    argparser.add_argument("--data_dir", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--map_file", type=str, help="map file from training")
    argparser.add_argument("--date_file", type=str, help="all the date in each sentence")
    argparser.add_argument("--dep_file", type=str, help="dependency file")
    argparser.add_argument("--run_dir", type=str, help="the output directory for saving the constructed forest")
    argparser.add_argument("--use_lemma", action="store_true", help="if use lemmatized tokens")
    argparser.add_argument("--parallel", action="store_true", help="if to linearize parallel sequences")
    argparser.add_argument("--realign", action="store_true", help="if to realign the data")
    argparser.add_argument("--use_stats", action="store_true", help="if use a built-up statistics")
    argparser.add_argument("--stats_dir", type=str, help="the statistics directory")
    argparser.add_argument("--min_prd_freq", type=int, default=50, help="threshold for filtering predicates")
    argparser.add_argument("--min_var_freq", type=int, default=50, help="threshold for filtering non predicate variables")
    argparser.add_argument("--index_unknown", action="store_true", help="if to index the unknown predicates or non predicate variables")

    args = argparser.parse_args()
    linearize_amr(args)
