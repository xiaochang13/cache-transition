#!/usr/bin/python
import sys
import time
import pickle
import os
import cPickle
import hypergraph
from fragment_hypergraph import FragmentHGNode, FragmentHGEdge
from amr_graph import *
from amr_utils import *
import logger
import gflags
from HRGSample import *
from rule import Rule
import argparse
from re_utils import *
from bitarray import bitarray
from collections import defaultdict
from filter_stop_words import *
from lemma_util import initialize_lemma
from preprocess import *
from date_extraction import *
from copy import deepcopy
from treewidth import depthFirstPi, buildPiSeq

FLAGS = gflags.FLAGS

gflags.DEFINE_string(
    'fragment_nonterminal',
    'X',
    'Nonterminal used for phrase forest.')
gflags.DEFINE_bool(
    'delete_unaligned',
    False,
    'Delete unaligned words in phrase decomposition forest.')
gflags.DEFINE_bool(
    'href',
    False,
    'Delete unaligned words in phrase decomposition forest.')
gflags.DEFINE_integer(
    'max_type',
    7,
    'Set the maximum attachment nodes each nontermial can have.')

FRAGMENT_NT = '[%s]' % FLAGS.fragment_nonterminal
p_rule_f = open('poisoned_rule.gr', 'w')
q_rule_f = open('another_poisoned_rule.gr', 'w')
#unalign_f = open('unaligned_info', 'w')
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

def getDateAttr(frag):
    date_relations = set(['time', 'year', 'month', 'day', 'weekday', 'century', 'era', 'decade', 'dayperiod', 'season', 'timezone'])
    assert len(frag.roots) == 1
    root_index = frag.roots[0]
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

def collapseTokens(tok_seq, pos_seq, span_to_type):
    n_toks = len(tok_seq)
    collapsed = set()

    new_alignment = defaultdict(list)
    collapsed_seq = []
    collapsed_pos = []
    #aligned_toks = set()
    for i in xrange(n_toks):
        if i in collapsed:
            continue
        for j in xrange(n_toks+1):
            if (i, j) in span_to_type:
                node_index, node_label = span_to_type[(i, j)]
                collapsed |= set(xrange(i, j))
                curr_index = len(collapsed_seq)
                new_alignment[node_index].append((curr_index, curr_index+1, None))
                if 'NE' in node_label or 'DATE' in node_label:
                    collapsed_seq.append(node_label)
                    collapsed_pos.append(node_label)
                elif node_label.lower() != node_label:
                    assert j > i
                    curr_pos = pos_seq[j-1]
                    collapsed_seq.append(curr_pos)
                    collapsed_pos.append(node_label)
                else:
                    collapsed_seq.append(tok_seq[j-1])
                    collapsed_pos.append(pos_seq[j-1])

        if i not in collapsed:
            collapsed_seq.append(tok_seq[i])
            collapsed_pos.append(pos_seq[i])

    return collapsed_seq, collapsed_pos, new_alignment


#Extract the span to fragment alignment, don't allow one to multiple, but one to multiple
def initAlignments(amr, tok_seq, pos_seq, all_alignments, unaligned, verb_map, nodeid_to_frag,
                   pred_freq_thre=50, var_freq_thre=50, use_pos=False):

    span_to_frag = {}
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
        curr_node_index, incoming_edge_index, parent, depth = stack.pop()
        curr_node = amr.nodes[curr_node_index]
        curr_var = curr_node.node_str()

        curr_frag = None
        rel = amr.edges[incoming_edge_index].label if isinstance(incoming_edge_index, int) \
            else incoming_edge_index

        if curr_node_index in visited: #A reentrancy found
            seq.append((rel+LBR, None))
            seq.append((RET + ('-%d' % ret_index), None))
            ret_index += 1
            aux_stack.append((RBR+rel, None))
            continue

        visited.add(curr_node_index)
        seq.append((rel+LBR, None))

        if curr_node_index in all_alignments:
            curr_frag, exclude_rels, cur_symbol, categorized = amr.getFragment(
                curr_node_index, verb_map, pred_freq_thre, var_freq_thre, nodeid_to_frag)
            spans = all_alignments[curr_node_index]
            assert len(spans) > 0
            try:
                for start, end in spans:
                    span_to_frag[(start, end)] = curr_frag
            except:
                print all_alignments
                print span_to_frag
                print spans
                sys.exit(1)
        else:
            exclude_rels = []

        for edge_index in reversed(curr_node.v_edges):
            curr_edge = amr.edges[edge_index]
            child_index = curr_edge.tail
            if curr_edge.label in exclude_rels:
                visited.add(child_index)
                if cur_symbol[:2] != 'NE': #Might have other relations
                    tail_node = amr.nodes[child_index]
                    for next_edge_index in reversed(tail_node.v_edges):
                        next_edge = amr.edges[next_edge_index]
                        next_child_index = next_edge.tail
                        stack.append((next_child_index, next_edge.label, curr_var, depth+1))
                continue
            stack.append((child_index, curr_edge.label, curr_var, depth+1))

    return span_to_frag

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

#def filter_with_maxtype(curr_node):
#    root_index = curr_node.frag.root
#    ext_set = curr_node.frag.ext_set
#    nonterm_type = len(ext_set) if root_index in ext_set else (len(ext_set) + 1)
#    if nonterm_type > FLAGS.max_type:
#        curr_node.set_nosample(True)

#Enlarge a chart with a set of items
def enlarge_chart(prev_chart, new_items):
    for node1 in new_items:
        flag = False
        for node2 in prev_chart:
            if node1.frag == node2.frag:
                for edge in node1.incoming:
                    node2.add_incoming(edge)
                    flag = True

        if not flag: #node1 has never appeared in the previous chart
            prev_chart.add(node1)

#Add one item to a chart
def add_one_item(prev_chart, item):
    if len(prev_chart) == 0:
        prev_chart.add(item)
        return

    for node in prev_chart:
        assert node.frag == item.frag
        for edge in item.incoming:
            node.add_incoming(edge)
            if item.width < node.width:
                node.width = item.width

#To verify if a chart item has covered the graph
#i.e. covered all nodes and all edges
def is_goal_item(chart_item):
    fragment = chart_item.frag
    nodes = fragment.nodes
    edges = fragment.edges
    return len(edges) == edges.count()
    #return (len(nodes) == nodes.count()) and (len(edges) == edges.count())

def initialize_edge_alignment(aligned_fragments, edge_alignment):
    for frag in aligned_fragments:
        edge_alignment |= frag.edges

#This method output all the unaligned node information of the current AMR graph
def output_all_unaligned_nodes(edge_alignment, amr_graph):
    un_seq = []
    un_nodes = []
    for i in xrange(len(amr_graph.nodes)):
        curr_node = amr_graph.nodes[i]
        c_edge_index = curr_node.c_edge
        if edge_alignment[c_edge_index] == 0: #Found a concept that is not aligned
            un_seq.append(str(curr_node))
            un_nodes.append(curr_node)
    #print >> unalign_f, ' '.join(un_seq)
    return un_nodes

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

#Build one node that would cover unaligned edges going out of a node
def build_one_node(curr_frag, curr_start, curr_end):

    new_node = FragmentHGNode(FRAGMENT_NT, curr_start, curr_end, curr_frag)
    new_node.width = len(curr_frag.ext_set)

    #rule_str = new_node.lexical_rule()
    #print >>rule_f, rule_str

    return new_node

def build_bimap(tok2frags):
    frag2map = defaultdict(set)
    index2frags = defaultdict(set)
    for index in tok2frags:
        for frag in tok2frags[index]:
            index2frags[index].add(frag)
            frag2map[frag].add(index)
            #matched_list = extract_patterns(str(frag), '~e\.[0-9]+(,[0-9]+)*')
            #matched_indexes = parse_indexes(matched_list)
            #for matched_index in matched_indexes:
            #    frag2map[frag].add(matched_index)
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

def extract_fragments(s2g_alignment, amr_graph):
    alignments = s2g_alignment.strip().split()
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
                if 'op' in par_edge.label:
                    op_toks.append((start, curr_node.c_edge))

            if curr_node.is_entity():
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
            frag.setSpan(index, index+1)

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
    #root_node = amr_graph.nodes[frag.root]
    #for edge_index in root_node.v_edges:
    #    if frag.edges[edge_index] == 1:
    #        return edge_index
    #assert True, 'This is impossible'
    #return None

# extract all the binarized fragments combinations for AMR
# each chart item is a set of fragments are consistent with a span of aligned strings
def fragment_decomposition_forest(fragments, amr_graph, unaligned_fragments, capacities):
    def combine_nodes(i, j, node1, node2):
        curr_width = max(node1.width, node2.width)
        new_frag, b_size = general_combine_fragments(node1.frag, node2.frag)
        curr_width = max(curr_width, b_size)
        new_node = FragmentHGNode(FRAGMENT_NT, i, j, new_frag, False, False, False)
        new_node.width = curr_width
        edge = FragmentHGEdge()
        edge.add_tail(node1)
        edge.add_tail(node2)
        new_node.add_incoming(edge)
        return new_node

    assert len(unaligned_fragments) == 0

    n = len(fragments) #These fragments are aligned, and have some order based on the strings

    chart = [[None for j in range(n+1)] for i in range(n+1)]

    start_time = time.time()

    curr_width = 0

    #The leaves of the forest are identified concept fragments
    for i in xrange(n):
        j = i + 1

        new_node = build_one_node(fragments[i], i, j)
        #new_node.width = len(new_node.frag.ext_list)
        #chart[i][j].add(new_node)
        chart[i][j] = new_node

    #These are the unaligned concepts in the graph
    unaligned_nodes = []


    start_time = time.time()
    for span in xrange(2, n+1):
        for i in xrange(0, n):
            j = i + span
            if j > n:
                continue
            curr_time = time.time()
            #if curr_time - start_time > 30:
            #    return None

            for k in xrange(i+1, j):
                assert  chart[i][k] is not None and chart[k][j] is not None
                #if len(chart[i][k]) == 0 or len(chart[k][j]) == 0:
                #    continue
                node1 = chart[i][k]
                node2 = chart[k][j]
                #curr_time = time.time()

                new_node = combine_nodes(i, j, node1, node2)
                #print 'current:', i, j, new_node.width, node1.width, node2.width, len(node1.frag.ext_set | node2.frag.ext_set)
                if chart[i][j] is None:
                    chart[i][j] = new_node

                #print new_frag.ext_list
                #print '%d-%d' % (new_frag.start, new_frag.end)
                #print '%d-%d' % (i, j)
                #print new_frag.rootLabels()
                #print '%d-%d:%s' % (new_frag.start, new_frag.end, ' '.join(new_frag.str_list()))
                #print str(new_frag)

                #if len(new_node.frag.str_list()) < 9:
                #    rule_str = new_node.lexical_rule()
                #    print >> rule_f, rule_str

                #add_one_item(chart[i][j], new_node)
                else:
                    if new_node.width < chart[i][j].width:
                        chart[i][j].width = new_node.width
                        chart[i][j].incoming = [new_node.incoming[0]]
                        #chart[i][j].edge = len(chart[i][j].incoming)

                    assert len(new_node.incoming) == 1
                    #for edge in new_node.incoming:
                    #    chart[i][j].add_incoming(edge)



    if chart[0][n] is None:
        print '##################################'
        print 'The goal chart is empty, fail to build a goal item'
        print 'Alignment fragments:'
        for frag in fragments:
            print '%s :   %s' % (frag.str_side(), str(frag))
        print 'Unaligned fragments:'
        for frag in unaligned_fragments:
            print str(frag)
        print '#################################'
        return None

    #for node in chart[0][n]:
    #    assert len(chart[0][n]) == 1 and is_goal_item(node)
    #    #if is_goal_item(node):
    hg = hypergraph.Hypergraph(chart[0][n])
    assert is_goal_item(chart[0][n])
    return hg, chart[0][n].width

    #if hg is None:
    #    print '##################################'
    #    print 'No goal item in the final chart'
    #    print 'Alignment fragments:'
    #    for frag in fragments:
    #        print str(frag)

    #    return None
    #return hg

def removeAligned(spans, aligned_toks):
    new_spans = []
    for (start, end) in spans:
        covered = set(xrange(start, end))
        if len(aligned_toks & covered) > 0:
            continue
        new_spans.append((start, end))
        aligned_toks |= covered
    return new_spans

def construct_forest(args):
    sys.setrecursionlimit(sys.getrecursionlimit() * 30)

    logger.file = open(os.path.join(args.save_dir, 'logger'), 'w')

    singleton_num = 0.0
    multiple_num = 0.0
    total_num = 0.0

    empty_num = 0.0
    amr_file = os.path.join(args.data_dir, 'amr')
    alignment_file = os.path.join(args.data_dir, 'alignment')
    lemma_file = os.path.join(args.data_dir, 'lemmatized_token')
    tok_file = os.path.join(args.data_dir, 'token')
    pos_file = os.path.join(args.data_dir, 'pos')

    amr_graphs = load_amr_graphs(amr_file)
    alignments = [line.strip().split() for line in open(alignment_file, 'r')]
    #toks = [line.strip().split() for line in open(tok_file, 'r')]
    toks = [line.strip().split() for line in open(lemma_file, 'r')]
    lemmas = [line.strip().split() for line in open(lemma_file, 'r')]
    poss = [line.strip().split() for line in open(pos_file, 'r')]

    assert len(amr_graphs) == len(alignments) and len(amr_graphs) == len(toks) and len(amr_graphs) == len(poss), '%d %d %d %d %d' % (len(amr_graphs), len(alignments), len(toks), len(poss))

    dumped_rule_file = os.path.join(args.save_dir, 'dumped_rules')

    global rule_f

    rule_f = open(dumped_rule_file, 'w')

    num_self_cycle = 0

    amr_statistics = AMR_stats()

    if args.use_stats:
        amr_statistics.loadFromDir(args.stats_dir)
        print amr_statistics
    else:
        os.system('mkdir -p %s' % args.stats_dir)
        amr_statistics.collect_stats(amr_graphs)
        amr_statistics.dump2dir(args.stats_dir)

    n_success = 0.0
    total_width = 0.0
    max_width = 0.0
    width_dist = defaultdict(int)

    messed_sents = set([16453, 16247, 4179, 2040])
    for (sent_index, (tok_seq, lemma_seq, pos_seq, alignment_seq, amr)) in enumerate(zip(
            toks, lemmas, poss, alignments, amr_graphs)):

        test_case = 1
        #if sent_index < test_case:
        #    continue
        #elif sent_index > test_case:
        #    break
        #if sent_index > 3:
        #    break
        #if sent_index > 1:
        #    continue
        #if sent_index < 573:
        #    continue

        if sent_index not in messed_sents:
            continue

        capacities = defaultdict(int)  #Map from each reentrancy index to a number
        logger.writeln('Sentence #%d' % sent_index)
        logger.writeln('tok sequence: %s' % (' '.join(tok_seq)))
        logger.writeln('lemma sequence: %s' % (' '.join(lemma_seq)))
        logger.writeln('AMR graph:\n%s' % str(amr))
        print 'Sentence #%d' % sent_index
        print str(amr)

        amr.setStats(amr_statistics)

        #has_cycle = False
        if amr.check_self_cycle():
            num_self_cycle += 1
            #has_cycle = True
            logger.writeln('self cycle detected')

        amr.set_sentence(tok_seq)
        amr.set_lemmas(lemma_seq)
        amr.set_poss(pos_seq)

        has_multiple = False
        no_alignment = False

        aligned_set = set()

        (opt_toks, role_toks, node_to_span, edge_to_span, temp_aligned) = extractNodeMapping(alignment_seq, amr)

        temp_unaligned = set(xrange(len(pos_seq))) - temp_aligned

        aligned_toks = set()

        all_alignments = defaultdict(list)
        nodeid_to_frag = {}

        ####Extract named entities#####
        for (frag, wiki_label) in amr.extract_entities():
            if len(opt_toks) == 0:
                logger.writeln("No alignment for the entity found")
                no_alignment = True

            assert len(frag.roots) == 1
            root_index = frag.roots[0]

            (aligned_indexes, entity_spans) = all_aligned_spans(frag, opt_toks, role_toks, temp_unaligned)
            root_node = amr.nodes[root_index]

            entity_mention_toks = root_node.namedEntityMention()

            if entity_spans:
                try:
                    entity_spans = removeAligned(entity_spans, aligned_toks)
                except:
                    print entity_spans
                    sys.exit(1)

            total_num += 1.0
            if entity_spans:
                frag.setCategory("NER")

                nodeid_to_frag[root_index] = frag
                entity_spans = removeRedundant(tok_seq, entity_spans, entity_mention_toks)
                if len(entity_spans) == 1:
                    singleton_num += 1.0
                    logger.writeln('Single fragment')
                    for (start, end) in entity_spans:
                        aligned_toks |= set(xrange(start, end))
                        logger.writeln(' '.join(tok_seq[start:end]))
                        all_alignments[root_index].append((start, end, wiki_label))
                else:
                    multiple_num += 1.0
                    assert len(entity_spans) > 1
                    logger.writeln('Multiple fragment')
                    logger.writeln(aligned_indexes)
                    logger.writeln(' '.join([tok_seq[index] for index in aligned_indexes]))

                    for (start, end) in entity_spans:
                        logger.writeln(' '.join(tok_seq[start:end]))
                        aligned_toks |= set(xrange(start, end))
                        all_alignments[root_index].append((start, end, wiki_label))
            else:
                empty_num += 1.0
                no_alignment = True

        if no_alignment:
            logger.write("No alignment found")
            continue

        ####Process date entities
        date_entity_frags = amr.extract_all_dates()
        for frag in date_entity_frags:
            #all_date_indices, index_to_attr = getDateAttr(frag)
            covered_toks, non_covered, index_to_toks = getSpanSide(tok_seq, alignment_seq, frag, temp_unaligned)

            covered_set = set(covered_toks)

            assert len(frag.roots) == 1
            root_index = frag.roots[0]

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
                    frag.setCategory("DATE")
                    nodeid_to_frag[root_index] = frag

                    all_alignments[root_index].append((start, end, None))
                    aligned_toks |= set(xrange(start, end))

            else:
                for index in temp_unaligned:
                    curr_tok = tok_seq[index]
                    found = False
                    for un_tok in non_covered:
                        if curr_tok[0] in '0123456789' and curr_tok in un_tok:
                            print 'recovered: %s' % curr_tok
                            found = True
                            break
                    if found:
                        frag.setCategory("DATE")
                        assert root_index not in nodeid_to_frag
                        nodeid_to_frag[root_index] = frag

                        all_alignments[root_index].append((index, index+1, None))
                        aligned_toks.add(index)

        #Verbalization list
        #verb_map = {}
        #for (index, curr_tok) in enumerate(tok_seq):
        #    if curr_tok in VERB_LIST:

        #        for subgraph in VERB_LIST[curr_tok]:

        #            matched_frags = amr.matchSubgraph(subgraph)
        #            if matched_frags:
        #                temp_aligned.add(index)

        #            for (node_index, ex_rels) in matched_frags:
        #                all_alignments[node_index].append((index, index+1, None))
        #                verb_map[node_index] = subgraph

        for node_index in node_to_span:
            if node_index in all_alignments:
                continue

            try:
                spans = [(x,y) for (x, y,_) in node_to_span[node_index]]
                all_spans = removeAligned(spans, aligned_toks)
            except:
                print node_to_span[node_index]
                sys.exit(1)

            if all_spans:
                for start, end in all_spans:
                    aligned_toks |= set(xrange(start, end))
                    all_alignments[node_index].append((start, end, None))

        unaligned_toks = set(xrange(len(pos_seq))) - aligned_toks
        verb_map = {}
        assert len(tok_seq) == len(pos_seq)
        try:
            new_amr, _, span_to_type = AMRGraph.collapsedAmr(amr, all_alignments, tok_seq, pos_seq, unaligned_toks,
                                        verb_map, nodeid_to_frag, 30, 30)
        except:
            print 'problematic case here:', sent_index
            continue

        collapsed_toks, collapse_pos, new_alignment = collapseTokens(tok_seq, pos_seq, span_to_type)
        logger.writeln("After collapse:")
        logger.writeln(str(new_amr))

        new_amr.set_sentence(collapsed_toks)
        new_amr.set_lemmas(collapsed_toks)
        new_amr.set_poss(collapse_pos)
        new_amr.setStats(amr_statistics)

        span_indexes = []
        for node_index in new_alignment:
            for (start, end, _) in new_alignment[node_index]:
                span_indexes.append((start, end, node_index))
        sorted_indexes = sorted(span_indexes, key=lambda x: (x[0], x[1]))

        print_info = False
        sorted_indexes = [z for (x, y, z) in sorted_indexes]

        if args.depth:
            pi_seq, edge_map, edge_list = depthFirstPi(new_amr, collapsed_toks, new_alignment, sorted_indexes, print_info)
        else:
            pi_seq, edge_map, edge_list = buildPiSeq(new_amr, collapsed_toks, new_alignment, sorted_indexes, print_info)

            if args.random:
                random.shuffle(pi_seq)
            elif args.reversed:
                pi_seq = pi_seq[::-1]

        assert len(pi_seq) == len(new_amr.nodes)
        aligned = []
        for node_index in pi_seq:
            curr_node = new_amr.nodes[node_index]
            curr_frag = new_amr.build_entity_fragment(curr_node)
            aligned.append(curr_frag)

        #span_to_frag = initAlignments(new_amr, collapsed_toks, collapse_pos, new_alignment, unaligned_toks,
        #                              verb_map, nodeid_to_frag, 0, 0)

        #edge_alignment = bitarray(len(new_amr.edges))
        #if edge_alignment.count() != 0:
        #    edge_alignment ^= edge_alignment

        #assert edge_alignment.count() == 0

        #sorted_fragments = sorted(span_to_frag.items(), key=lambda x: (x[0][0], x[0][1]))

        #aligned = []

        #for ((start, end), frag) in sorted_fragments:
        #    frag.build_ext_list()
        #    frag.build_ext_set()
        #    edge_alignment |= frag.edges

        #    root_edge = frag.rootEdge()
        #    capacities[root_edge] += 1
        #    if capacities[root_edge] > 1:
        #        print "multiple to one found:", sent_index
        #        frag = deepcopy(frag)

        #    frag.setSpan(start, end)
        #    aligned.append(frag)

        #    print "%d-%d, %s : %s , %s" % (start, end, ' '.join(collapsed_toks[start:end]),
        #                              str(frag), frag.category)

        print ' '.join(collapsed_toks)
        print str(new_amr)
        rule_f.write('#Sentence %d: %s\n' % (sent_index, ' '.join(tok_seq)))
        rule_f.write('#Collapsed Sentence: %s\n' % (' '.join(collapsed_toks)))

        unaligned = []

        hg, curr_width = fragment_decomposition_forest(aligned, new_amr, unaligned, capacities)

        if hg:
            print 'current width:', curr_width
            n_success += 1.0

            total_width += curr_width
            if curr_width > max_width:
                max_width = curr_width
            width_dist[curr_width] += 1

            curr_sample = Sample(hg, sent_index)
            derivation_rules = curr_sample.extract_derivation()
            for rule in derivation_rules:
                rule_str = filter_vars(rule.dumped_format())
                print >> rule_f, rule_str
        else:
            print 'Failed to construct forest'
            logger.writeln("Failed to build a forest")

        #    constructed_forests.append(hg)
        #    sent_indexes.append(sent_index)

        #else:
        #    logger.writeln("Failed to build a forest")

    #forest_f = open(forest_file, 'wb')
    #cPickle.dump(constructed_forests, forest_f)
    #forest_f.close()
    rule_f.close()
    print 'Final sucessessful sentences:', n_success
    sorted_width = sorted(width_dist.items(), key=lambda x: -x[1])
    for (width, count) in sorted_width:
        print 'width: %d, count: %d' % (width, count)
    print 'Maximum width:', max_width
    print 'Average width:', total_width/n_success
    #lemma_rule_f.close()

    #used_sent_f = open(used_sent_file, 'wb')
    #cPickle.dump(sent_indexes, used_sent_f)
    #used_sent_f.close()
    #logger.writeln('total used sents is: %d' % used_sents)
    #logger.writeln('total dump forests: %d' % len(constructed_forests))
    logger.writeln('finished')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--amr_file", type=str, help="the original AMR graph files", required=False)
    argparser.add_argument("--stop", type=str, help="stop words file", required=False)
    argparser.add_argument("--lemma", type=str, help="lemma file", required=False)
    argparser.add_argument("--dump_graph", action="store_true", help="if only to dump graph object")
    argparser.add_argument("--preprocess", action="store_true", help="if needed to preprocess the AMR graphs into dumped AMR graph objects")
    argparser.add_argument("--parallel", action="store_true", help="if to run multiple process to run the forest construction")
    argparser.add_argument("--dump_rules", action="store_true", help="if to dump lexical rules")
    argparser.add_argument("--use_stats", action="store_true", help="if to use stats")
    argparser.add_argument("--refine", action="store_true", help="if to refine the nonterminals")
    argparser.add_argument("--random", action="store_true", help="if use random order")
    argparser.add_argument("--reversed", action="store_true", help="if to reverse the pi sequence")
    argparser.add_argument("--depth", action="store_true", help="if to reverse the pi sequence")
    argparser.add_argument("--data_dir", type=str, help="the data directory for dumped AMR graph objects, alignment and tokenized sentences")
    argparser.add_argument("--save_dir", type=str, help="the output directory for saving the constructed forest")
    argparser.add_argument("--stats_dir", type=str, help="the statistics directory")
    argparser.add_argument("--nodes", type=str, help="nodes for running processes")
    argparser.add_argument("--sent_per_node", type=int, help="number of sentences for each node")
    argparser.add_argument("--nsplit", type=int, help="if to split the forest into multiple")

    args = argparser.parse_args()
    construct_forest(args)
