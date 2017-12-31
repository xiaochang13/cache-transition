'''
categorize amr; generate linearized amr sequence
'''
import sys, os, re, codecs
import string
import gflags
from amr_graph import AMR
from collections import OrderedDict, defaultdict
#from constants import TOP,LBR,RBR,RET,SURF,CONST,END,VERB
from constants import *
from parser import ParserError
from date_extraction import *
FLAGS=gflags.FLAGS
gflags.DEFINE_integer("min_prd_freq",50,"threshold for filtering out predicates")
gflags.DEFINE_integer("min_var_freq",50,"threshold for filtering out non predicate variables")
gflags.DEFINE_string("seq_cat",'freq',"mode to output sequence category")
class AMR_stats(object):
    def __init__(self):
        self.num_reentrancy = 0
        self.num_predicates = defaultdict(int)
        self.num_nonpredicate_vals = defaultdict(int)
        self.num_consts = defaultdict(int)
        self.num_entities = defaultdict(int)
        self.num_named_entities = defaultdict(int)

    def collect_stats(self, amrs):
        for amr in amrs:
            named_entity_nums, entity_nums, predicate_nums, variable_nums, const_nums, reentrancy_nums = amr.statistics()

            self.update(reentrancy_nums, predicate_nums, variable_nums, const_nums, entity_nums, named_entity_nums)

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

        dump_file(pred_f, self.num_predicates)
        dump_file(non_pred_f, self.num_nonpredicate_vals)
        dump_file(const_f, self.num_consts)
        dump_file(entity_f, self.num_entities)
        dump_file(named_entity_f, self.num_named_entities)

    def __str__(self):
        s = ''
        s += 'Total number of reentrancies: %d\n' % self.num_reentrancy
        s += 'Total number of predicates: %d\n' % len(self.num_predicates)
        s += 'Total number of non predicates variables: %d\n' % len(self.num_nonpredicate_vals)
        s += 'Total number of constants: %d\n' % len(self.num_consts)
        s += 'Total number of entities: %d\n' % len(self.num_entities)
        s += 'Total number of named entities: %d\n' % len(self.num_named_entities)
        return s


class AMR_seq:
    def __init__(self, stats=None):
        self.stats = stats
        self.min_prd_freq = FLAGS.min_prd_freq
        self.min_var_freq = FLAGS.min_var_freq
        #pass

    def linearize_amr(self, instance):
        '''
        given an amr graph, output a linearized and categorized amr sequence;
        TODO: use dfs prototype
        '''
        #pass

        amr = instance[0]

        r = amr.roots[0] # single root assumption
        old_depth = -1
        depth = -1
        stack = [(r,TOP,None,0)]
        aux_stack = []
        seq = []

        while stack:
            old_depth = depth
            cur_var, rel, parent, depth = stack.pop()
            #exclude_rels = []

            i = 0
            #print seq
            #print stack
            #print aux_stack
            while old_depth - depth >= i:
                if aux_stack == []:
                    import pdb
                    pdb.set_trace()
                seq.append(aux_stack.pop())
                i+=1


            if (parent, rel, cur_var) in amr.reentrance_triples:
                #seq.extend([rel+LBR,RET,RBR+rel])
                seq.append(rel+LBR)
                seq.append(RET)
                aux_stack.append(RBR+rel)
                continue

            seq.append(rel+LBR)
            exclude_rels, cur_symbol = self.get_symbol(cur_var, instance, mode=FLAGS.seq_cat)
            seq.append(cur_symbol)
            aux_stack.append(RBR+rel)

            for rel, var in reversed(amr[cur_var].items()):
                if rel not in exclude_rels:
                    stack.append((var[0],rel,cur_var,depth+1))

        seq.extend(aux_stack[::-1])
        #seq.append(END)

        return seq

    def _get_pred_symbol(self, var, instance, mode):
        amr, alignment, tok, pos = instance
        if mode == 'basic':
            pred_name = amr.node_to_concepts[var]
            return pred_name
        elif mode == 'freq':
            pred_name = amr.node_to_concepts[var]
            if self.stats.num_predicates[pred_name] >= self.min_prd_freq:
                return pred_name
            else:
                sense = pred_name.split('-')[-1]
                return VERB+sense

                # TODO further categorize the verb type using verbalization list
        else:
            raise Exception('Unexpected mode %s' % (mode))

    def _get_variable_symbal(self, var, instance, mode):
        amr, alignment, tok, pos = instance
        if mode == 'basic':
            variable_name = amr.node_to_concepts[var]
            return variable_name
        elif mode == 'freq':
            variable_name = amr.node_to_concepts[var]
            if self.stats.num_nonpredicate_vals[variable_name] >= self.min_var_freq:
                return variable_name
            else:
                return SURF

                # TODO further categorize the variable type
        else:
            raise Exception('Unexpected mode %s' % (mode))

    def get_symbol(self, var, instance, mode='basic'):
        '''
        get symbol for each amr concept and variable
        mode:
            "basic": no frequence filtering
            "freq": frequence filtering
        '''
        amr = instance[0]

        if amr.is_named_entity(var):
            exclude_rels = ['wiki','name']
            entity_name = amr.node_to_concepts[var]
            return exclude_rels, 'NE_'+entity_name
        elif amr.is_entity(var):
            entity_name = amr.node_to_concepts[var]
            return [], 'ENT_'+entity_name
        elif amr.is_predicate(var):
            return [], self._get_pred_symbol(var, instance, mode=mode)
        elif amr.is_const(var):
            if var in ['interrogative', 'imperative', 'expressive', '-']:
                return [], var
            else:
                return [], CONST

        else:
            #variable_name = amr.node_to_concepts[var]
            #return [], variable_name
            return [], self._get_variable_symbal(var, instance, mode=mode)

        return [],var

    def restoreAMRs(self, amr_conll_file, toks, lemmas, poss, reprs, mle_map, lem_mle_map):
        def register_var(token):
            num = 0
            while True:
                currval = ('%s%d' % (token[0], num)) if token[0] in string.letters else ('a%d' % num)
                if currval in var_set:
                    num += 1
                else:
                    var_set.add(currval)
                    return currval

        def children(nodelabel, node_to_children):
            ret = set()
            stack = [nodelabel]
            visited_nodes = set()
            while stack:
                curr_node = stack.pop()
                if curr_node in visited_nodes:
                    continue
                visited_nodes.add(curr_node)
                ret.add(curr_node)
                if curr_node in node_to_children:
                    ret |= node_to_children[curr_node]
                    for child in node_to_children[curr_node]:
                        stack.append(child)
            return ret

        #In case there is a loop, root is a node that all parents can be visited?
        def valid_root(nodelabel, node_to_children, parents):
            canbevisited = children(nodelabel, node_to_children)
            return len(parents-canbevisited) == 0

        def replace_unk(word, lemma, pos):
            if word in mle_map:
                return mle_map[word]
            elif lemma in lem_mle_map:
                return lem_mle_map[lemma]
            elif "V" in pos:
                return lemma + "-01"
            elif word != lemma:
                print>>sys.stderr, "Comparison:", word, lemma
                if word == lemma + "ing":
                    return lemma + "-01"
                else:
                    return lemma
            else:
                return word

        def is_const(s):
            const_set = set(['interrogative', 'imperative', 'expressive', '-'])
            return s.isdigit() or s in const_set or isNumber(s)

        def value(s): #Compute the value of the number representation
            number = 1

            for v in s.split():
                if isNumber(v):
                    if '.' in v:
                        if len(s.split()) == 1:
                            return v
                        number *= float(v)
                    else:
                        v = v.replace(",", "")
                        number *= int(v)
                else:
                    v = v.lower()
                    assert v in quantities, v
                    number *= quantities[v]
                    number = int(number)
            return str(number)

        amr_graphs = []

        amr_col_lines = [line.strip() for line in open(amr_conll_file, 'r')]

        n_lines = len(amr_col_lines)

        #print 'A total of lines:', n_lines

        #Initialize the first sentence
        start_sent = True
        line_no = 0
        sent_index = 0

        tok_seq = toks[sent_index]
        lemma_seq = lemmas[sent_index]
        pos_seq = poss[sent_index]
        cate_repr = reprs[sent_index]

        var_set = set()
        nodeid_to_label = {}
        label_to_rels = defaultdict(list)
        visited = set()
        vertices = set()
        amr = AMR()
        ent_index = 0

        connected_nodes = set()

        label_to_children = defaultdict(set)
        label_to_parents = defaultdict(set)
        while line_no < n_lines:
            curr_line = amr_col_lines[line_no]
            if start_sent: #Skip the first two lines of each sentence
                line_no += 2
                start_sent = False
                continue

            #start_sent = False

            curr_line = amr_col_lines[line_no]

            if curr_line == "": #The end of a sentence
                sent_index += 1

                #all the relations in the AMR
                for nodelabel in label_to_rels:

                    visited_rels = set()
                    rels = label_to_rels[nodelabel]
                    try:
                        triples = rels.split("#")
                    except:
                        print nodelabel
                        print rels
                        sys.exit(1)
                    #print nodelabel, ":", rels
                    for rs in reversed(triples):
                        l = rs.split(":")[0]
                        if "UNKNOWN" in l:
                            continue
                        concept_index = int(rs.split(":")[1])
                        #print rs
                        taillabel = nodeid_to_label[concept_index]
                        if not is_const(taillabel):
                            if l in visited_rels and taillabel in connected_nodes:
                                continue
                        visited_rels.add(l)
                        connected_nodes.add(taillabel)

                        label_to_children[nodelabel].add(taillabel)
                        label_to_parents[taillabel].add(nodelabel)

                        #if l in visited_rels:
                        #    continue
                        #visited_rels.add(l)

                        #print 'triple:', nodelabel, l, taillabel
                        amr._add_triple(nodelabel, l, tuple([taillabel]))

                for root in amr.roots:
                    visited |= children(root, label_to_children)

                unfound = vertices - visited

                for label in unfound:
                    if label in visited:
                        continue
                    parents = set()
                    if label in label_to_parents:
                        parents = label_to_parents[label]
                    if valid_root(label, label_to_children, parents):
                        amr.roots.append(label)
                        visited |= children(label, label_to_children)

                try:
                    assert len(visited) == len(vertices)
                except:
                    print (vertices - visited)
                    print (visited - vertices)
                    print sent_index
                    sys.exit(1)

                ent_index = 0

                amr_graphs.append(amr)

                #print sent_index
                print amr.to_amr_string()
                print ""

                amr = AMR()
                connected_nodes = set()
                var_set = set()
                visited = set()
                vertices = set()

                label_to_children = defaultdict(set)
                label_to_parents = defaultdict(set)
                label_to_rels = {}
                nodeid_to_label = {}

                line_no += 1
                if sent_index < len(toks):
                    tok_seq = toks[sent_index]
                    lemma_seq = lemmas[sent_index]
                    pos_seq = poss[sent_index]
                    cate_repr = reprs[sent_index]
                    start_sent = True
                continue

            fields = curr_line.split()
            tok_index = int(fields[0])
            word = fields[1]
            lemma = lemma_seq[tok_index]
            pos = pos_seq[tok_index]
            assert tok_seq[tok_index] == fields[1]

            concept_index = fields[2]

            if concept_index == "NONE": #Unaligned word
                assert 'NE' not in word and "DATE" not in word, word
            else: #A new concept is identified
                concept_index = int(concept_index)
                concept = fields[3]
                if "UNKNOWN" in concept:
                    concept = replace_unk(word, lemma, pos)

                rels = fields[4]
                par_rels = fields[5]
                if "NE" in word or "DATE" in word or word == "NUMBER" or "VB_" in word: #Named entities
                    curr_ner, tok_repr, wiki_label = cate_repr[ent_index]
                    #curr_ner, tok_repr, root_repr, wiki_label = cate_repr[ent_index]
                    ent_index += 1
                    assert curr_ner == word, "%s %s %s" % (curr_ner, word, tok_repr)
                    if "NE" in word:   #Need to build a named entity
                        if word == "NE":
                            root_repr = "person" #Here the concept id net sucks, why this happened???
                        else:
                            assert word[:3] == "NE_"
                            root_repr = word[3:]
                        nodelabel = register_var(root_repr)
                        nodeid_to_label[concept_index] = nodelabel
                        concept = root_repr

                        l = 'wiki'
                        child = tuple([("\"%s\"" % wiki_label)])
                        amr._add_triple(nodelabel, l, child)

                        l = 'name'
                        name_v = register_var("name")
                        amr.node_to_concepts[name_v] = "name"

                        child = tuple([name_v])
                        amr._add_triple(nodelabel, l, child)

                        for (op_index, s) in enumerate(tok_repr.split()):
                            l = "op%d" % (op_index+1)
                            child = tuple([("\"%s\"" % s)])
                            amr._add_triple(name_v, l, child)
                    elif word == "NUMBER":
                        value_repr = value(tok_repr)
                        concept = value_repr
                        assert rels == "NONE", value_repr
                        if (par_rels == "NONE"):
                            nodelabel = register_var("number")
                            concept = "number"

                            l = "quant"
                            child = tuple([value_repr])
                            amr._add_triple(nodelabel, l, child)
                            nodeid_to_label[concept_index] = nodelabel
                            amr.node_to_concepts[nodelabel] = "num"
                        else:
                            nodeid_to_label[concept_index] = value_repr
                            nodelabel = value_repr
                            rels = "NONE"
                    elif "VB_" in word:
                        tok_repr = tok_repr.lower()
                        assert tok_repr in VERB_LIST, "%s %s" % (tok_repr, word)
                        subgraph = "NONE"
                        suffix = word[3:]
                        s = None
                        if suffix == "VERB":
                            for graph in VERB_LIST[tok_repr]:
                                for root_str in graph:
                                    if re.match('.*-[0-9]+', root_str) is not None:

                                        s = "ok"
                                        nodelabel = register_var(root_str)
                                        concept = root_str
                                        nodeid_to_label[concept_index] = nodelabel
                                        #amr.node_to_concepts[nodelabel] = root_str
                                        for (rel, tail_concept) in graph[root_str].items():
                                            child_v = register_var(tail_concept)

                                            child = tuple([child_v])
                                            amr._add_triple(nodelabel, rel, child)
                                            amr.node_to_concepts[child_v] = tail_concept
                                            foo = amr[child_v]

                                if s is not None:
                                    break
                        else:
                            for graph in VERB_LIST[tok_repr]:
                                for root_str in graph:
                                    if root_str == suffix:
                                        s = "ok"
                                        nodelabel = register_var(root_str)
                                        nodeid_to_label[concept_index] = nodelabel
                                        amr.node_to_concepts[nodelabel] = root_str
                                        for (rel, tail_concept) in graph[root_str].items():
                                            child_v = register_var(tail_concept)
                                            child = tuple([child_v])
                                            amr._add_triple(nodelabel, rel, child)
                                            amr.node_to_concepts[child_v] = tail_concept
                                            foo = amr[child_v]
                                if s is not None:
                                    break
                        assert s is not None

                    else: #Date entities
                        date_rels = dateRepr(tok_repr.split())
                        #print date_rels
                        root_repr = 'date-entity'
                        nodelabel = register_var(root_repr)
                        amr.node_to_concepts[nodelabel] = root_repr #Newly added
                        nodeid_to_label[concept_index] = nodelabel
                        concept = root_repr
                        for l, subj in date_rels:
                            child = tuple([subj])
                            amr._add_triple(nodelabel, l, child)


                else:
                    if not is_const(concept):
                        nodelabel = register_var(concept)
                        assert not concept_index in nodeid_to_label
                        nodeid_to_label[concept_index] = nodelabel
                        #amr.node_to_concepts[
                    else:
                        nodeid_to_label[concept_index] = concept
                        nodelabel = concept


                if is_const(concept):
                    if (rels != "NONE") or (par_rels == "NONE"):
                        nodelabel = register_var(concept)
                        nodeid_to_label[concept_index] = nodelabel
                        amr.node_to_concepts[nodelabel] = concept
                        vertices.add(nodelabel)
                        foo = amr[nodelabel]
                        if rels != "NONE": #Save the relations for further processing
                            label_to_rels[nodelabel] = rels
                        if par_rels == "NONE": #Roots are nodes without parents
                            amr.roots.append(nodelabel)
                    else:
                        vertices.add(concept)
                else:
                    vertices.add(nodelabel)

                    if (not is_const(concept)) and (not nodelabel in amr.node_to_concepts):
                        amr.node_to_concepts[nodelabel] = concept
                        foo = amr[nodelabel] #Put current node in the AMR
                    #print 'node label:', nodelabel, concept

                    if rels != "NONE": #Save the relations for further processing
                        label_to_rels[nodelabel] = rels
                    if par_rels == "NONE": #Roots are nodes without parents
                        amr.roots.append(nodelabel)
            line_no += 1

        return amr_graphs

    def restore_amr(self, tok_seq, lemma_seq, amrseq, repr_map):
        '''
        Given a linearized amr sequence, restore its amr graph
        Deal with non-matching parenthesis
        '''
        def rebuild_seq(parsed_seq):
            new_seq = []
            stack = []
            try:

                while parsed_seq[-1][1] == "LPAR": #Delete the last few left parenthesis
                    parsed_seq = parsed_seq[:-1]

                assert len(parsed_seq) > 0, parsed_seq
                for (token, type) in parsed_seq:
                    if type == "LPAR": #Left parenthesis
                        if stack and stack[-1][1] == "LPAR":
                            new_token = 'ROOT'
                            new_type = 'NONPRED'
                            stack.append((new_token, new_type))
                            new_seq.append((new_token, new_type))

                        stack.append((token, type))
                        new_seq.append((token, type))
                    elif type == "RPAR": #Right parenthesis
                        assert stack
                        if stack[-1][1] == "LPAR": #No concept for this edge, remove
                            stack.pop()
                            new_seq.pop()
                        elif stack[-2][0][:-1] == '-TOP-':
                            continue
                        else:
                            stack.pop()
                            ledgelabel, ltype = stack.pop()
                            try:
                                assert ltype == "LPAR", ('%s %s'% (ledgelabel, ltype))
                            except:
                                print stack
                                print ledgelabel, ltype
                                print token, type
                                sys.exit(1)
                            redgelabel = ')%s' % (ledgelabel[:-1])
                            new_seq.append((redgelabel, "RPAR"))
                    else:
                        if stack[-1][1] == "LPAR":
                            stack.append((token, type))
                            new_seq.append((token, type))
                while stack:
                    while stack[-1][1] == "LPAR" and stack:
                        stack.pop()

                    if not stack:
                        break

                    stack.pop()
                    ledgelabel, ltype = stack.pop()
                    assert ltype == "LPAR"
                    redgelabel = ')%s' % (ledgelabel[:-1])
                    new_seq.append((redgelabel, "RPAR"))

                return new_seq
            except:
                print parsed_seq
                print new_seq
                #sys.exit(1)
                return new_seq

        def make_compiled_regex(rules):
            regexstr =  '|'.join('(?P<%s>%s)' % (name, rule) for name, rule in rules)
            return re.compile(regexstr)

        def register_var(token):
            num = 0
            while True:
                currval = ('%s%d' % (token[0], num)) if token[0] in string.letters else ('a%d' % num)
                if currval in var_set:
                    num += 1
                else:
                    var_set.add(currval)
                    return currval

        def buildLinearEnt(entity_name, ops, wiki_label):
            ops_strs = ['op%d( "%s" )op%d' % (index, s, index) for (index, s) in enumerate(ops, 1)]
            new_op_strs = []
            for tok in ops_strs:
                if '"("' in tok or '")"' in tok:
                    break
                new_op_strs.append(tok)
            wiki_str = 'wiki( "%s" )wiki' % wiki_label
            ent_repr = '%s %s name( name %s )name' % (entity_name, wiki_str, ' '.join(new_op_strs))
            #ent_repr = '%s %s name( name %s )name' % (entity_name, wiki_str, ' '.join(ops_strs))
            return ent_repr

        def buildLinearVerbal(lemma, node_repr):
            assert lemma in VERB_LIST
            subgraph = None
            for g in VERB_LIST[lemma]:
                if node_repr in g:
                    subgraph = g
                    break

            verbal_repr = node_repr
            for rel, subj in subgraph[node_repr].items():
                verbal_repr = '%s %s( %s )%s' % (verbal_repr, rel, subj, rel)
            return verbal_repr

        def buildLinearDate(rels):
            date_repr = 'date-entity'
            for rel, subj in rels:
                date_repr = '%s %s( %s )%s' % (date_repr, rel, subj, rel)
            return date_repr

        amr = AMR()
        #print amrseq.strip()
        seq = amrseq.strip().split()

        new_seq = []

        #Deal with redundant RET
        skip = False
        for (i, tok) in enumerate(seq):
            if skip:
                skip = False
                continue
            if tok in repr_map:
                #tok_nosuffix = re.sub('-[0-9]+', '', tok)

                start, end, node_repr, wiki_label = repr_map[tok]

                if 'NE' in tok: #Is a named entity
                    #print ' '.join(tok_seq[start:end])
                    branch_form = buildLinearEnt(node_repr, tok_seq[start:end], wiki_label)  #Here rebuild the op representation of the named entity
                    new_seq.append(branch_form)
                elif 'VERBAL' in tok:  #Is in verbalization list, should rebuild
                    assert end == start + 1
                    branch_form = buildLinearVerbal(lemma_seq[start], node_repr)
                    new_seq.append(branch_form)
                elif 'DATE' in tok: #Rebuild a date entity
                    rels = dateRepr(tok_seq[start:end])
                    branch_form = buildLinearDate(rels)
                    new_seq.append(branch_form)
                else:
                    new_seq.append(node_repr)
            else:
                if 'NE' in tok:
                    tok = re.sub('-[0-9]+', '', tok)
                    tok = tok[3:]
                elif 'ENT' in tok:
                    tok = re.sub('-[0-9]+', '', tok)
                    tok = tok[4:]
                elif 'RET' in tok or isSpecial(tok):  #Reentrancy, currently not supported
                    prev_elabel = new_seq[-1]
                    try:
                        assert prev_elabel[-1] == '(', prev_elabel
                    except:
                        print 'RET after a label', prev_elabel
                        return None
                    prev_elabel = prev_elabel[:-1]
                    if i +1 < len(seq):
                        next_elabel = seq[i+1]
                        if next_elabel[0] == ')':
                            next_elabel = next_elabel[1:]
                            if prev_elabel == next_elabel:
                                skip = True
                                new_seq.pop()
                                continue
                    else:
                        new_seq.pop()
                        continue
                        #print 'Weird here'
                        #print seq

                new_seq.append(tok)

        amrseq = ' '.join(new_seq)
        seq = amrseq.split()
        triples = []

        stack = []
        state = 0
        node_idx = 0; # sequential new node index
        mapping_table = {};  # old new index mapping table

        var_set = set()

        const_set = set(['interrogative', 'imperative', 'expressive', '-'])
        lex_rules = [
            ("LPAR", '[^\s()]+\('),  #Start of an edge
            ("RPAR",'\)[^\s()]+'),  #End of an edge
            #("SURF", '-SURF-'),  #Surface form non predicate
            ("VERB", '-VERB-\d+'), # predicate
            ("CONST", '"[^"]+"'), # const
            ("REENTRANCY", '-RET-'),  #Reentrancy
            ("ENTITY", 'ENT_([^\s()]+)'),  #Entity
            ("NER", 'NE_([^\s()]+)'), #Named entity
            ("PRED", '([^\s()]+)-[0-9]+'), #Predicate
            ("NONPRED", '([^\s()]+)'),  #Non predicate variables
            ("POLARITY", '\s(\-|\+)(?=[\s\)])')
        ]

        token_re = make_compiled_regex(lex_rules)

        parsed_seq = []
        for match in token_re.finditer(amrseq):
            token = match.group()
            type = match.lastgroup
            parsed_seq.append((token, type))

        PNODE = 1
        CNODE = 2
        LEDGE = 3
        REDGE = 4
        RCNODE = 5
        parsed_seq = rebuild_seq(parsed_seq)
        if not parsed_seq:
            return None

        token_seq = [token for (token, type) in parsed_seq]

        seq_length = len(parsed_seq)
        for (currpos, (token, type)) in enumerate(parsed_seq):
            if state == 0: #Start state
                assert type == "LPAR", ('start with symbol: %s' % token)
                edgelabel = token[:-1]
                stack.append((LEDGE, edgelabel))
                state = 1

            elif state == 1: #Have just identified an left edge, next expect a concept
                if type == "NER":
                    nodelabel = register_var(token)
                    nodeconcept = token
                    stack.append((PNODE,nodelabel,nodeconcept))
                    state = 2
                elif type == "ENTITY":
                    nodelabel = register_var(token)
                    nodeconcept = token
                    stack.append((PNODE,nodelabel,nodeconcept))
                    state = 2
                elif type == "PRED":
                    nodelabel = register_var(token)
                    nodeconcept = token
                    stack.append((PNODE,nodelabel,nodeconcept))
                    state = 2
                elif type == "NONPRED":
                    nodelabel = register_var(token)
                    nodeconcept = token
                    stack.append((PNODE,nodelabel,nodeconcept))
                    state = 2
                elif type == "CONST":
                    #if currpos + 1 < seq_length and parsed_seq[currpos+1][1] == "LPAR":
                    #    nodelabel = register_var(token)
                    #    nodeconcept = token
                    #    stack.append((PNODE,nodelabel,nodeconcept))
                    #else:
                    stack.append((PNODE,token.strip(),None))
                    state = 2
                elif type == "REENTRANCY":
                    if currpos + 1 < seq_length and parsed_seq[currpos+1][1] == "LPAR":
                        nodelabel = register_var(token)
                        nodeconcept = token
                        stack.append((PNODE,nodelabel,nodeconcept))
                    else:
                        stack.append((PNODE,token.strip(),None))
                    state = 2
                elif type == "POLARITY":
                    if currpos + 1 < seq_length and parsed_seq[currpos+1][1] == "LPAR":
                        nodelabel = register_var(token)
                        nodeconcept = token
                        stack.append((PNODE,nodelabel,nodeconcept))
                    else:
                        stack.append((PNODE,token.strip(),None))
                    state = 2
                else: raise ParserError , "Unexpected token %s"%(token.encode('utf8'))

            elif state == 2: #Have just identified a PNODE concept
                if type == "LPAR":
                    edgelabel = token[:-1]
                    stack.append((LEDGE, edgelabel))
                    state = 1
                elif type == "RPAR":
                    assert stack[-1][0] == PNODE
                    forgetme, nodelabel, nodeconcept = stack.pop()
                    if not nodelabel in amr.node_to_concepts and nodeconcept is not None:
                        amr.node_to_concepts[nodelabel] = nodeconcept

                    foo = amr[nodelabel]
                    if stack and stack[-1][1] != "-TOP-": #This block needs to be updated
                        stack.append((CNODE, nodelabel, nodeconcept))
                        state = 3
                    else: #Single concept AMR
                        assert len(stack) == 1 and stack[-1][1] == "-TOP-", "Not start with TOP"
                        stack.pop()
                        if amr.roots:
                            break
                        amr.roots.append(nodelabel)
                        state = 0
                        #break
                else: raise ParserError, "Unexpected token %s"%(token)

            elif state == 3: #Have just finished a CNODE, which means wrapped up with one branch
                if type == "LPAR":
                    edgelabel = token[:-1]
                    stack.append((LEDGE, edgelabel))
                    state = 1

                elif type == "RPAR":
                    edges = []
                    while stack[-1][0] != PNODE:
                        children = []
                        assert stack[-1][0] == CNODE, "Expect a parsed node but none found"
                        forgetme, childnodelabel, childconcept = stack.pop()
                        children.append((childnodelabel,childconcept))

                        assert stack[-1][0] == LEDGE, "Found a non-left edge"
                        forgetme, edgelabel = stack.pop()

                        edges.append((edgelabel,children))

                    forgetme,parentnodelabel,parentconcept = stack.pop()

                    #check for annotation error
                    if parentnodelabel in amr.node_to_concepts:
                        print parentnodelabel, parentconcept
                        assert parentconcept is not None
                        if amr.node_to_concepts[parentnodelabel] == parentconcept:
                            sys.stderr.write("Wrong annotation format: Revisited concepts %s should be ignored.\n" % parentconcept)
                        else:
                            sys.stderr.write("Wrong annotation format: Different concepts %s and %s have same node label(index)\n" % (amr.node_to_concepts[parentnodelabel],parentconcept))
                            parentnodelabel = parentnodelabel + "1"


                    if not parentnodelabel in amr.node_to_concepts and parentconcept is not None:
                        amr.node_to_concepts[parentnodelabel] = parentconcept

                    for edgelabel,children in reversed(edges):
                        hypertarget = []
                        for node, concept in children:
                            if node is not None and not node in amr.node_to_concepts and concept:
                                amr.node_to_concepts[node] = concept
                            hypertarget.append(node)
                        hyperchild = tuple(hypertarget)
                        amr._add_triple(parentnodelabel,edgelabel,hyperchild)

                    if stack and stack[-1][1] != "-TOP-": #we have done with current level
                        state = 3
                        stack.append((CNODE, parentnodelabel, parentconcept))
                    else: #Single concept AMR
                        try:
                            assert len(stack) == 1 and stack[-1][1] == "-TOP-", "Not start with TOP"
                        except:
                            print 'Not start with TOP', parsed_seq
                            return None
                        stack.pop()
                        if amr.roots:
                            break
                        amr.roots.append(parentnodelabel)
                        state = 0
                        break
                        #state = 0
                        #amr.roots.append(parentnodelabel)
                else: raise ParserError, "Unexpected token %s"%(token.encode('utf8'))

        if state != 0 and stack:
            raise ParserError, "mismatched parenthesis"
        return amr




def readAMR(amrfile_path):
    amrfile = codecs.open(amrfile_path,'r',encoding='utf-8')
    #amrfile = open(amrfile_path,'r')
    comment_list = []
    comment = OrderedDict()
    amr_list = []
    amr_string = ''

    for line in amrfile.readlines():
        if line.startswith('#'):
            for m in re.finditer("::([^:\s]+)\s(((?!::).)*)",line):
                #print m.group(1),m.group(2)
                comment[m.group(1)] = m.group(2)
        elif not line.strip():
            if amr_string and comment:
                comment_list.append(comment)
                amr_list.append(amr_string)
                amr_string = ''
                comment = {}
        else:
            amr_string += line.strip()+' '

    if amr_string and comment:
        comment_list.append(comment)
        amr_list.append(amr_string)
    amrfile.close()

    return (comment_list,amr_list)

def amr2sequence(toks, amr_graphs, alignments, poss, out_seq_file, amr_stats):
    amr_seq = AMR_seq(stats=amr_stats)
    with open(out_seq_file, 'w') as outf:
        print 'Linearizing ...'
        for i,g in enumerate(amr_graphs):
            print 'No ' + str(i) + ':' + ' '.join(toks[i])
            instance = (g, alignments[i], toks[i], poss[i])
            seq = ' '.join(amr_seq.linearize_amr(instance))
            print >> outf, seq

def sequence2amr(toks, lemmas, amrseqs, cate_to_repr, out_amr_file):
    amr_seq = AMR_seq()
    with open(out_amr_file, 'w') as outf:
        print 'Restoring AMR graphs ...'
        for i, (s, repr_map) in enumerate(zip(amrseqs, cate_to_repr)):
            print 'No ' + str(i) + ':' + ' '.join(toks[i])
            #print repr_map
            restored_amr = amr_seq.restore_amr(toks[i], lemmas[i], s, repr_map)
            if not restored_amr:
                print >> outf, '(a / amr-unknown)'
                print >> outf, ''
                continue
            if 'NONE' in restored_amr.to_amr_string():
                print s
                print repr_map
            print >> outf, restored_amr.to_amr_string()
            print >> outf, ''
        outf.close()

def isSpecial(symbol):
    for l in ['ENT', 'NE', 'VERB', 'SURF', 'CONST']:
        if l in symbol:
            return True
    return False

def loadCategoryMap(map_file):
    cate_to_repr = []   #Maintain a list of information for each sentence
    for line in open(map_file):
        curr_attributes = []
        if line.strip():
            fields = line.strip().split('##')
            for map_tok in fields:
                fs = map_tok.strip().split('++')
                try:
                    assert len(fs) == 3, line
                except:
                    print line
                    print fields
                    print curr_map
                    print fs
                    sys.exit(1)
                if '@@@-@@@' in fs[1]:
                    fs[1] = fs[1].replace('@@@-@@@', ' @-@ ').replace('@@@@@:@@@', ' @:@ ')
                fs[1] = fs[1].replace('@@', ' ')
                #print fs[0], fs[1], fs[2], fs[3]
                curr_attributes.append((fs[0], fs[1], fs[2]))

        cate_to_repr.append(curr_attributes)
    return cate_to_repr

def loadMLEConceptID(map_file):
    mle_map = {}

    #First load all possible mappings each span has
    with open(map_file, 'r') as map_f:
        for line in map_f:
            if line.strip():
                spans = line.strip().split(' #### ')
                mle_map[spans[0].strip()] = spans[1].strip()

    return mle_map

if __name__ == "__main__":

    gflags.DEFINE_string("data_dir",'../train',"data directory")
    gflags.DEFINE_string("amrseq_file",'../dev.decode.amrseq',"amr sequence file")

    argv = FLAGS(sys.argv)

    amr_file = os.path.join(FLAGS.data_dir, 'amr.col')

    #sent_file = os.path.join(FLAGS.data_dir, 'sentence')
    tok_file = os.path.join(FLAGS.data_dir, 'tok')
    lemma_file = os.path.join(FLAGS.data_dir, 'lem')
    pos_file = os.path.join(FLAGS.data_dir, 'pos')

    map_file = os.path.join(FLAGS.data_dir, 'cate_map')
    #conceptID_file = os.path.join(FLAGS.data_dir, 'conceptID.dict')
    conceptID_file = './conceptID.dict'
    #lemConceptID_file = os.path.join(FLAGS.data_dir, 'lemConceptID.dict')
    lemConceptID_file = './lemConceptID.dict'
    mle_map = loadMLEConceptID(conceptID_file)
    lem_mle_map = loadMLEConceptID(lemConceptID_file)

    cate_to_repr = loadCategoryMap(map_file)

    #amrseqs = [line.strip() for line in open(FLAGS.amrseq_file, 'r')]

    #assert len(cate_to_repr) == len(amrseqs)

    toks = [line.strip().split() for line in open(tok_file, 'r')]
    lemmas = [line.strip().split() for line in open(lemma_file, 'r')]
    poss = [line.strip().split() for line in open(pos_file, 'r')]

    #amr_result_file = os.path.join(FLAGS.data_dir, 'eval.amr')
    amr_seq = AMR_seq()
    amr_seq.restoreAMRs(amr_file, toks, lemmas, poss, cate_to_repr, mle_map, lem_mle_map)
    #sequence2amr(toks, lemmas, amrseqs, cate_to_repr, amr_result_file)


