from collections import defaultdict
from ioutil import *
from oracle_data import *
from cacheTransition import CacheTransition
from cacheConfiguration import CacheConfiguration
import time
import argparse
NULL = "-NULL-"
UNKNOWN = "-UNK-"
class CacheTransitionParser(object):
    def __init__(self, size):
        self.cache_size = size
        self.connectWordDistToCount = defaultdict(int)
        self.nonConnectDistToCount = defaultdict(int)
        self.depConnectDistToCount = defaultdict(int)
        self.depNonConnectDistToCount = defaultdict(int)

        self.wordConceptCounts = defaultdict(int)
        self.lemmaConceptCounts = defaultdict(int)
        self.mleConceptID = defaultdict(int)
        self.lemMLEConceptID = defaultdict(int)
        self.conceptIDDict = defaultdict(int)
        self.unalignedSet = defaultdict(int)

        self.wordIDs = {}
        self.lemmaIDs = {}
        self.conceptIDs = {}
        self.posIDs = {}
        self.depIDs = {}
        self.arcIDs = {}

    def OracleExtraction(self, data_dir, output_dir):
        print "Input data directory:", data_dir
        print "Output directory:", output_dir
        os.system("mkdir -p %s" % output_dir)

        oracle_output = os.path.join(output_dir, "oracle_output.txt")
        json_output = os.path.join(output_dir, "oracle_examples100.json")

        data_set = loadDataset(data_dir)
        oracle_set = OracleData()

        for tree in data_set.dep_trees:
            tree.buildDepDist(tree.dist_threshold)

        data_num = data_set.dataSize()
        data_set.genDictionaries()
        # data_set.genConceptMap()

        # data_set.saveConceptID(output_dir)
        # data_set.saveMLEConceptID(output_dir)

        cache_transition = CacheTransition(
            data_set.known_concepts, data_set.known_rels,
            data_set.unaligned_set, self.cache_size)

        cache_transition.makeTransitions()

        push_actions, arc_binary_actions, arc_label_actions = defaultdict(int), defaultdict(int), defaultdict(int)
        num_pop_actions = 0
        num_shift_actions = 0

        feat_dim = -1

        # Run oracle on the training data.
        with open(oracle_output, 'w') as oracle_wf:
            for sent_idx in range(data_num):
                training_instance = data_set.getInstance(sent_idx)
                tok_seq, lem_seq, pos_seq = training_instance[0], training_instance[1], training_instance[2]
                dep_tree, amr_graph = training_instance[3], training_instance[4]
                amr_graph.initTokens(tok_seq)
                amr_graph.initLemma(lem_seq)

                if sent_idx >= 100:
                    break

                length = len(tok_seq)

                c = CacheConfiguration(self.cache_size, length)
                c.wordSeq, c.lemSeq, c.posSeq = tok_seq, lem_seq, pos_seq
                concept_seq = amr_graph.getConceptSeq()
                c.tree = dep_tree
                c.conceptSeq = concept_seq
                c.setGold(amr_graph)

                feat_seq = []

                word_align = []
                concept_align = []

                # word_idx = 0 # Start processing from leftmost word.

                start_time = time.time()

                oracle_seq = []
                succeed = True

                # oracle_example = OracleExample()
                concept_idx = 0

                while not cache_transition.isTerminal(c):
                    oracle_action = cache_transition.getOracleAction(c)
                    # print oracle_action

                    if time.time() - start_time > 4.0:
                        print >> sys.stderr, "Overtime sentence #%d" % sent_idx
                        print >> sys.stderr, "Sentence: %s" % " ".join(tok_seq)
                        succeed = False
                        break

                    word_idx = c.nextBufferElem()

                    if "ARC" in oracle_action:
                        parts = oracle_action.split(":")
                        arc_decisions = parts[1].split("#")
                        assert len(arc_decisions) == self.cache_size

                        for (cache_idx, arc_label) in enumerate(arc_decisions):
                            curr_arc_action = "ARC%d:%s" % (cache_idx, arc_label)

                            shiftpop_feats = c.shiftPopFeatures()
                            cache_feats = c.getCacheFeat(word_idx, concept_idx, cache_idx)
                            pushidx_feats = c.pushIDXFeatures()

                            all_feats = shiftpop_feats + cache_feats + pushidx_feats

                            if arc_label == "O":
                                arc_binary_actions["O"] += 1
                                feat_seq.append(["PHASE=ARCBINARY"]+ all_feats)
                                oracle_seq.append("NOARC")
                                word_align.append(word_idx)
                                concept_align.append(concept_idx)
                            else:
                                arc_binary_actions["Y"] += 1
                                feat_seq.append(["PHASE=ARCBINARY"]+ all_feats)
                                oracle_seq.append("ARC")
                                word_align.append(word_idx)
                                concept_align.append(concept_idx)

                                feat_seq.append(["PHASE=ARCLABEL"]+ all_feats)
                                arc_label_actions[arc_label] += 1
                                oracle_seq.append(arc_label)
                                word_align.append(word_idx)
                                concept_align.append(concept_idx)
                                # print curr_arc_action
                                # print word_idx, concept_idx, cache_idx
                                # print feat_seq[-1]
                            cache_transition.apply(c, curr_arc_action)


                    else:
                        #Currently assume vertex generated separately.
                        if "conGen" not in oracle_action and "conID" not in oracle_action:
                            oracle_seq.append(oracle_action)
                            word_align.append(word_idx)
                            concept_align.append(concept_idx)
                            cache_feats = c.getCacheFeat()
                            if oracle_action == "POP":
                                shiftpop_feats = c.shiftPopFeatures(word_idx, concept_idx, True)
                                pushidx_feats = c.pushIDXFeatures()
                                phase = "PHASE=SHTPOP"
                                num_pop_actions += 1
                            else:
                                phase = "PHASE=PUSHIDX"
                                push_actions[oracle_action] += 1
                                concept_idx += 1
                                if concept_idx == len(concept_seq):
                                    concept_idx = -1
                            all_feats = [phase] + shiftpop_feats + cache_feats + pushidx_feats
                            feat_seq.append(all_feats)
                        elif "NULL" not in oracle_action:
                            #if "PUSH" in oracle_action:
                            oracle_seq.append("SHIFT")
                            num_shift_actions += 1
                            word_align.append(word_idx)
                            concept_align.append(concept_idx)
                            shiftpop_feats = c.shiftPopFeatures(word_idx, concept_idx, True)
                            cache_feats = c.getCacheFeat()
                            pushidx_feats = c.pushIDXFeatures()
                            all_feats = shiftpop_feats + cache_feats + pushidx_feats
                            feat_seq.append(["PHASE=SHTPOP"] + all_feats)
                            if feat_dim == -1:
                                feat_dim = len(feat_seq[-1])
                        cache_transition.apply(c, oracle_action)

                if succeed:
                    assert len(feat_seq) == len(oracle_seq)
                    for feats in feat_seq:
                        assert len(feats) == feat_dim, "Feature dimensions not consistent: %s" % str(feats)
                    oracle_example = OracleExample(tok_seq, lem_seq, pos_seq, concept_seq, feat_seq, oracle_seq,
                                                   word_align, concept_align)
                    oracle_set.addExample(oracle_example)
                    print>> oracle_wf, "Sentence #%d: %s" % (sent_idx, " ".join(tok_seq))
                    print>> oracle_wf, "AMR graph:\n%s" %  str(amr_graph).strip()
                    print>> oracle_wf, "Constructed AMR graph:\n%s" % str(c.hypothesis).strip()
                    print>> oracle_wf, "Oracle sequence: %s" % " ".join(oracle_seq)
                    print>> oracle_wf, "Oracle feature sequence: %s" % " ".join(["#".join(feats) for feats
                                                                                   in feat_seq])
                    print>> oracle_wf, "Oracle word alignments: %s" % str(word_align)
                    print>> oracle_wf, "Oracle concept alignments: %s\n" % str(concept_align)

                    if not c.gold.compare(c.hypothesis):
                        print "Oracle sequence not constructing the gold graph for sentence %d!" % sent_idx
                        print " ".join(tok_seq)
                        print str(amr_graph)
                    else:
                        print "check okay for sentence %d" % sent_idx
                else:
                    print "Failed sentence %d" % sent_idx
                    print " ".join(tok_seq)
                    print str(amr_graph)
                    print "Oracle sequence so far: %s\n" % " ".join(oracle_seq)



            oracle_wf.close()

        saveCounter(arc_binary_actions, "arc_binary_actions.txt")
        saveCounter(arc_label_actions, "arc_label_actions.txt")
        saveCounter(push_actions, "pushidx_actions.txt")
        print "A total of %d shift actions" % num_shift_actions
        print "A total of %d pop actions" % num_pop_actions
        oracle_set.toJSON(json_output)

    def isPredicate(self, s):
        length = len(s)
        if length < 3 or not (s[length-3] == '-'):
            return False
        last_char = s[-1]
        return last_char >= '0' and last_char <= '9'

    def conceptCategory(self, s, conceptArcChoices):
        """
        To be implemented!
        :param s:
        :param conceptArcChoices:
        :return:
        """
        if s in conceptArcChoices:
            return s
        if s == "NE" or "NE_" in s:
            return "NE"
        return "OTHER"

    def getWordID(self, s):
        if s in self.wordIDs:
            return self.wordIDs[s]
        return UNKNOWN

    def getLemmaID(self, s):
        if s in self.lemmaIDs:
            return self.lemmaIDs[s]
        return UNKNOWN

    def getConceptID(self, s):
        if s in self.conceptIDs:
            return self.conceptIDs[s]
        return UNKNOWN

    def getPOSID(self, s):
        if s in self.posIDs:
            return self.posIDs[s]
        return UNKNOWN

    def getDepID(self, s):
        if s in self.depIDs:
            return self.depIDs[s]
        return UNKNOWN

    def getArcID(self, s):
        if s in self.arcIDs:
            return self.arcIDs[s]
        return UNKNOWN

    def actionType(self, s):
        if s == "POP" or "conID" in s or "conGen" in s or "conEMP" in s:
            return 0
        if "ARC" in s:
            return 1
        else:
            return 2

    def generateTrainingExamples(self):
        return


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--data_dir", type=str, help="The data directory for the input files.")
    argparser.add_argument("--output_dir", type=str, help="The directory for the output files.")
    argparser.add_argument("--cache_size", type=int, default=6, help="Fixed cache size for the transition system.")

    args = argparser.parse_args()
    parser = CacheTransitionParser(args.cache_size)
    parser.OracleExtraction(args.data_dir, args.output_dir)