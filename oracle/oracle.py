from collections import defaultdict
from ioutil import *
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

        data_set = loadDataset(data_dir)

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

        # Run oracle on the training data.
        with open(oracle_output, 'w') as oracle_wf:
            for sent_idx in range(data_num):
                training_instance = data_set.getInstance(sent_idx)
                tok_seq, lem_seq, pos_seq = training_instance[0], training_instance[1], training_instance[2]
                dep_tree, amr_graph = training_instance[3], training_instance[4]
                amr_graph.initTokens(tok_seq)
                amr_graph.initLemma(lem_seq)

                length = len(tok_seq)

                c = CacheConfiguration(self.cache_size, length)
                c.wordSeq, c.lemSeq, c.posSeq = tok_seq, lem_seq, pos_seq
                c.tree = dep_tree
                c.setGold(amr_graph)

                word_idx = 0 # Start processing from leftmost word.

                start_time = time.time()

                oracle_seq = []
                succeed = True

                while not cache_transition.isTerminal(c):
                    oracle_action = cache_transition.getOracleAction(c, word_idx)
                    # print "Current action:", oracle_action

                    # If the next action processes a buffer word.
                    if "conID" in oracle_action or "conEMP" in oracle_action:
                        word_idx += 1

                    if time.time() - start_time > 1.0:
                        print >> sys.stderr, "Overtime sentence #%d" % sent_idx
                        print >> sys.stderr, "Sentence: %s" % " ".join(tok_seq)
                        succeed = False
                        break

                    oracle_seq.append(oracle_action)

                    if "ARC" in oracle_action:
                        parts = oracle_action.split(":")
                        arc_decisions = parts[1].split("#")
                        assert len(arc_decisions) == self.cache_size

                        for (cache_idx, arc_label) in enumerate(arc_decisions):
                            curr_arc_action = "ARC%d:%s" % (cache_idx, arc_label)
                            cache_transition.apply(c, curr_arc_action)
                    else:
                        cache_transition.apply(c, oracle_action)

                if succeed:
                    print>> oracle_wf, "Sentence #%d: %s" % (sent_idx, " ".join(tok_seq))
                    print>> oracle_wf, "AMR graph:\n%s" %  str(amr_graph)
                    print>> oracle_wf, "Oracle sequence: %s" % " ".join(oracle_seq)

                if sent_idx > 100:
                    break
            oracle_wf.close()

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
    parser = CacheTransitionParser(6)
    parser.OracleExtraction(args.data_dir, args.output_dir)