import json
class OracleExample(object):
    def __init__(self, toks_=None, lems_=None, poss_=None, concepts_=None,
                 feats_=None, actions_=None, w_align_=None, c_align_=None):
        self.toks, self.lems, self.poss, self.concepts = toks_, lems_, poss_, concepts_
        self.feats, self.actions, self.word_align, self.concept_align = feats_, actions_, w_align_, c_align_

    def toJSONobj(self):
        json_obj = {}
        json_obj["text"] = " ".join(self.toks)
        lem_str = " ".join(self.lems)
        pos_str = " ".join(self.poss)
        concept_str = " ".join(self.concepts)
        json_obj["concepts"] = concept_str
        json_obj["annotation"] =  {"lemmas": lem_str, "POSs": pos_str}

        word_align_str = " ".join([str(idx) for idx in self.word_align])
        concept_align_str = " ".join([str(idx) for idx in self.concept_align])
        feats_str = " ".join(["#".join(feat) for feat in self.feats])
        json_obj["actionseq"] = " ".join(self.actions)
        json_obj["alignment"] = {"word-align": word_align_str, "concept-align": concept_align_str}
        json_obj["feats"] = feats_str
        return json_obj

class OracleData(object):
    def __init__(self):
        self.examples = []
        self.num_examples = 0

    def addExample(self, example):
        self.examples.append(example)
        self.num_examples += 1

    def toJSON(self, json_output):
        dataset = []
        for idx in range(self.num_examples):
            curr_json_example = self.examples[idx].toJSONobj()
            dataset.append(curr_json_example)

        with open(json_output, "w") as json_wf:
            json.dump(dataset, json_wf)
            json_wf.close()
