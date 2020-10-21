import json
from sentence_transformers import SentenceTransformer
import numpy as np
from annoy import AnnoyIndex


class Retriever(object):
    def __init__(self, args, tokenizer, hidden_size=768):
        self.embedmodel = SentenceTransformer('bert-base-nli-mean-tokens')
        self.tokenizer = tokenizer
        self.embed_list = []
        self.sentence_list = []
        self.sentence2exp = {}
        self.annoy = AnnoyIndex(hidden_size, metric='angular')
        self.load_exp(args.labeled_data)
        self.loaded = False
        # self.load_unlabeled_sen(args.unlabeled_data)

    def load_exp(self, filename):
        sentence_list = []
        with open(filename, encoding='utf-8') as f:
            json_file = json.load(f)
            for item in json_file:
                if 'term' in item:
                    sentence = item['term'] + self.tokenizer.sep_token + item['sent']
                else:
                    sentence = item['sent']
                exp = item['exp']
                sentence_list.append(sentence)
                self.sentence2exp[sentence] = exp
        self.embed_list.extend(self.embedmodel.encode(sentence_list, show_progress_bar=True))
        for i, embed in enumerate(self.embed_list):
            self.annoy.add_item(i, embed)
        self.sentence_list.extend(sentence_list)

    def load_unlabeled_sen(self, filename):
        sentence_list = []
        with open(filename, encoding='utf-8') as f:
            json_file = json.load(f)
            for item in json_file:
                if 'term' in item:
                    sentence = item['term'] + self.tokenizer.sep_token + item['sent']
                else:
                    sentence = item['sent']
                sentence_list.append(sentence)
        self.sentence_list.extend(sentence_list)
        self.embed_list.extend(self.embedmodel.encode(sentence_list, show_progress_bar=True))
        for i, embed in enumerate(self.embed_list):
            self.annoy.add_item(i, embed)
        self.loaded = True

    def update_exp(self, sentence_list, exp_list):
        for sen, exp in zip(sentence_list, exp_list):
            self.sentence2exp[sen] = exp

    def retrieve(self, sentence, nums, get_similar=False):
        if self.annoy.get_n_trees() == 0:
            self.annoy.build(100)

        if sentence in self.sentence_list:
            index = self.sentence_list.index(sentence)
            result = self.annoy.get_nns_by_item(index, nums, include_distances=get_similar)
        else:
            target_embed = self.embedmodel.encode([sentence], show_progress_bar=False)[0]
            result = self.annoy.get_nns_by_vector(target_embed, nums, include_distances=get_similar)

        if get_similar:
            candidate_exp = [self.sentence2exp[self.sentence_list[index]] for index in result[0]]
            return candidate_exp, result[1]
        else:
            candidate_exp = [self.sentence2exp[self.sentence_list[index]] for index in result]
            return candidate_exp

    def compute_sim(self, embed1, embed2):
        return np.dot(embed1, embed2)
