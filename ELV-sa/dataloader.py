import json
from torch.utils.data import Dataset
import torch
from keras.preprocessing.sequence import pad_sequences
import random
from s2s_ft.modeling_decoding import BertForSeq2SeqDecoder, BertConfig
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import s2s_ft.s2s_loader as seq2seq_loader
import os
import tqdm
import numpy as np


class UnlabledDataset(Dataset):
    def __init__(self, tokenizer, device, percentage=100):
        self.tokenizer = tokenizer
        self.device = device
        self.percentage = percentage

    def tokenize(self, sentence, candidate_exp):
        sentence_list = [sentence + self.tokenizer.sep_token + exp for exp in candidate_exp]
        tokens = [self.tokenizer.encode(sen) for sen in sentence_list]
        return tokens

    def load_init(self, filename, label_to_id=None):
        self.sentence_list = []
        self.label_list = []
        self.rel_list = []
        label2id = {}
        if label_to_id is not None:
            with open(label_to_id) as f:
                label2id = json.load(f)
        with open(filename) as f:
            json_file = json.load(f)
            for line in json_file:
                if 'term' in line:
                    term = line['term']
                    sent = line['sent']
                    self.sentence_list.append(term + self.tokenizer.sep_token + sent)
                    label = line['label']
                    if label == -1:
                        label = 2
                    self.label_list.append(label)
                else:
                    assert label_to_id is not None
                    sent = line['sent']
                    self.sentence_list.append(sent)
                    rel = label2id[line['rel']]
                    self.rel_list.append(line['rel'])
                    self.label_list.append(rel)

    def get_valid(self, retriever):
        classifier_dataset = ClassifierDataset(self.tokenizer, retriever)
        classifier_dataset.update_unlabeled(self.sentence_list, self.label_list)
        return classifier_dataset

    def get_generator_dataset(self, args, classifier, retriever, generator_dataset, iters):
        retrieve_exp_list, similarity_list = [], []
        for sen in self.sentence_list:
            exp, similar = retriever.retrieve(sen, 10, get_similar=True)
            retrieve_exp_list.append(exp)
            similarity_list.append(similar)
        ground_truth_list = []
        for sen, retrieve_exp, similarity, label in zip(self.sentence_list, retrieve_exp_list, similarity_list, self.label_list):
            tokens = [sen + self.tokenizer.sep_token + exp for exp in retrieve_exp]
            tokens = [torch.tensor(self.tokenizer.encode(tok)) for tok in tokens]
            tokens = pad_sequence(tokens, padding_value=self.tokenizer.pad_token_id, batch_first=True).to(args.device)
            tokens = tokens.unsqueeze(1)
            logits = classifier(tokens)
            prob = logits[:, label]
            similarity = torch.tensor(similarity).float().to(args.device)
            score = similarity * prob
            index = torch.multinomial(F.softmax(score), num_samples=1)
            ground_truth = retrieve_exp[index.item()]
            ground_truth_list.append(ground_truth)
        generator_dataset.update_unlabeled(self.sentence_list, self.label_list, ground_truth_list)
        return generator_dataset

    def get_classifier_dataset(self, args, generator, retriever, classifier_dataset, iters):
        input_lines = []
        if args.num_labels == 3:
            for sen, label in zip(self.sentence_list, self.label_list):
                label = 'positive' if label == 1 else 'negative'
                src = (sen + self.tokenizer.sep_token + label)
                src = self.tokenizer.tokenize(src)
                input_lines.append(src)
        else:
            for sen, label in zip(self.sentence_list, self.rel_list):
                src = sen + self.tokenizer.sep_token + label
                src = self.tokenizer.tokenize(src)
                input_lines.append(src)

        config_file = args.config_name if args.config_name else os.path.join(args.generator_path, "config.json")
        config = BertConfig.from_json_file(config_file)

        mask_word_id, eos_word_ids, sos_word_id = self.tokenizer.convert_tokens_to_ids(
            [self.tokenizer.mask_token, self.tokenizer.sep_token, self.tokenizer.sep_token])

        generator = BertForSeq2SeqDecoder.from_pretrained(
            args.generator_path, config=config, mask_word_id=mask_word_id,
            eos_id=eos_word_ids, sos_id=sos_word_id)
        generator.to(args.device)

        preprocessor = seq2seq_loader.Preprocess4Seq2seqDecoder(
            list(self.tokenizer.vocab.keys()), self.tokenizer.convert_tokens_to_ids, pos_shift=args.pos_shift,
            source_type_id=config.source_type_id, target_type_id=config.target_type_id,
            cls_token=self.tokenizer.cls_token, sep_token=self.tokenizer.sep_token,
            pad_token=self.tokenizer.pad_token)

        input_lines = list(enumerate(input_lines))
        output_lines = [""] * len(input_lines)
        next_i = 0
        with torch.no_grad():
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size * args.gradient_accumulation_steps]
                buf_id = [x[0] for x in _chunk]
                buf = [x[1] for x in _chunk]
                next_i += args.batch_size * args.gradient_accumulation_steps
                max_a_len = max([len(x) for x in buf])
                instances = []
                for instance in [(x, max_a_len) for x in buf]:
                    instances.append(preprocessor(instance))

                with torch.no_grad():
                    batch = seq2seq_loader.batch_list_to_batch_tensors(
                        instances)
                    batch = [t.to(args.device) if t is not None else None for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
                    traces = generator(input_ids, token_type_ids,
                                       position_ids, input_mask, task_idx=task_idx, mask_qkv=mask_qkv)
                    if args.beam_size > 1:
                        traces = {k: v.tolist() for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces.tolist()

                    def detokenize(tk_list):
                        r_list = []
                        for tk in tk_list:
                            if tk.startswith('##') and len(r_list) > 0:
                                r_list[-1] = r_list[-1] + tk[2:]
                            else:
                                r_list.append(tk)
                        return r_list

                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = self.tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in (self.tokenizer.sep_token, self.tokenizer.pad_token):
                                break
                            output_tokens.append(t)
                        output_sequence = ' '.join(detokenize(output_tokens))
                        if '\n' in output_sequence:
                            output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))
                        output_lines[buf_id[i]] = output_sequence

        save_path = os.path.join(args.output_dir, "output_generation")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        filename = '/train_explanation_{}.txt'.format(iters)
        with open(save_path + filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))

        retriever.update_exp(self.sentence_list, output_lines)
        classifier_dataset.update_unlabeled(self.sentence_list, self.label_list)
        return classifier_dataset

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, idx):
        return self.sentence_list[idx], self.label_list[idx]


class GeneratorDataset(Dataset):
    def __init__(self, tokenizer, args):
        self.max_source_len = args.max_source_seq_length
        self.max_target_len = args.max_target_seq_length
        self.tokenizer = tokenizer
        self.random_prob = args.random_prob
        self.keep_prob = args.keep_prob
        self.vocab_size = tokenizer.vocab_size
        self.pad_id = tokenizer.pad_token_id
        self.sep_id = tokenizer.sep_token_id
        self.cls_id = tokenizer.cls_token_id
        self.mask_id = tokenizer.mask_token_id

    def load_init(self, filename, label_to_id=None):
        self.sentence_list = []
        self.label_list = []
        label2id = {}
        if label_to_id is not None:
            with open(label_to_id) as f:
                label2id = json.load(f)
        with open(filename) as f:
            json_file = json.load(f)
            sentence_list = []
            exp_list = []
            labels = []
            rel_list = []
            for item in json_file:
                if 'term' in item:
                    term = item['term']
                    sent = item['sent']
                    exp = item['exp']
                    label = item['label'] if 'label' in item else -100
                    if label == -1:
                        label = 2
                    sentence_list.append(term + self.tokenizer.sep_token + sent)
                    exp_list.append(exp)
                    labels.append(label)
                else:
                    assert label_to_id is not None
                    sent = item['sent']
                    sentence_list.append(sent)
                    exp = item['exp']
                    exp_list.append(exp)
                    rel = label2id[item['rel']]
                    rel_list.append(item['rel'])
                    labels.append(rel)

        result = self.__process(sentence_list, labels, exp_list, rel_list)
        self.labeled_source_ids, self.labeled_target_ids, self.labeled_pseudo_ids, \
            self.labeled_num_source_tokens, self.labeled_num_target_tokens = result

        self.source_ids = torch.tensor(pad_sequences(self.labeled_source_ids, value=self.pad_id, padding='post', truncating='post', maxlen=self.max_source_len))
        self.target_ids = torch.tensor(pad_sequences(self.labeled_target_ids, value=self.pad_id, padding='post', truncating='post', maxlen=self.max_target_len))
        self.pseudo_ids = torch.tensor(pad_sequences(self.labeled_pseudo_ids, value=self.pad_id, padding='post', truncating='post', maxlen=self.max_target_len))
        self.num_source_tokens = self.labeled_num_source_tokens
        self.num_target_tokens = self.labeled_num_target_tokens

    def __process(self, sentence_list, label_list, exp_list, rel_list=[]):
        features_list = []
        for i, (sen, label, exp) in enumerate(zip(sentence_list, label_list, exp_list)):
            sep_token = self.tokenizer.sep_token
            if rel_list == []:
                label = 'positive' if label == 1 else 'negtive'
            else:
                label = rel_list[i]
            src = sen + sep_token + label
            source_tokens = self.tokenizer.tokenize(src)
            target_tokens = self.tokenizer.tokenize(exp)
            features_list.append({
                "source_ids": self.tokenizer.convert_tokens_to_ids(source_tokens),
                "target_ids": self.tokenizer.convert_tokens_to_ids(target_tokens),
            })

        source_ids_list = []
        target_ids_list = []
        pseudo_ids_list = []
        num_source_tokens = []
        num_target_tokens = []
        for feature in features_list:
            source_ids = self.__trunk([self.cls_id] + feature["source_ids"], self.max_source_len)
            target_ids = self.__trunk(feature["target_ids"], self.max_target_len)
            pseudo_ids = []
            for tk_id in target_ids:
                p = random.random()
                if p < self.keep_prob:
                    pseudo_ids.append(tk_id)
                elif p < self.keep_prob + self.random_prob:
                    pseudo_ids.append(random.randint(0, self.vocab_size - 1))
                else:
                    pseudo_ids.append(self.mask_id)

            source_ids_list.append(source_ids)
            target_ids_list.append(target_ids)
            pseudo_ids_list.append(pseudo_ids)
            num_source_tokens.append(len(source_ids))
            num_target_tokens.append(len(target_ids))

        return source_ids_list, target_ids_list, pseudo_ids_list, num_source_tokens, num_target_tokens

    def update_unlabeled(self, sentence_list, label_list, exp_list):
        result = self.__process(sentence_list, label_list, exp_list)
        self.unlabeled_source_ids, self.unlabeled_target_ids, self.unlabeled_pseudo_ids, \
            self.unlabeled_num_source_tokens, self.unlabeled_num_target_tokens = result

        self.num_source_tokens = self.unlabeled_num_source_tokens + self.labeled_num_source_tokens
        self.num_target_tokens = self.unlabeled_num_target_tokens + self.labeled_num_target_tokens

        source_ids = self.unlabeled_source_ids + self.labeled_source_ids
        target_ids = self.unlabeled_target_ids + self.labeled_target_ids
        pseudo_ids = self.unlabeled_pseudo_ids + self.labeled_pseudo_ids

        self.source_ids = torch.tensor(pad_sequences(source_ids, value=self.pad_id, padding='post', truncating='post', maxlen=self.max_source_len))
        self.target_ids = torch.tensor(pad_sequences(target_ids, value=self.pad_id, padding='post', truncating='post', maxlen=self.max_target_len))
        self.pseudo_ids = torch.tensor(pad_sequences(pseudo_ids, value=self.pad_id, padding='post', truncating='post', maxlen=self.max_target_len))

    def __len__(self):
        return len(self.source_ids)

    def __trunk(self, ids, max_len):
        if len(ids) > max_len - 1:
            ids = ids[:max_len - 1]
        ids = ids + [self.sep_id]
        return ids

    def __pad(self, ids, max_len):
        if len(ids) < max_len:
            return ids + [self.pad_id] * (max_len - len(ids))
        else:
            assert len(ids) == max_len
            return ids

    def __getitem__(self, idx):
        return self.source_ids[idx], self.target_ids[idx], self.pseudo_ids[idx], self.num_source_tokens[idx], self.num_target_tokens[idx]


class ClassifierDataset(Dataset):
    def __init__(self, tokenizer, retriever):
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.labeled_sentence_list = []
        self.labeled_label = []

    def load_init(self, filename, label_to_id=None):
        label2id = {}
        if label_to_id is not None:
            with open(label_to_id) as f:
                label2id = json.load(f)
        with open(filename) as f:
            json_file = json.load(f)
            sentence_list = []
            exp_list = []
            labels = []
            rel_list = []
            for item in json_file:
                if 'term' in item:
                    term = item['term']
                    sent = item['sent']
                    exp = item['exp']
                    label = item['label'] if 'label' in item else -100
                    if label == -1:
                        label = 2
                    sentence_list.append(term + self.tokenizer.sep_token + sent)
                    exp_list.append(exp)
                    labels.append(label)
                else:
                    assert label_to_id is not None
                    sent = item['sent']
                    sentence_list.append(sent)
                    exp = item['exp']
                    exp_list.append(exp)
                    rel = label2id[item['rel']]
                    rel_list.append(item['rel'])
                    labels.append(rel)

        self.labeled_sentence_list = sentence_list
        self.labeled_label = labels

    def update_unlabeled(self, sentence_list, label_list, nums=10):
        self.labels = self.labeled_label + label_list
        self.sentence_list = self.labeled_sentence_list + sentence_list
        candidate_exp_list = [self.retriever.retrieve(sen, nums) for sen in self.sentence_list]

        all_samples = []
        for sen, exp in zip(self.sentence_list, candidate_exp_list):
            all_samples.extend(self.tokenize(sen, exp))

        pad_index = self.tokenizer.pad_token_id
        tokens = pad_sequence([torch.tensor(x) for x in all_samples], padding_value=pad_index, batch_first=True)

        self.samples = []
        for i in range(0, len(tokens), 10):
            self.samples.append(torch.tensor(tokens[i:i + 10]))

    def tokenize(self, sentence, candidate_exp):
        sentence_list = [sentence + self.tokenizer.sep_token + exp for exp in candidate_exp]
        tokens = [self.tokenizer.encode(sen)for sen in sentence_list]
        return tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]
