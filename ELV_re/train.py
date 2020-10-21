from torch.utils.data import DataLoader
import torch
import logging
from transformers import AdamW
from s2s_ft import utils
import os
from sklearn import metrics
from s2s_ft.modeling_decoding import BertForSeq2SeqDecoder, BertConfig
import s2s_ft.s2s_loader as seq2seq_loader
import numpy as np
logger = logging.getLogger(__name__)
import tqdm

def train_generator(generator, train_iter, optimizer, retriever, args, iters):
    logging.info('Train generator')
    global_step = 0
    logging_loss = 0.0
    for epoch in range(args.num_generator_epochs_per_iters):
        for step, batch in enumerate(train_iter):
            inputs = {'source_ids': batch[0].long().to(args.device),
                      'target_ids': batch[1].long().to(args.device),
                      'pseudo_ids': batch[2].long().to(args.device),
                      'num_source_tokens': batch[3].long().to(args.device),
                      'num_target_tokens': batch[4].long().to(args.device)
                      }
            loss = generator(**inputs)
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            logging_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                generator.zero_grad()
            global_step += 1
            optimizer.step()
            generator.zero_grad()
        logger.info("Training Generator Iters [%d] Epoch [%d] Step [%d]: %.2f", iters, epoch, global_step, logging_loss)
        logging_loss = 0.0

    save_path = os.path.join(args.output_dir, "train_generator_{}".format(iters))
    args.generator_path = save_path
    os.makedirs(save_path, exist_ok=True)
    model_to_save = generator.module if hasattr(generator, "module") else generator
    model_to_save.save_pretrained(save_path)
    logging.info('Train generator finished')


def train_classifier(classifier, train_iter, valid_dataset, optimizer, tokenizer, retriever, args, iters):
    logging.info('Train classifier')
    classifier.train()
    classifier.zero_grad()
    logging_loss = 0.0
    global_step = 0
    for epoch in range(args.num_classifier_epochs_per_iters):
        for step, (inputs, labels, pos) in enumerate(train_iter):
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            pos = pos.to(args.device)
            loss = classifier(inputs, labels=labels, entity_pos=pos)
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            logging_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                classifier.zero_grad()
            if (step + 1) % args.logging_steps == 0:
                logger.info("Training Classifier Iters [%d] Epoch [%d] Step [%d] - Step [%d]: %.2f", iters, epoch, global_step - args.logging_steps, global_step, logging_loss)
                logging_loss = 0.0
            global_step += 1
        logging_loss = 0.0
        precision, recall, f1_score = valid_classifier(classifier, valid_dataset, tokenizer, retriever, args, iters)
        logger.info("Valid Iters [%d] Epoch [%d] precision: %.5f, recall: %.5f, f1: %.5f", iters, epoch, precision, recall, f1_score)
    save_path = os.path.join(args.output_dir, "training_classifier")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    model_to_save = classifier.module if hasattr(classifier, "module") else classifier
    torch.save(model_to_save.state_dict(), save_path + '/training_classifier_{}.pt'.format(iters))


def prepare_for_training(args, model):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    else:
        amp = None

    if amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    return model, optimizer


def valid_generator(valid_dataset, tokenizer, args, iters, test=False):
    input_lines = []
    if args.num_labels == 3:
        sentence_list, label_list = valid_dataset.sentence_list, valid_dataset.label_list
        for sen, label in zip(sentence_list, label_list):
            sep_token = tokenizer.sep_token
            label = 'positive' if label == 1 else 'negative'
            src = (sen + sep_token + label)
            src = tokenizer.tokenize(src)
            input_lines.append(src)
    else:
        sentence_list, rel_list = valid_dataset.sentence_list, valid_dataset.rel_list
        for sen, rel in zip(sentence_list, rel_list):
            sep_token = tokenizer.sep_token
            src = (sen + sep_token + rel)
            src = tokenizer.tokenize(src)
            input_lines.append(src)

    config_file = args.config_name if args.config_name else os.path.join(args.generator_path, "config.json")
    config = BertConfig.from_json_file(config_file)

    mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
        [tokenizer.mask_token, tokenizer.sep_token, tokenizer.sep_token])

    generator = BertForSeq2SeqDecoder.from_pretrained(
        args.generator_path, config=config, mask_word_id=mask_word_id, 
        eos_id=eos_word_ids, sos_id=sos_word_id)
    generator.to(args.device)

    preprocessor = seq2seq_loader.Preprocess4Seq2seqDecoder(
        list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_source_seq_length,
        max_tgt_length=args.max_target_seq_length, pos_shift=args.pos_shift,
        source_type_id=config.source_type_id, target_type_id=config.target_type_id,
        cls_token=tokenizer.cls_token, sep_token=tokenizer.sep_token,
        pad_token=tokenizer.pad_token)

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
                    output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                    output_tokens = []
                    for t in output_buf:
                        if t in (tokenizer.sep_token, tokenizer.pad_token):
                            break
                        output_tokens.append(t)
                    output_sequence = ' '.join(detokenize(output_tokens))
                    if '\n' in output_sequence:
                        output_sequence = " [X_SEP] ".join(output_sequence.split('\n'))
                    output_lines[buf_id[i]] = output_sequence
    save_path = os.path.join(args.output_dir, "output_generation")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if test:
        filename = '/test_explanation_{}.txt'.format(iters)
    else:
        filename = '/valid_explanation_{}.txt'.format(iters)
    with open(save_path + filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))


def valid_classifier(classifier, valid_dataset, tokenizer, retriever, args, iters, test=False):
    classifier.eval()
    pre_rel_list = []
    gd_rel_list = []
    valid_classifier_dataset = valid_dataset.get_valid(retriever, test=test)
    valid_iter = DataLoader(valid_classifier_dataset, batch_size=args.batch_size, shuffle=False)
    logger.info("Validating...")
    with torch.no_grad():
        for inputs, labels, pos in tqdm.tqdm(valid_iter):
            inputs = inputs.to(args.device)
            pos = pos.to(args.device)
            logits = classifier(inputs, entity_pos=pos)
            pre_rel = torch.argmax(logits, dim=-1)
            pre_rel_list.extend(pre_rel.tolist())
            gd_rel_list.extend(labels.tolist())

    if args.num_labels == 3:
        f1_score = metrics.f1_score(gd_rel_list, pre_rel_list, average='macro')
        precision = metrics.precision_score(gd_rel_list, pre_rel_list, average='macro')
        recall = metrics.recall_score(gd_rel_list, pre_rel_list, average='macro')
    else:
        labels = list(range(1, args.num_labels))
        f1_score = metrics.f1_score(gd_rel_list, pre_rel_list, labels=labels, average='micro')
        precision = metrics.precision_score(gd_rel_list, pre_rel_list, labels=labels, average='micro')
        recall = metrics.recall_score(gd_rel_list, pre_rel_list, labels=labels, average='micro')
    if test:
        logger.info("Test Iters [%d] precision: %.5f, recall: %.5f, f1: %.5f", iters, precision, recall, f1_score)
    return precision, recall, f1_score


def train(generator, classifier, classifier_tokenizer, generator_tokenizer, unlabeled_dataset, generator_dataset, classifier_dataset, valid_dataset, test_dataset, retriever, args):
    classifier, classifier_optimizer = prepare_for_training(args, classifier)
    generator, generator_optimizer = prepare_for_training(args, generator)

    precision_list_valid = []
    recall_list_valid = []
    f1_list_valid = []
    precision_list_test = []
    recall_list_test = []
    f1_list_test = []

    generator_iter = DataLoader(generator_dataset, batch_size=args.batch_size, shuffle=True)
    train_generator(generator, generator_iter, generator_optimizer, retriever, args, -1)

    for iters in range(args.num_iters):
        classifier_dataset = unlabeled_dataset.get_classifier_dataset(args, generator, retriever, classifier_dataset, iters)
        classifier_iter = DataLoader(classifier_dataset, batch_size=args.batch_size, shuffle=True)
        train_classifier(classifier, classifier_iter, valid_dataset, classifier_optimizer, classifier_tokenizer, retriever, args, iters)
        precision, recall, f1_score = valid_classifier(classifier, valid_dataset, classifier_tokenizer, retriever, args, iters)
        precision_list_valid.append(precision)
        recall_list_valid.append(recall)
        f1_list_valid.append(f1_score)
        precision, recall, f1_score = valid_classifier(classifier, test_dataset, classifier_tokenizer, retriever, args, iters, test=True)
        precision_list_test.append(precision)
        recall_list_test.append(recall)
        f1_list_test.append(f1_score)

        generator_dataset = unlabeled_dataset.get_generator_dataset(args, classifier, retriever, generator_dataset, iters)
        generator_iter = DataLoader(generator_dataset, batch_size=args.batch_size, shuffle=True)
        train_generator(generator, generator_iter, generator_optimizer, retriever, args, iters)
        # valid_generator(valid_dataset, generator_tokenizer, args, iters)
        # valid_generator(test_dataset, generator_tokenizer, args, iters, test=True)

        save_path = os.path.join(args.output_dir, "metrics_valid")
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(save_path + '/precision.npy', np.array(precision_list_valid))
        np.save(save_path + '/recall.npy', np.array(recall_list_valid))
        np.save(save_path + '/f1.npy', np.array(f1_list_valid))

       save_path = os.path.join(args.output_dir, "metrics_test_{}".format(iters))

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        np.save(save_path + '/precision.npy', np.array(precision_list_test))
        np.save(save_path + '/recall.npy', np.array(recall_list_test))
        np.save(save_path + '/f1.npy', np.array(f1_list_test))