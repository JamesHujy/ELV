import tqdm
import argparse
import logging
import os
import json
import torch
import random
import numpy as np

from classifier.classifier_bert import Classifier
from s2s_ft.modeling import BertForSequenceToSequence

from retriever import Retriever

from s2s_ft.configuration_unilm import UnilmConfig
from s2s_ft.tokenization_unilm import UnilmTokenizer

from s2s_ft.config import BertForSeq2SeqConfig
from train import train
from dataloader import UnlabledDataset, ClassifierDataset, GeneratorDataset

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--train_source_file", default=None, type=str, required=True,
    #                     help="Training data contains source")
    # parser.add_argument("--train_target_file", default=None, type=str, required=True,
    #                     help="Training data contains target")
    parser.add_argument("--labeled_data", default=None, type=str, required=True,
                        help="Labeled training data (json format) for training.")
    parser.add_argument("--unlabeled_data", default=None, type=str, required=True,
                        help="Unlabeled training data (json format) for training.")
    parser.add_argument("--valid_data", default=None, type=str, required=True,
                        help="Valid data (json format) for training.")
    parser.add_argument("--test_data", default=None, type=str, required=True,
                        help="test data (json format) for training.")
    parser.add_argument("--num_labels", default=2, type=int, required=True,
                        help="Lebels to be classified")
    parser.add_argument("--label_to_id", default=None, type=str,
                        help="label_to_id files for RE task")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    parser.add_argument("--log_dir", default=None, type=str,
                        help="The output directory where the log will be written.")

    # Other parameters
    parser.add_argument("--labeled_data_percent", default=100, required=False, type=int,
                        help="Choose the percentage of labeled data")
    parser.add_argument("--config_name", default=None, type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default=None, type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default=None, type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument("--max_source_seq_length", default=128, type=int,
                        help="The maximum total source sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_target_seq_length", default=64, type=int,
                        help="The maximum total target sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")

    parser.add_argument("--cached_train_features_file", default=None, type=str,
                        help="Cached training features file")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--label_smoothing", default=0.1, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_training_steps", default=-1, type=int,
                        help="set total number of training steps to perform")
    parser.add_argument('--num_classifier_epochs_per_iters', type=int, default=10,
                        help="Training epochs when initlizing.")
    parser.add_argument('--num_generator_epochs_per_iters', type=int, default=10,
                        help="Training epochs when initlizing.")
    parser.add_argument('--num_iters', type=int, default=10,
                        help="Training epochs when initlizing.")
    parser.add_argument("--num_warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--random_prob", default=0.1, type=float,
                        help="prob to random replace a masked token")
    parser.add_argument("--keep_prob", default=0.1, type=float,
                        help="prob to keep no change for a masked token")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--pos_shift', action='store_true',
                        help="Using position shift for fine-tuning.")
    args = parser.parse_args()
    return args


def get_model_and_tokenizer(args):
    model_config = UnilmConfig.from_pretrained(
        args.config_name if args.config_name else 'unilm-base-cased',
        cache_dir=args.cache_dir if args.cache_dir else None)
    config = BertForSeq2SeqConfig.from_exist_config(
        config=model_config, label_smoothing=args.label_smoothing,
        max_position_embeddings=args.max_source_seq_length + args.max_target_seq_length)

    logger.info("Model config for seq2seq: %s", str(config))

    tokenizer = UnilmTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else 'unilm-base-cased',
        do_lower_case=args.do_lower_case, cache_dir=args.cache_dir if args.cache_dir else None)

    generator = BertForSequenceToSequence.from_pretrained(
        'unilm-base-cased', config=config, model_type='unilm',
        reuse_position_embedding=True,
        cache_dir=args.cache_dir if args.cache_dir else None)
    generator.to(args.device)

    classifer = Classifier(config.hidden_size, args.num_labels)
    classifer.to(args.device)

    logger.info("Initialize retriever.")
    retriever = Retriever(args, tokenizer)
    return generator, classifer, tokenizer, retriever


def prepare(args):
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'train_opt.json'), 'w'), sort_keys=True, indent=2)

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.batch_size = args.per_gpu_train_batch_size * args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex
            apex.amp.register_half_function(torch, 'einsum')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")


def main():
    args = get_args()
    prepare(args)
    generator, classifier, tokenizer, retriever = get_model_and_tokenizer(args)

    unlabeled_dataset = UnlabledDataset(tokenizer, args.device)
    unlabeled_dataset.load_init(args.unlabeled_data, label_to_id=args.label_to_id)
    valid_dataset = UnlabledDataset(tokenizer, args.device)
    valid_dataset.load_init(args.valid_data, label_to_id=args.label_to_id)
    test_dataset = UnlabledDataset(tokenizer, args.device)
    test_dataset.load_init(args.test_data, label_to_id=args.label_to_id)
    generator_dataset = GeneratorDataset(tokenizer, args)
    generator_dataset.load_init(args.labeled_data, label_to_id=args.label_to_id)
    classifier_dataset = ClassifierDataset(tokenizer, retriever)
    classifier_dataset.load_init(args.labeled_data, label_to_id=args.label_to_id)

    train(generator, classifier, tokenizer, unlabeled_dataset, generator_dataset, classifier_dataset, valid_dataset, test_dataset, retriever, args)


if __name__ == "__main__":
    main()
