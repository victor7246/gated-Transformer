import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, EncoderDecoderModel

from models import EncoderDecoderModelWithGates, EncoderModelWithGates
from trainer import train_job

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Denoise trainer", conflict_handler='resolve')
    parser.add_argument('--train_file', type=str, help='Path to the data file', required=True)
    parser.add_argument('--model_path', type=str, help='Path to save trained model', required=True)
    parser.add_argument('--min_len_src', type=int, help='Minimum length of source texts', required=False, default=20)
    parser.add_argument('--max_len_src', type=int, help='Maximum length of source texts', required=False, default=300)
    parser.add_argument('--min_len_tgt', type=int, help='Minimum length of target texts', required=False, default=20)
    parser.add_argument('--max_len_tgt', type=int, help='Maximum length of target texts', required=False, default=300)
    parser.add_argument('--batch_size', type=int, help='Batch size for training and validation', required=False, default=8)
    parser.add_argument('--epochs', type=int, help='Number of training epochs', required=False, default=15)
    parser.add_argument('--lr', type=float, help='Learning rate', required=False, default=0.00005)
    parser.add_argument('--early_stopping_rounds', type=int, help='Number of rounds for early stopping of training', required=False, default=5)
    parser.add_argument('--kfold', type=int, help='K-Fold for training and validation', required=False, default=10)

    parser.add_argument('--model_type', type=str, help='Type of models - seq2seq, BART, T5', required=True)
    parser.add_argument('--pretrained_encoder_path', type=str, help='Pretrained encoder model name', required=True)
    parser.add_argument('--pretrained_decoder_path', type=str, help='Pretrained decoder model name', required=False, default=None)

    parser.add_argument('--mask_gate', help='Indicator for masking gate', default=False, action="store_true")
    parser.add_argument('--copy_gate', help='Indicator for copy gate', default=False, action="store_true")
    parser.add_argument('--generate_gate', help='Indicator for generate gate', default=False, action="store_true")
    parser.add_argument('--skip_gate', help='Indicator for skip gate', default=False, action="store_true")

    parser.add_argument('--seed', type=int, help='Random seed', required=False, default=66)

    #args, _ = parser.parse_known_args()
    args = parser.parse_args()

    print (args)

    try:
        assert args.model_type in ['seq2seq','bart','t5']
    except:
        raise ValueError("Model type not in ['seq2seq','bart', 't5']")

    #try:
    #    assert (args.model_type == 'seq2seq' and args.pretrained_encoder_path and args.pretrained_decoder_path) or (args.model_type != 'seq2seq' and args.pretrained_encoder_path)
    #except:
    #    raise ValueError("Check the pretrained paths")


    train = pd.read_csv(args.train_file)
    train = train.dropna().reset_index(drop=True)

    try:
        assert ('source' in train.columns and 'target' in train.columns)
        train = train[['source','target']]
    except:
        raise ValueError("Source and Target columns not found in data")

    encoder_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_encoder_path)

    if args.pretrained_decoder_path:
        decoder_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_decoder_path)
    else:
        decoder_tokenizer = encoder_tokenizer

    if encoder_tokenizer.pad_token_id is None:
        encoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if decoder_tokenizer.pad_token_id is None:
        decoder_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if encoder_tokenizer.mask_token_id is None:
        encoder_tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    if decoder_tokenizer.mask_token_id is None:
        decoder_tokenizer.add_special_tokens({'mask_token': '[MASK]'})

    kf = KFold(n_splits=args.kfold)

    for train_index, val_index in kf.split(train):
        break

    val = train.iloc[val_index]
    train = train.iloc[train_index]

    trainX = torch.Tensor(np.asarray([encoder_tokenizer.encode(i, max_length=args.max_len_src, truncation=True, padding='max_length', add_special_tokens=True) \
                                  for i in tqdm(train.source.values)]))
    trainy = torch.Tensor(np.asarray([decoder_tokenizer.encode(i, max_length=args.max_len_tgt, truncation=True, padding='max_length', add_special_tokens=True) \
                                      for i in tqdm(train.target.values)]))

    valX = torch.Tensor(np.asarray([encoder_tokenizer.encode(i, max_length=args.max_len_src, truncation=True, padding='max_length', add_special_tokens=True) \
                                    for i in tqdm(val.source.values)]))
    valy = torch.Tensor(np.asarray([decoder_tokenizer.encode(i, max_length=args.max_len_tgt, truncation=True, padding='max_length', add_special_tokens=True) \
                                    for i in tqdm(val.target.values)]))

    trainX = torch.tensor(trainX, dtype=torch.long)
    trainy = torch.tensor(trainy, dtype=torch.long)
    valX = torch.tensor(valX, dtype=torch.long)
    valy = torch.tensor(valy, dtype=torch.long)

    gates = []
    if args.mask_gate is True:
        gates.append('mask')
    if args.copy_gate is True:
        gates.append('copy')
    if args.generate_gate is True:
        gates.append('generate')
    if args.skip_gate is True:
        gates.append('skip')

    print ("Running model with {} gates".format(gates))

    if args.model_type == 'seq2seq':
        if args.pretrained_decoder_path:
            model = EncoderDecoderModelWithGates.from_encoder_decoder_pretrained(args.pretrained_encoder_path, args.pretrained_decoder_path, gates=gates)
        else:
            model = EncoderDecoderModel.from_pretrained(args.pretrained_encoder_path)
            model = EncoderDecoderModelWithGates(config=model.config,encoder=model.encoder, decoder=model.decoder, gates=gates)

        model.config.encoder.max_length = args.max_len_src
        model.config.decoder.max_length = args.max_len_tgt

        model.config.encoder.min_length = args.min_len_src
        model.config.decoder.min_length = args.min_len_tgt

        model.encoder_tokenizer = encoder_tokenizer
        model.decoder_tokenizer = decoder_tokenizer

    else:
        model = EncoderModelWithGates(args.model_type, args.pretrained_encoder_path, gates=gates)
        model.encoder.config.max_length = args.max_len_src
        model.decoder.config.max_length = args.max_len_tgt

        model.encoder.config.min_length = args.min_len_src
        model.decoder.config.min_length = args.min_len_tgt

        model.encoder_tokenizer = encoder_tokenizer
        model.decoder_tokenizer = decoder_tokenizer

    encoder_mask_id = encoder_tokenizer.mask_token_id
    decoder_mask_id = decoder_tokenizer.mask_token_id

    print ("Total number of parameters {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad == True)))

    train_data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(trainX, trainy), batch_size=args.batch_size)

    val_data_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(valX, valy), batch_size=args.batch_size)

    print ("Train and val loader length {} and {}".format(len(train_data_loader), len(val_data_loader)))

    try:
        os.makedirs(args.model_path)
    except:
        pass

    train_job(model, encoder_mask_id, decoder_mask_id, train_data_loader, val_data_loader, args.lr, args.epochs, args.early_stopping_rounds, args.model_path, seed_val=args.seed)
