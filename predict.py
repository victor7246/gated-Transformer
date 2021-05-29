import os
import random
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, EncoderDecoderModel
from rouge_score import rouge_scorer

from models import EncoderDecoderModelWithGates, EncoderModelWithGates
from scorers import WRR, bleu_score

pd.options.display.max_columns = 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Denoise trainer", conflict_handler='resolve')
    parser.add_argument('--data_file', type=str, help='Path to the data file', required=True)
    parser.add_argument('--model_path', type=str, help='Path to save trained model', required=True)
    parser.add_argument('--min_len_src', type=int, help='Minimum length of source texts', required=False, default=20)
    parser.add_argument('--max_len_src', type=int, help='Maximum length of source texts', required=False, default=300)
    parser.add_argument('--min_len_tgt', type=int, help='Minimum length of target texts', required=False, default=20)
    parser.add_argument('--max_len_tgt', type=int, help='Maximum length of target texts', required=False, default=300)

    parser.add_argument('--model_type', type=str, help='Type of models - seq2seq, BART, T5', required=True)
    parser.add_argument('--pretrained_encoder_path', type=str, help='Pretrained encoder model name', required=True)
    parser.add_argument('--pretrained_decoder_path', type=str, help='Pretrained decoder model name', required=False, default=None)

    parser.add_argument('--mask_gate', help='Indicator for masking gate', default=False, action="store_true")
    parser.add_argument('--copy_gate', help='Indicator for copy gate', default=False, action="store_true")
    parser.add_argument('--generate_gate', help='Indicator for generate gate', default=False, action="store_true")
    parser.add_argument('--skip_gate', help='Indicator for skip gate', default=False, action="store_true")

    parser.add_argument('--seed', type=int, help='Random seed', required=False, default=66)

    parser.add_argument('--teacher_forcing', type=int, help='Teacher Forcing', required=False, default=1)

    args, _ = parser.parse_known_args()

    try:
        assert args.model_type in ['seq2seq','bart','t5']
    except:
        raise ValueError("Model type not in ['seq2seq','bart', 't5']")

    #try:
    #    assert (args.model_type == 'seq2seq' and args.pretrained_encoder_path and args.pretrained_decoder_path) or (args.model_type != 'seq2seq' and args.pretrained_encoder_path)
    #except:
    #    raise ValueError("Check the pretrained paths")


    val = pd.read_csv(args.data_file)
    val = val.dropna().reset_index(drop=True)

    try:
        assert ('source' in val.columns)
    except:
        raise ValueError("Source column not found in data")

    encoder_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_encoder_path)

    if args.pretrained_decoder_path:
        decoder_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_decoder_path)
    else:
        decoder_tokenizer = encoder_tokenizer

    valX = torch.Tensor(np.asarray([encoder_tokenizer.encode(i, max_length=args.max_len_src, truncation=True, padding='max_length', add_special_tokens=True) \
                                    for i in tqdm(val.source.values)]))
    if 'target' in val.columns:
        valy = torch.Tensor(np.asarray([decoder_tokenizer.encode(i, max_length=args.max_len_tgt, truncation=True, padding='max_length', add_special_tokens=True) \
                                    for i in tqdm(val.target.values)]))

    valX = torch.tensor(valX, dtype=torch.long)
    if 'target' in val.columns:
        valy = torch.tensor(valy, dtype=torch.long)

    gates = []
    if args.mask_gate == True:
        gates.append('mask')
    if args.copy_gate == True:
        gates.append('copy')
    if args.generate_gate == True:
        gates.append('generate')
    if args.skip_gate == True:
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

    if args.model_type == 't5':
        encoder_mask_id = encoder_tokenizer.additional_special_tokens_ids[0]
        decoder_mask_id = decoder_tokenizer.additional_special_tokens_ids[0]
    else:
        encoder_mask_id = encoder_tokenizer.mask_token_id
        decoder_mask_id = decoder_tokenizer.mask_token_id

    #print ("Total number of parameters {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad == True)))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.empty_cache()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    if 'target' in val.columns:
        val_data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(valX,valy), batch_size=4)
    else:
        val_data_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(valX), batch_size=4)

    model.load_state_dict(torch.load(os.path.join(args.model_path,'model.pth')))

    model.eval()
    all_val_logits = []
    all_generate_probs = []
    all_copy_probs = []
    all_masking_probs = []
    all_skip_probs = []

    # Evaluate data for one epoch
    for batch in tqdm(val_data_loader):
        input_ids = batch[0].to(device)
        if 'target' in val.columns:
            output_ids = batch[1].to(device)
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            if 'target' in val.columns and args.teacher_forcing == 1:
                outputs, generate_prob, copy_prob,masking_prob, skip_prob = model(input_ids=input_ids, encoder_mask_token_id = torch.tensor([[encoder_mask_id]]).to(device),\
                                     decoder_mask_token_id = decoder_mask_id, labels=output_ids, return_dict=True)
            else:
                outputs, generate_prob, copy_prob,masking_prob, skip_prob = model(input_ids=input_ids, encoder_mask_token_id = torch.tensor([[encoder_mask_id]]).to(device),\
                                     decoder_mask_token_id = decoder_mask_id, return_dict=True)
            logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        
        all_val_logits.extend(logits.argmax(-1))
        all_generate_probs.extend(generate_prob.detach().cpu().numpy())
        all_copy_probs.extend(copy_prob.detach().cpu().numpy())
        all_masking_probs.extend(masking_prob.detach().cpu().numpy())
        all_skip_probs.extend(skip_prob.detach().cpu().numpy())

    predicted_texts = []

    if len(all_val_logits) != val.shape[0]:
        all_val_logits = np.concatenate(all_val_logits, axis=0)
    #all_generate_probs = np.concatenate(all_generate_probs, axis=0)
    #all_copy_probs = np.concatenate(all_copy_probs, axis=0)
    #all_masking_probs = np.concatenate(all_masking_probs, axis=0)
    #all_skip_probs = np.concatenate(all_skip_probs, axis=0)

    #print (all_generate_probs.shape, all_copy_probs.shape, np.asarray(all_masking_probs).shape, all_skip_probs.shape)

    #print (all_val_logits.shape)
    for i in all_val_logits:
        text = decoder_tokenizer.decode(i)
        text = text.replace('<s>','')
        text = text.replace('</s>','')
        text = text.replace('<pad>','')
        #text = [k for k in text if k not in ['<s>','</s>','<pad>']]
        predicted_texts.append(text.strip())
        #predicted_texts.append(" ".join(text).strip())

    val['predicted_target'] = predicted_texts

    val['text_len'] = val.source.apply(lambda x: len(encoder_tokenizer.encode(x, max_length=512, add_special_tokens=True)))
    val = val[val.text_len < args.max_len_src].reset_index(drop=True)

    scorer = rouge_scorer.RougeScorer(['rouge1','rougeL'], use_stemmer=True)

    if 'target' in val.columns:
        val['WRR'] = val.apply(lambda x: WRR(x.target, x.predicted_target), axis=1)
        val['BLEU'] = val.apply(lambda x: bleu_score(x.target, x.predicted_target), axis=1)
        val['Rogue'] = val.apply(lambda x: scorer.score(x.target.lower(),x.predicted_target.lower())['rougeL'].fmeasure,axis=1)

        print (val[['WRR','BLEU','Rogue']].describe())

    val.to_csv(os.path.join(args.model_path,'validation_output.csv'),index=False)

    try:
        np.save(os.path.join(args.model_path,'generate_probs.npy'), np.asarray(all_generate_probs)[:,:,0])
        np.save(os.path.join(args.model_path,'copy_probs.npy'), np.asarray(all_copy_probs)[:,:,0])
        np.save(os.path.join(args.model_path,'mask_probs.npy'), np.asarray(all_masking_probs)[:,:,0])
        np.save(os.path.join(args.model_path,'skip_probs.npy'), np.asarray(all_skip_probs)[:,:,0])
    except:
        pass
