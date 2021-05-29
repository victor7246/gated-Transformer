# gated-Transformer
Gated Pretrained Transformer model for robust denoised sequence-to-sequence modeling. It uses a gating unit to detect and correct noises from text data and generate de-noised target from a generative decoder.

![gated-Transformer](https://github.com/LCS2-IIITD/HIT-ACL2021-Codemixed-Representation/blob/main/image/model.png)

In this work we conduct our experiment on three tasks - 

* Denoising corrputed text
* Denoised machine translation 
* Denoise summarization

gated-Transformer is shown effective on - 

* OCR noise
* Random spelling mistakes
* Keyboard based noises
* Insertion, Deletion, Swapping based noises

### Installation for experiments

	$ pip install -r requirements.txt

### Commands to run

#### Training a denoiser

pretrained transformer sequence-to-sequence model (e.g. - BART, T5)

	$ cd ./drive/MyDrive/gated-denoise/ && python train.py --train_file ./data.csv \
                                                      --model_path ./model/ --model_type bart --pretrained_encoder_path "facebook/bart-base" \
                                                      --mask_gate --copy_gate --generate_gate --skip_gate \
                                                      --epochs 15

Transformer encoder-decoder model (e.g. - BERT2BERT)

	$ cd ./drive/MyDrive/gated-denoise/ && python train.py --train_file ./data.csv \
                                                      --model_path ./model/ --model_type seq2seq \
                                                      --pretrained_encoder_path "bert-base-uncased" --pretrained_decoder_path "bert-base-uncased" \
                                                      --mask_gate --copy_gate --generate_gate --skip_gate \
                                                      --epochs 15

#### Inference

	$ $ cd ./drive/MyDrive/gated-denoise/ && python predict.py --data_file ./data.csv \
                                                      --model_path ./model/ --model_type bart --pretrained_encoder_path "facebook/bart-base" \
                                                      --mask_gate --copy_gate --generate_gate --skip_gate 

### Citation
If you find this repo useful, please cite our paper:
```BibTex
@inproceedings{,
  author    = {Ayan Sengupta and
  			   Amit Kumar and
               Sourabh Kumar Bhattacharjee and
               Suman Roy},
  title     = {Gated Transformer for Robust De-noised Sequence-to-Sequence Modelling},
  booktitle = {},
  publisher = {},
  year      = {},
  url       = {},
  doi       = {},
}
