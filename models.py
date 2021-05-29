import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, \
                         AutoConfig, EncoderDecoderModel, BertForMaskedLM, BartForConditionalGeneration, T5ForConditionalGeneration

from transformers.configuration_utils import PretrainedConfig
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.encoder_decoder.configuration_encoder_decoder import EncoderDecoderConfig
from typing import Optional

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "EncoderDecoderConfig"

class EncoderDecoderModelWithGates(PreTrainedModel):
    r"""
    :class:`~transformers.EncoderDecoder` is a generic model class that will be instantiated as a transformer
    architecture with one of the base model classes of the library as encoder and another one as decoder when created
    with the :meth`~transformers.AutoModel.from_pretrained` class method for the encoder and
    :meth`~transformers.AutoModelForCausalLM.from_pretrained` class method for the decoder.
    """
    config_class = EncoderDecoderConfig
    base_model_prefix = "encoder_decoder"

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        gates: list = ['mask','copy','generate','skip'],
        encoder_tokenizer: Optional = None,
        decoder_tokenizer: Optional = None
    ):
        
        assert config is not None or (
            encoder is not None and decoder is not None
        ), "Either a configuration or an Encoder and a decoder has to be provided"
        if config is None:
            config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            assert isinstance(config, self.config_class), "config: {} has to be of type {}".format(
                config, self.config_class
            )
        # initialize with config
        super().__init__(config)

        if encoder is None:
            
            encoder = AutoModel.from_config(config.encoder)

        if decoder is None:
            
            decoder = AutoModelForCausalLM.from_config(config.decoder)

        #tokenizer = AutoTokenizer.from_pretrained(config.encoder._name_or_path)
            
        self.encoder = encoder
        self.decoder = decoder
        self.gates = gates
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

        assert (
            self.encoder.get_output_embeddings() is None
        ), "The encoder {} should not have a LM Head. Please use a model without LM Head"

        # tie encoder, decoder weights if config set accordingly
        #self.tie_weights()
        
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.embedding = self.encoder.embeddings
        
        if 'mask' in self.gates:
            self.mask_gate = nn.Linear(config.encoder.hidden_size, 1, bias=True)
        
        if 'copy' in self.gates:
            self.copy_gate = nn.Linear(config.decoder.hidden_size, 1, bias=True)
        
        if 'generate' in self.gates:
            self.generate_gate = nn.Linear(config.decoder.hidden_size, 1, bias=True)
        
        if 'skip' in self.gates:
            self.skip_gate = nn.Linear(config.decoder.hidden_size, 1, bias=True)
        
        
        #for param in self.mask_gate.parameters():
        #    param.requires_grad = 'mask' in self.gates
            
        #for param in self.copy_gate.parameters():
        #    param.requires_grad = 'copy' in self.gates
            
        #for param in self.generate_gate.parameters():
        #    param.requires_grad = 'generate' in self.gates
            
        #for param in self.skip_gate.parameters():
        #    param.requires_grad = 'skip' in self.gates
    
    def tie_weights(self):
        # tie encoder & decoder if needed
        if self.config.tie_encoder_decoder:
            # tie encoder and decoder base model
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(
                self.encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix
            )

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_pretrained_model_name_or_path: str = None,
        decoder_pretrained_model_name_or_path: str = None,
        gates: list = ['mask','copy','generate','skip'],
        *model_args,
        **kwargs
    ) -> PreTrainedModel:

        kwargs_encoder = {
            argument[len("encoder_") :]: value for argument, value in kwargs.items() if argument.startswith("encoder_")
        }

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        # remove encoder, decoder kwargs from kwargs
        for key in kwargs_encoder.keys():
            del kwargs["encoder_" + key]
        for key in kwargs_decoder.keys():
            del kwargs["decoder_" + key]

        # Load and initialize the encoder and decoder
        # The distinction between encoder and decoder at the model level is made
        # by the value of the flag `is_decoder` that we need to set correctly.
        encoder = kwargs_encoder.pop("model", None)
        if encoder is None:
            assert (
                encoder_pretrained_model_name_or_path is not None
            ), "If `model` is not defined as an argument, a `encoder_pretrained_model_name_or_path` has to be defined"

            if "config" not in kwargs_encoder:

                encoder_config = AutoConfig.from_pretrained(encoder_pretrained_model_name_or_path)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:

                    logger.info(
                        f"Initializing {encoder_pretrained_model_name_or_path} as a encoder model from a decoder model. Cross-attention and casual mask are disabled."
                    )
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False

                kwargs_encoder["config"] = encoder_config

            encoder = AutoModel.from_pretrained(encoder_pretrained_model_name_or_path, *model_args, **kwargs_encoder)

        decoder = kwargs_decoder.pop("model", None)
        if decoder is None:
            assert (
                decoder_pretrained_model_name_or_path is not None
            ), "If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined"

            if "config" not in kwargs_decoder:

                decoder_config = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path)
                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(
                        f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers."
                    )
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True

                kwargs_decoder["config"] = decoder_config

            if kwargs_decoder["config"].is_decoder is False or kwargs_decoder["config"].add_cross_attention is False:
                logger.warning(
                    f"Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` passed to `.from_encoder_decoder_pretrained(...)` are set to `True` or do not pass a `decoder_config` to `.from_encoder_decoder_pretrained(...)`"
                )

            decoder = AutoModelForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)

        # instantiate config with corresponding kwargs
        config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config, **kwargs)
        return cls(encoder=encoder, decoder=decoder, config=config, gates=gates)

    def forward(
        self,
        encoder_mask_token_id,
        decoder_mask_token_id,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,  # TODO: (PVP) implement :obj:`use_cache`
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,  # TODO: (PVP) implement :obj:`use_cache`
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
        **kwargs,
    ):
        r"""
        Returns:
        Examples::
            >>> from transformers import EncoderDecoderModel, BertTokenizer
            >>> import torch
            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased') # initialize Bert2Bert from pre-trained checkpoints
            >>> # forward
            >>> input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)
            >>> # training
            >>> outputs = model(input_ids=input_ids, decoder_input_ids=input_ids, labels=input_ids)
            >>> loss, logits = outputs.loss, outputs.logits
            >>> # save and load from pretrained
            >>> model.save_pretrained("bert2bert")
            >>> model = EncoderDecoderModel.from_pretrained("bert2bert")
            >>> # generation
            >>> generated = model.generate(input_ids, decoder_start_token_id=model.config.decoder.pad_token_id)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if decoder_input_ids is None:
            if labels is not None:
                decoder_input_ids = self._shift_right(labels)
            else:
                decoder_input_ids = input_ids

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )

        mask_e = self.embedding(encoder_mask_token_id)
        
        encoder_hidden_states = encoder_outputs[0]
        
        if 'mask' in self.gates:
            masking_prob = torch.sigmoid(self.mask_gate(encoder_hidden_states))
            encoder_hidden_states = masking_prob * mask_e + (1-masking_prob)*encoder_hidden_states
        else:
            masking_prob = torch.zeros_like(input_ids)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs_decoder,
        )
        
        #print (decoder_outputs.logits.shape)
        #decoder_hidden_states = decoder_outputs[0] #decoder_outputs.hidden_states[-1]
        decoder_hidden_states = decoder_outputs.hidden_states[-1]

        all_probs = []
        if 'copy' in self.gates:
            copy_prob = torch.sigmoid(self.copy_gate(decoder_hidden_states))
            all_probs.append(copy_prob.unsqueeze(0))
        else:
            all_probs.append(torch.zeros_like(decoder_hidden_states[:,:,:1]).unsqueeze(0))

        if 'generate' in self.gates:
            generate_prob = torch.sigmoid(self.generate_gate(decoder_hidden_states))
            all_probs.append(generate_prob.unsqueeze(0))
        else:
            all_probs.append(torch.zeros_like(decoder_hidden_states[:,:,:1]).unsqueeze(0))

        if 'skip' in self.gates:
            skip_prob = torch.sigmoid(self.skip_gate(decoder_hidden_states))
            all_probs.append(skip_prob.unsqueeze(0))
        else:
            all_probs.append(torch.zeros_like(decoder_hidden_states[:,:,:1]).unsqueeze(0))

        decoder_input_one_hot = torch.nn.functional.one_hot(decoder_input_ids, num_classes=self.config.decoder.vocab_size) #self.decoder_tokenizer.vocab_size
        
        stacks = nn.Softmax(dim=0)(torch.stack(all_probs)) #[copy_prob.unsqueeze(0),generate_prob.unsqueeze(0),skip_prob.unsqueeze(0)]
        copy_prob = stacks[0]
        generate_prob = stacks[1]
        skip_prob = stacks[2]
        
        skip_logits = torch.zeros_like(decoder_outputs.logits)
        skip_logits[:,:,decoder_mask_token_id] = 1
        
        logits = decoder_outputs.logits

        if 'generate' in self.gates:
            logits = generate_prob * nn.LogSoftmax(dim=-1)(logits)

        if 'copy' in self.gates:
            logits += copy_prob*decoder_input_one_hot

        if 'skip' in self.gates:
            logits += skip_prob*skip_logits

        #logits = generate_prob*nn.LogSoftmax(dim=-1)() + copy_prob*decoder_input_one_hot + skip_prob*skip_logits
        #logits = decoder_outputs.logits
        #logits = nn.LogSoftmax(dim=-1)(logits)
        
        if labels is not None:
            #shift_logits = logits[..., :-1, :].contiguous()
            #shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            #loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            #loss_fn = nn.NLLLoss()
            #loss = loss = loss_fn(logits.transpose(1, 2), labels)
        else:
            loss = None
            
        # TODO(PVP): currently it is not possible to use `past`
        if not return_dict:
            return decoder_outputs + encoder_outputs
            #print ("inside loop")
            #return (1-copy_prob)*decoder_outputs + copy_prob*encoder_outputs
    
        #print (decoder_outputs.hidden_states.shape)
        #print (encoder_outputs.last_hidden_state.shape)
        #print (encoder_outputs.hidden_states.shape)
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=None,  # TODO(PVP) - need to implement cache for BERT, etc... before this works
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        ), generate_prob, copy_prob,masking_prob, skip_prob

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, encoder_outputs=None, **kwargs):
        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids)
        decoder_attention_mask = decoder_inputs["attention_mask"] if "attention_mask" in decoder_inputs else None
        input_dict = {
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_input_ids": decoder_inputs["input_ids"],
            "encoder_outputs": encoder_outputs,
        }

        # Ideally all models should have a :obj:`use_cache`
        # leave following to ifs until all have it implemented
        if "use_cache" in decoder_inputs:
            input_dict["decoder_use_cache"] = decoder_inputs["use_cache"]

        if "past_key_values" in decoder_inputs:
            input_dict["past_key_values"] = decoder_inputs["past_key_values"]

        return input_dict

    def _reorder_cache(self, past, beam_idx):
        # apply decoder cache reordering here
        return self.decoder._reorder_cache(past, beam_idx)

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.decoder.config.bos_token_id or self.config.decoder_start_token_id or 101
        pad_token_id = self.decoder.config.pad_token_id or -100

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined."

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids

class EncoderModelWithGates(nn.Module):
    def __init__(self, model_type: str, pretrained_path: str, \
                gates: list = ['mask','copy','generate','skip'], \
                encoder_tokenizer: Optional = None, \
                decoder_tokenizer: Optional = None):
        super().__init__()
        
        self.model_type = model_type
        self.gates = gates
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer

        if model_type == 't5':
            model = T5ForConditionalGeneration.from_pretrained(pretrained_path)
            self.encoder = model.encoder
            self.decoder = model.decoder
            self.lm_head = model.lm_head
            self.config = model.config
            self.model_dim = model.model_dim

        elif model_type == 'bart':
            model = BartForConditionalGeneration.from_pretrained(pretrained_path)
            self.encoder = model.model.encoder
            self.decoder= model.model.decoder
            self.lm_head = model.lm_head
            self.config = model.config
            self.model_dim = model.config.hidden_size

        else:
            raise ValueError("Models can be either T5 or BART")
        
        self.register_buffer("final_logits_bias", torch.zeros((1, self.decoder.config.vocab_size)))

        self.embedding = self.encoder.embed_tokens
        
        #self.mask_gate = nn.Linear(self.encoder.config.hidden_size, 1, bias=True)
        #self.copy_gate = nn.Linear(self.decoder.config.hidden_size, 1, bias=True)
        #self.generate_gate = nn.Linear(self.decoder.config.hidden_size, 1, bias=True)
        #self.skip_gate = nn.Linear(self.decoder.config.hidden_size, 1, bias=True)
        
        
        #for param in self.mask_gate.parameters():
        #    param.requires_grad = 'mask' in self.gates
            
        #for param in self.copy_gate.parameters():
        #    param.requires_grad = 'copy' in self.gates
            
        #for param in self.generate_gate.parameters():
        #    param.requires_grad = 'generate' in self.gates
            
        #for param in self.skip_gate.parameters():
        #    param.requires_grad = 'skip' in self.gates
        
        if 'mask' in self.gates:
            self.mask_gate = nn.Linear(self.encoder.config.hidden_size, 1, bias=True)
        
        if 'copy' in self.gates:
            self.copy_gate = nn.Linear(self.decoder.config.hidden_size, 1, bias=True)
        
        if 'generate' in self.gates:
            self.generate_gate = nn.Linear(self.decoder.config.hidden_size, 1, bias=True)
        
        if 'skip' in self.gates:
            self.skip_gate = nn.Linear(self.decoder.config.hidden_size, 1, bias=True)

    def forward(self, \
               encoder_mask_token_id, \
               decoder_mask_token_id, \
               input_ids, \
               decoder_input_ids = None, \
               labels=None,
               return_dict=True):
        
        if decoder_input_ids is None:
            if labels is not None:
                decoder_input_ids = self._shift_right(labels)
            else:
                decoder_input_ids = input_ids

        encoder_outputs = self.encoder(input_ids)
        encoder_hidden_states = encoder_outputs[0] #encoder_outputs.last_hidden_state

        mask_e = self.embedding(encoder_mask_token_id)

        if 'mask' in self.gates:
            masking_prob = torch.sigmoid(self.mask_gate(encoder_hidden_states))
            encoder_hidden_states = masking_prob * mask_e + (1-masking_prob)*encoder_hidden_states
        else:
            masking_prob = torch.zeros_like(input_ids)
        
        #print (decoder_outputs.logits.shape)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids,encoder_hidden_states=encoder_hidden_states, return_dict=True)
        decoder_hidden_states = decoder_outputs[0] #decoder_outputs.last_hidden_state
        
        all_probs = []
        if 'copy' in self.gates:
            copy_prob = torch.sigmoid(self.copy_gate(decoder_hidden_states))
            all_probs.append(copy_prob.unsqueeze(0))
        else:
            all_probs.append(torch.zeros_like(decoder_hidden_states[:,:,:1]).unsqueeze(0))

        if 'generate' in self.gates:
            generate_prob = torch.sigmoid(self.generate_gate(decoder_hidden_states))
            all_probs.append(generate_prob.unsqueeze(0))
        else:
            all_probs.append(torch.zeros_like(decoder_hidden_states[:,:,:1]).unsqueeze(0))

        if 'skip' in self.gates:
            skip_prob = torch.sigmoid(self.skip_gate(decoder_hidden_states))
            all_probs.append(skip_prob.unsqueeze(0))
        else:
            all_probs.append(torch.zeros_like(decoder_hidden_states[:,:,:1]).unsqueeze(0))
        
        if self.model_type == 'bart':
            logits = self.lm_head(decoder_hidden_states) + self.final_logits_bias
        elif self.model_type == 't5':
            logits = self.lm_head(decoder_hidden_states * (self.model_dim ** -0.5))

        decoder_input_one_hot = torch.nn.functional.one_hot(decoder_input_ids, num_classes=logits.shape[-1]) #self.decoder.config.vocab_size
        
        stacks = nn.Softmax(dim=0)(torch.stack(all_probs)) #[copy_prob.unsqueeze(0),generate_prob.unsqueeze(0),skip_prob.unsqueeze(0)]
        copy_prob = stacks[0]
        generate_prob = stacks[1]
        skip_prob = stacks[2]

        skip_logits = torch.zeros_like(logits)
        skip_logits[:,:,decoder_mask_token_id] = 1

        if 'generate' in self.gates:
            logits = generate_prob * nn.LogSoftmax(dim=-1)(logits)

        if 'copy' in self.gates:
            logits += copy_prob*decoder_input_one_hot

        if 'skip' in self.gates:
            logits += skip_prob*skip_logits

        #logits = generate_prob*nn.LogSoftmax(dim=-1)(decoder_outputs.logits) + copy_prob*decoder_input_one_hot + skip_prob*skip_logits
        #logits = decoder_outputs.logits
        #logits = nn.LogSoftmax(dim=-1)(logits)
        
        if labels is not None:
            #shift_logits = logits[..., :-1, :].contiguous()
            #shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            #loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            #loss_fn = nn.NLLLoss()
            #loss = loss = loss_fn(logits.transpose(1, 2), labels)
        else:
            loss = None
            
        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=None,  # TODO(PVP) - need to implement cache for BERT, etc... before this works
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        ), generate_prob, copy_prob,masking_prob, skip_prob

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids