from typing import List, Tuple

import clip
import torch
from torch import nn
from torch import tensor

from .utils import CLIP
from detectron2.modeling.backbone import Backbone   #changed
from collections import OrderedDict    #changed


class PromptExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._buffer_init = False
        self.with_trainable_params = False

    def init_buffer(self, clip_model):
        self._buffer_init = True

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        raise NotImplementedError()


class PredefinedPromptExtractor(PromptExtractor):
    def __init__(self, templates: List[str]):
        super().__init__()
        self.templates = templates

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        text_features_bucket = []
        for template in self.templates:
            noun_tokens = [clip.tokenize(template.format(noun)) for noun in noun_list]
            text_inputs = torch.cat(noun_tokens).to(
                clip_model.text_projection.data.device
            )
            text_features = clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_bucket.append(text_features)
        del text_inputs
        # ensemble by averaging
        text_features = torch.stack(text_features_bucket).mean(dim=0)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features


class ImageNetPromptExtractor(PredefinedPromptExtractor):
    def __init__(self):
        super().__init__(CLIP.IMAGENET_PROMPT)


class VILDPromptExtractor(PredefinedPromptExtractor):
    def __init__(self):
        super().__init__(CLIP.VILD_PROMPT)


class LearnablePromptExtractor(PromptExtractor):
    def __init__(self, prompt_dim: int, prompt_shape: Tuple[int, int]):
        super().__init__()
        assert len(prompt_shape) == 2, "prompt_shape must be a tuple of length 2"
        self.prompt_dim = prompt_dim
        self.prompt_shape = prompt_shape
        self.prefix_prompt = self._init_prompt(self.n_prefix)
        self.suffix_prompt = self._init_prompt(self.n_suffix)
        self._buffer_init = False
        self.with_trainable_params = True

    def _init_prompt(self, length):
        if length == 0:
            return None
        prompt_tensor = torch.empty(length, self.prompt_dim)
        nn.init.normal_(prompt_tensor, std=0.02)
        return nn.Parameter(prompt_tensor)

    def init_buffer(self, clip_model):
        sentence = "X."
        prompt = clip.tokenize(sentence)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(
                clip_model.dtype
            )  # 2,77,512
        self.register_buffer("start_signal", embedding[0, :1, :])  # 1,512
        self.register_buffer("dot_signal", embedding[0, 2:3, :])  # 1,512
        self.register_buffer("end_signal", embedding[0, 3:4, :])  # 1,512
        self.register_buffer("pad_signal", embedding[0, 4:5, :])  # 1,512
        self.noun_bucket = {}
        self._buffer_init = True

    def forward(self, noun_list: List[str], clip_model: nn.Module):
        if not self._buffer_init:
            raise RuntimeError(
                f"Buffer of {self.__class__.__name__} is not initialized"
            )
        self._update_noun_features(noun_list, clip_model)

        prefix = [self.start_signal]
        if self.prefix_prompt is not None:
            prefix.append(self.prefix_prompt)
        prefix = torch.cat(prefix)
        suffix = [self.dot_signal, self.end_signal]
        if self.suffix_prompt is not None:
            suffix.insert(0, self.suffix_prompt)
        suffix = torch.cat(suffix)
        # only process those which are not in bucket
        lengths = [
            len(prefix) + len(suffix) + len(self.noun_bucket[noun])
            for noun in noun_list
        ]
        embeddings = torch.stack(
            [
                torch.cat(
                    [prefix, self.noun_bucket[noun], suffix]
                    + [self.pad_signal.expand(77 - length, -1)]
                )
                for noun, length in zip(noun_list, lengths)
            ]
        )  # cls,77,512
        indices = torch.Tensor(lengths).long().to(embeddings.device) - 1
        text_features = self.get_text_feature(embeddings, indices, clip_model)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def _update_noun_features(self, noun_list, clip_model):
        left_class_names = [noun for noun in noun_list if noun not in self.noun_bucket]
        if len(left_class_names) > 0:
            with torch.no_grad():
                tokens, name_lengths = clip.tokenize(
                    left_class_names, return_length=True
                )
                name_lengths = [
                    n - 2 for n in name_lengths
                ]  # remove start end end prompt
                text_embeddings = clip_model.token_embedding(
                    tokens.to(self.device)
                ).type(clip_model.dtype)
                text_embeddings = [
                    embedding[1 : 1 + length]
                    for embedding, length in zip(text_embeddings, name_lengths)
                ]
            self.noun_bucket.update(
                {
                    name: embedding
                    for name, embedding in zip(left_class_names, text_embeddings)
                }
            )

    @staticmethod
    def get_text_feature(x, indices, clip_model):
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), indices] @ clip_model.text_projection
        return x

    @property
    def n_prefix(self):
        return self.prompt_shape[0]

    @property
    def n_suffix(self):
        return self.prompt_shape[1]

    @property
    def device(self):
        return self.start_signal.device

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """

        repr = f"prefix_prompt:{self.n_prefix},suffix_prompt:{self.n_suffix},dimension:{self.prompt_dim}\n"
        repr = repr + "[Normal_Init(mu=0,std=0.02)]"
        return repr
    
    
class ConditionalLearnablePromptExtractor(PromptExtractor):
    def __init__(self, prompt_dim: int, prompt_shape: Tuple[int, int]):          #clip_model as arguement
        super().__init__()
        assert len(prompt_shape) == 2, "prompt_shape must be a tuple of length 2"
        self.prompt_dim = prompt_dim
        self.prompt_shape = prompt_shape
        self.prefix_prompt = self._init_prompt(self.n_prefix)
        self.suffix_prompt = self._init_prompt(self.n_suffix)
        self._buffer_init = False
        self.with_trainable_params = False #changed
        self.with_conditional_trainable_params = True
        
        
        
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(1024, prompt_dim // 16)),   #channelsize=1024
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(prompt_dim // 16, prompt_dim))
        ]))
        
        #if cfg.MODEL.CLIP_ADAPTER.PREC == "fp16":    # change the cgf file location  # CLIP's default precision is fp16
        #    self.meta_net.half()

    def _init_prompt(self, length):
        if length == 0:
            return None
        prompt_tensor = torch.empty(length, self.prompt_dim)
        nn.init.normal_(prompt_tensor, std=0.02)
        return nn.Parameter(prompt_tensor)

    def init_buffer(self, clip_model):
        sentence = "X."
        prompt = clip.tokenize(sentence)
        #print(prompt.shape,"prompt")
        #exit()
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt).type(
                clip_model.dtype
            )  # 2,77,512
        self.register_buffer("start_signal", embedding[0, :1, :])  # 1,512
        self.register_buffer("dot_signal", embedding[0, 2:3, :])  # 1,512
        self.register_buffer("end_signal", embedding[0, 3:4, :])  # 1,512
        self.register_buffer("pad_signal", embedding[0, 4:5, :])  # 1,512
        self.noun_bucket = {}
        self._buffer_init = True

    def forward(self, noun_list: List[str], clip_model: nn.Module, features):
        if not self._buffer_init:
            raise RuntimeError(
                f"Buffer of {self.__class__.__name__} is not initialized"
            )
        self._update_noun_features(noun_list, clip_model,features)
        
        
 
        batch_size=features.shape[0]
        # batch_size = 1
        m = torch.nn.AvgPool2d((features.shape[2], features.shape[3]))
        features = m(features)
        features = torch.squeeze(features,3)
        features = torch.squeeze(features,2)
    
        
        prefix = [self.start_signal]
        if self.prefix_prompt is not None:
            prefix.append(self.prefix_prompt)
        prefix = torch.cat(prefix)                   #type=tensor, [33,512]
        suffix = [self.dot_signal, self.end_signal]
        if self.suffix_prompt is not None:
            suffix.insert(0, self.suffix_prompt)
        suffix = torch.cat(suffix)                   #type=tensor, [2,512]
        # only process those which are not in bucket
        
       
        
        lengths = [
            len(prefix) + len(suffix) + self.noun_bucket[noun].shape[1]
            for noun in noun_list
        ]   # 512
         
       
        bias = self.meta_net(features)      #it should be (batch, ctx_dim) that is (24,512)
        # print(bias.shape)
        bias = bias.unsqueeze(1)
        # bias = torch.cuda.FloatTensor(batch_size,1,512).fill_(0)
        prefix = prefix.unsqueeze(0) 
        suffix = suffix.unsqueeze(0)
        prefix = prefix + bias # like a shifted ctx
        suffix = suffix + bias
        # print(suffix.shape, prefix.shape)
         
        if len(self.pad_signal.shape) == 2:
            self.pad_signal = self.pad_signal.unsqueeze(0)
            self.pad_signal = self.pad_signal.repeat(batch_size,1,1)       #batch size
        elif len(self.pad_signal.shape) == 3:    
            if self.pad_signal.shape[0] != batch_size:
                self.pad_signal = self.pad_signal[0]
                self.pad_signal = self.pad_signal.unsqueeze(0)
                self.pad_signal = self.pad_signal.repeat(batch_size,1,1)
                   
            
        embeddings = torch.stack(
            [
                torch.cat(
                    [prefix, self.noun_bucket[noun], suffix]
                    + [self.pad_signal.expand(-1, 77 - length, -1)], dim=1,
                )
                for noun, length in zip(noun_list, lengths)
            ]
        )  # cls,77,512   # lengths,bs,77,512
        
        print(embeddings.shape,"embeddings")
        #exit()
        cls = len(lengths)
        indices = torch.Tensor(lengths).long().to(embeddings.device) - 1     #pointtolastwordinsent   #indices.repeat(batch_size)
        #print(indices,"INDICES")
        #print(indices.shape,"shape of indices")
        #exit()
        embeddings = embeddings.permute(1,0,2,3)
        # embeddings = embeddings.reshape(cls*batch_size, 77, 512)         #batch_size
        text_features = []
        for emb in embeddings:
            text_feature = self.get_text_feature(emb, indices, clip_model,features)
            text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            text_features.append(text_feature)
        text_features = torch.stack(text_features)
        # text_features = text_features.reshape(batch_size,(cls*batch_size)//batch_size,512)
        

        return text_features

    def _update_noun_features(self, noun_list, clip_model,features):
        batch_size=features.shape[0]
        # batch_size = 1
        left_class_names = [noun for noun in noun_list if noun not in self.noun_bucket]
        if len(left_class_names) > 0:
            
            with torch.no_grad():
                tokens, name_lengths = clip.tokenize(
                    left_class_names, return_length=True
                )
                name_lengths = [
                    n - 2 for n in name_lengths
                ]  # remove start end end prompt
                text_embeddings = clip_model.token_embedding(
                    tokens.to(self.device)
                ).type(clip_model.dtype)
                text_embeddings = [
                    embedding[1 : 1 + length]
                    for embedding, length in zip(text_embeddings, name_lengths)
                ]
            self.noun_bucket.update(
                {
                    name: embedding
                    for name, embedding in zip(left_class_names, text_embeddings)
                }
             
            )
            
            
        for noun in noun_list:
        
            if len(self.noun_bucket[noun].shape) == 2: 
                self.noun_bucket[noun] = self.noun_bucket[noun].unsqueeze(0)
                self.noun_bucket[noun] = self.noun_bucket[noun].repeat(batch_size,1,1) 
            elif len(self.noun_bucket[noun].shape) == 3:    
                
                if self.noun_bucket[noun].shape[0] != batch_size:
                    self.noun_bucket[noun] = self.noun_bucket[noun][0]
                    self.noun_bucket[noun] = self.noun_bucket[noun].unsqueeze(0)
                    self.noun_bucket[noun] = self.noun_bucket[noun].repeat(batch_size,1,1)
                    
                    
                    
              
    
    
    
    
    #batch size, 
    #emb.reshape(b*n,l,d)
    @staticmethod
    def get_text_feature(x, indices, clip_model,features):
        batch_size=features.shape[0]
        # batch_size = 1
        x = x + clip_model.positional_embedding.type(clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = clip_model.ln_final(x).type(clip_model.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        #indices = indices.repeat(batch_size)    #batch_size
        x = x[torch.arange(x.shape[0]), indices] @ clip_model.text_projection  #imp
        return x
       
     
    @property
    def n_prefix(self):
        return self.prompt_shape[0]

    @property
    def n_suffix(self):
        return self.prompt_shape[1]

    @property
    def device(self):
        return self.start_signal.device

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """

        repr = f"prefix_prompt:{self.n_prefix},suffix_prompt:{self.n_suffix},dimension:{self.prompt_dim}\n"
        repr = repr + "[Normal_Init(mu=0,std=0.02)]"
        return repr
