from collections import defaultdict
from typing import Dict, List, Tuple, Union, overload

import torch
from fitbert.delemmatize import Delemmatizer
from fitbert.utils import mask as _mask
from functional import pseq, seq        #functional seq, pseq
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    AlbertForMaskedLM,
    AlbertTokenizer
)


class FitBert:
    def __init__(
        self,
        model=None,
        tokenizer=None,
        model_name="bert-large-uncased",
        mask_token="***mask***",
        disable_gpu=False,
    ):
        self.mask_token = mask_token
        self.delemmatizer = Delemmatizer()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not disable_gpu else "cpu"
        )
        print("using model:", model_name)
        print("device:", self.device)

        if not model:
            if "distilbert" in model_name:
                self.bert = DistilBertForMaskedLM.from_pretrained(model_name)
            elif "Albert" in model_name:
                self.bert = AlbertForMaskedLM.from_pretrained(model_name)
            else:
                self.bert = BertForMaskedLM.from_pretrained(model_name)
            self.bert.to(self.device)
        else:
            self.bert = model

        if not tokenizer:
            if "distilbert" in model_name:
                self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            elif "Albert" in model_name:
                self.tokenizer = AlbertTokenizer.from_pretrained(bert-large-uncased)
            else:
                self.tokenizer = BertTokenizer.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer

        self.bert.eval()

    @staticmethod
    def softmax(x):
        return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)

    @staticmethod
    #check is multi options
    def is_multi(options: List[str]) -> bool:
        return seq(options).filter(lambda x: len(x.split()) != 1).non_empty()

    def mask(self, s: str, span: Tuple[int, int]) -> Tuple[str, str]:
        return _mask(s, span, mask_token=self.mask_token)       #self.mask_token = "***mask***"

    def _tokens_to_masked_ids(self, tokens, mask_ind):          # them cu phap phu hop Bert (CLS,SEP,MASK)
        masked_tokens = tokens[:]
        masked_tokens[mask_ind] = "[MASK]"
        masked_tokens = ["[CLS]"] + masked_tokens + ["[SEP]"]
        masked_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
        return masked_ids

    def _get_sentence_probability(self, sent: str) -> float:    

        tokens = self.tokenizer.tokenize(sent)                  #chuyen doi sang token [nx1]
        input_ids = (
            seq(tokens)
            .enumerate()
            .starmap(lambda i, x: self._tokens_to_masked_ids(tokens, i))
            .list()
        )
       
        tens = torch.tensor(input_ids).to(self.device)          # chuyen vao thanh tensor [n x 1]
        with torch.no_grad():
            preds = self.bert(tens)[0]                          # qua model de tinh toan ket qua [n x m]
            probs = self.softmax(preds)                         # ket qua truyen qua ham Softmax 
            tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)   #convert token Z[nx1] to dang dau vao Bert
            
            # prob co ve giong nhu tich cua tat ca prods
            prob = (
                seq(tokens_ids)
                .enumerate()
                .starmap(lambda i, x: float(probs[i][i + 1][x].item()))
                .reduce(lambda x, y: x * y, 1)
            )   
               
            del tens, preds, probs, tokens, input_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return prob
        
    # Vlookup dict in package delemmatizer --- thay the tu co duoi 's'
    def _delemmatize_options(self, options: List[str]) -> List[str]:
        options = (
            seq(options[:])
            .flat_map(lambda x: self.delemmatizer(x))
            .union(options)
            .list()
        )
        return options
    # phong doan don / guess single
    def guess_single(self, masked_sent: str, n: int = 1):

        pre, post = masked_sent.split(self.mask_token)

        tokens = ["[CLS]"] + self.tokenizer.tokenize(pre)           #bien thanh token
        target_idx = len(tokens)
        tokens += ["[MASK]"]
        tokens += self.tokenizer.tokenize(post) + ["[SEP]"]

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)    # bien thanh indexs
        tens = torch.tensor(input_ids).unsqueeze(0)                 # truyen vao tensor
        tens = tens.to(self.device)
        with torch.no_grad():
            preds = self.bert(tens)[0]                              # truyen vao model
            probs = self.softmax(preds)                             # lai truyen vao ham softmax

            pred_top = torch.topk(probs[0, target_idx], n)          # prediction top???
            pred_prob = pred_top[0].tolist()                        # predicted xac suat???
            pred_idx = pred_top[1].tolist()

            pred_tok = self.tokenizer.convert_ids_to_tokens(pred_idx)   # chuyen nguoc ve dang token

            # xoa het cac thu. Giai phong bo nho
            del pred_top, pred_idx, tens, preds, probs, input_ids, tokens
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return pred_tok, pred_prob
    # rank don :)))            words : List[str] gom cac dap an
    def rank_single(self, masked_sent: str, words: List[str]):

        pre, post = masked_sent.split(self.mask_token)              # tach phan truoc va sau ***MASK***

        tokens = ["[CLS]"] + self.tokenizer.tokenize(pre)
        target_idx = len(tokens)
        tokens += ["[MASK]"]
        tokens += self.tokenizer.tokenize(post) + ["[SEP]"]         # tokenizer for bert

        # bien doi dap an
        words_ids = (                                                       
            seq(words)
            .map(lambda x: self.tokenizer.tokenize(x))              # tokenizer for bert (dap an)
            .map(lambda x: self.tokenizer.convert_tokens_to_ids(x)[0])      # convert token to ids (dap an)
        )

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)     # convert token to ids (cau hoi)
        tens = torch.tensor(input_ids).unsqueeze(0)
        tens = tens.to(self.device)
        with torch.no_grad():
            preds = self.bert(tens)[0]                                 # chay cau hoi qua BerMaskLM
            probs = self.softmax(preds)

            # rank doi ???
            ranked_pairs = (
                seq(words_ids)
                .map(lambda x: float(probs[0][target_idx][x].item()))       # map
                .zip(words)
                .sorted(key=lambda x: x[0], reverse=True)               # sap xep dao nguoc
            )
            
            # ranked_options =  map (ranked_pairs) [1]
            # ranked_options_prob ... la gi?
            ranked_options = (seq(ranked_pairs).map(lambda x: x[1])).list()    
            ranked_options_prob = (seq(ranked_pairs).map(lambda x: x[0])).list()    # rank xac suat cua options

            del tens, preds, probs, tokens, words_ids, input_ids
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return ranked_options, ranked_options_prob

    # rank nhieu 
    def rank_multi(self, masked_sent: str, options: List[str]):
        ranked_pairs = (
            seq(options)
            .map(lambda x: masked_sent.replace(self.mask_token, x))         # thay the OPTIONS vao MASK
            .map(lambda x: self._get_sentence_probability(x))               # get xac suat
            .zip(options)                                                   # zip : ghep lai 0 la rank, 1 la xac suat
            .sorted(key=lambda x: x[0], reverse=True)                       # sap xep dao nguoc
        )
        ranked_options = (seq(ranked_pairs).map(lambda x: x[1])).list()     # rank options
        ranked_options_prob = (seq(ranked_pairs).map(lambda x: x[0])).list()    # rank xac suat options
        return ranked_options, ranked_options_prob
    
    
    # don gian hoa lua chon / tra ve dap an-chuoi da thay the - tu bat dau - tu ket thuc
    def _simplify_options(self, sent: str, options: List[str]):

        options_split = seq(options).map(lambda x: x.split())

        trans_start = list(zip(*options_split))

        start = (
            seq(trans_start)
            .take_while(lambda x: seq(x).distinct().len() == 1)
            .map(lambda x: x[0])
            .list()
        )

        options_split_reversed = seq(options_split).map(
            lambda x: seq(x[len(start) :]).reverse()
        )

        trans_end = list(zip(*options_split_reversed))

        end = (
            seq(trans_end)
            .take_while(lambda x: seq(x).distinct().len() == 1)
            .map(lambda x: x[0])
            .list()
        )

        start_words = seq(start).make_string(" ")
        end_words = seq(end).reverse().make_string(" ")

        options = (
            seq(options_split)
            .map(lambda x: x[len(start) : len(x) - len(end)])
            .map(lambda x: seq(x).make_string(" ").strip())
            .list()
        )

        sub = seq([start_words, self.mask_token, end_words]).make_string(" ").strip()
        sent = sent.replace(self.mask_token, sub)

        return options, sent, start_words, end_words

    
    # tinh rank cua dap an
    def rank(
        self,
        sent: str,
        options: List[str],
        delemmatize: bool = False,
        with_prob: bool = False,
    ):
        """
        Rank a list of candidates

        returns: Either a List of strings,
        or if `with_prob` is True, a Tuple of List[str], List[float]

        """
        
        # tao list dap an
        options = seq(options).distinct().list()
        
        # VloopUP ... thay the dap an duoi s
        if delemmatize:
            options = seq(self._delemmatize_options(options)).distinct().list()
        
        # la co 1 dap an thi thoi :))))
        if seq(options).len() == 1:
            return options
        
        # don gina hoa dap an
        options, sent, start_words, end_words = self._simplify_options(sent, options)
        
        # neu co nhieu dap an
        if self.is_multi(options):
            ranked, prob = self.rank_multi(sent, options)
        else:
            ranked, prob = self.rank_single(sent, options)

        ranked = (
            seq(ranked)
            .map(lambda x: [start_words, x, end_words])
            .map(lambda x: seq(x).make_string(" ").strip())
            .list()
        )
        if with_prob:
            return ranked, prob             # tra lai rank va xac suat
        else:
            return ranked                   # chi tra lai rank

    def rank_with_prob(self, sent: str, options: List[str], delemmatize: bool = False):
        ranked, prob = self.rank(sent, options, delemmatize, True)
        return ranked, prob
    
    # du doan
    def guess(self, sent: str, n: int = 1) -> List[str]:
        pred_tok, _ = self.guess_single(sent, n)
        return pred_tok
    # du doan kem xac suat
    def guess_with_prob(self, sent: str, n: int = 1):
        pred_tok, pred_prob = self.guess_single(sent, n)
        return pred_tok, pred_prob
    
    # tra lai cau hoan chinh
    def fitb(self, sent: str, options: List[str], delemmatize: bool = False) -> str:
        ranked = self.rank(sent, options, delemmatize)
        best_word = ranked[0]
        return sent.replace(self.mask_token, best_word)
    
    # tra lai cac dap an dung
    def mask_fitb(self, sent: str, span: Tuple[int, int]) -> str:
        masked_str, replaced = self.mask(sent, span)
        options = [replaced]
        return self.fitb(masked_str, options, delemmatize=True)
