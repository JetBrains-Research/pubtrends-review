import logging
import os

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer

import review.config as cfg

logger = logging.getLogger(__name__)

EMBEDDINGS_ADDITIONAL = 20

FEATURES_NUMBER = 10
FEATURES_NN_INTERMEDIATE = 100
FEATURES_NN_OUT = 50
FEATURES_DROPOUT = 0.1

DECODER_DROPOUT = 0.1


def setup_cuda_device(model):
    logging.info('Setup single-device settings...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, device


def load_model(model_type, froze_strategy, article_len, features=False):
    logger.info(f'Loading model {model_type} {froze_strategy} {article_len} {features}')
    model = Summarizer(model_type, article_len, features)
    model.expand_posembs_ifneed()
    model.froze_backbone(froze_strategy)
    model.unfroze_head()
    logger.info(f'Parameters {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    return model


def get_token_id(tokenizer, tkn):
    return tokenizer.convert_tokens_to_ids([tkn])[0]


class Summarizer(nn.Module):
    """
    This is the main summarization model.
    See https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_tf_bert.py
    It operates the same input format as original BERT used underneath.
    See forward, evaluate params description.
    """
    enc_output: torch.Tensor
    dec_ids_mask: torch.Tensor
    encdec_ids_mask: torch.Tensor

    def __init__(self, model_type, article_len, additional_features, num_features=FEATURES_NUMBER):
        super(Summarizer, self).__init__()

        print(f'Initialize backbone and tokenizer for {model_type}')
        self.article_len = article_len
        if model_type == 'bert':
            self.backbone = self.initialize_bert()
            self.tokenizer = self.initialize_bert_tokenizer()
        elif model_type == 'roberta':
            self.backbone = self.initialize_roberta()
            self.tokenizer = self.initialize_roberta_tokenizer()
        else:
            raise Exception(f"Wrong model_type argument: {model_type}")
        self.backbone.resize_token_embeddings(EMBEDDINGS_ADDITIONAL + self.tokenizer.vocab_size)

        if additional_features:
            print('Adding additional features double fully connected nn')
            self.features = nn.Sequential(
                nn.Linear(num_features, FEATURES_NN_INTERMEDIATE),
                nn.LeakyReLU(),
                nn.Dropout(FEATURES_DROPOUT),
                nn.Linear(FEATURES_NN_INTERMEDIATE, FEATURES_NN_OUT)
            )
        else:
            self.features = None

        print('Initialize backbone embeddings pulling')

        def backbone_forward(input_ids, attention_mask, token_type_ids, position_ids):
            return self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids
            )

        self.encoder = lambda *args: backbone_forward(*args)[0]

        print('Initialize decoder')
        if additional_features:
            self.decoder = Classifier(768 + FEATURES_NN_OUT)  # Default BERT output with additional features
        else:
            self.decoder = Classifier(768)  # Default BERT output

    def expand_positional_embs_if_need(self):
        print('Expand positional embeddings if need')
        print('Positional embeddings', self.backbone.config.max_position_embeddings, self.article_len)
        if self.article_len > self.backbone.config.max_position_embeddings:
            old_maxlen = self.backbone.config.max_position_embeddings
            old_w = self.backbone.embeddings.position_embeddings.weight
            logging.info(f'Backbone pos embeddings expanded from {old_maxlen} upto {self.article_len}')
            self.backbone.embeddings.position_embeddings = nn.Embedding(
                self.article_len, self.backbone.config.hidden_size
            )
            self.backbone.embeddings.position_embeddings.weight[:old_maxlen].data.copy_(old_w)
            self.backbone.config.max_position_embeddings = self.article_len
            print('New positional embeddings', self.backbone.config.max_position_embeddings)

    @staticmethod
    def initialize_bert():
        return BertModel.from_pretrained(
            "bert-base-uncased", output_hidden_states=False
        )

    @staticmethod
    def initialize_bert_tokenizer():
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        tokenizer.BOS = "[CLS]"
        tokenizer.EOS = "[SEP]"
        tokenizer.PAD = "[PAD]"
        return tokenizer

    @staticmethod
    def initialize_roberta():
        backbone = RobertaModel.from_pretrained(
            'roberta-base', output_hidden_states=False
        )
        print('initialize token type emb, by default roberta doesnt have it')
        backbone.config.type_vocab_size = 2
        backbone.embeddings.token_type_embeddings = nn.Embedding(2, backbone.config.hidden_size)
        backbone.embeddings.token_type_embeddings.weight.data.normal_(
            mean=0.0, std=backbone.config.initializer_range
        )
        return backbone

    @staticmethod
    def initialize_roberta_tokenizer():
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', do_lower_case=True)
        tokenizer.BOS = "<s>"
        tokenizer.EOS = "</s>"
        tokenizer.PAD = "<pad>"
        return tokenizer

    def save(self, save_filename):
        """ Save model in filename

        :param save_filename: str
        """
        state = dict(
            encoder_dict=self.backbone.state_dict(),
            decoder_dict=self.decoder.state_dict()
        )
        if self.features:
            state['features_dict'] = self.features.state_dict()
        models_folder = os.path.expanduser(cfg.weights_path)
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        torch.save(state, f"{models_folder}/{save_filename}.pth")

    def load(self, load_filename):
        path = f"{os.path.expanduser(cfg.weights_path)}/{load_filename}.pth"
        state = torch.load(path, map_location=lambda storage, location: storage)
        self.backbone.load_state_dict(state['encoder_dict'])
        self.decoder.load_state_dict(state['decoder_dict'])
        if self.features:
            self.features.load_state_dict(state['features_dict'])

    def froze_backbone(self, froze_strategy):
        if froze_strategy == 'froze_all':
            for param in self.backbone.parameters():
                param.requires_grad_(False)

        elif froze_strategy == 'unfroze_last':
            for name, param in self.backbone.named_parameters():
                param.requires_grad_(
                    'encoder.layer.11' in name or
                    'encoder.layer.10' in name or
                    'encoder.layer.9' in name
                )

        elif froze_strategy == 'unfroze_all':
            for param in self.backbone.parameters():
                param.requires_grad_(True)

        else:
            raise Exception(f'Unsupported froze strategy {froze_strategy}')

    def unfroze_head(self):
        for param in self.decoder.parameters():
            param.requires_grad_(True)

    def forward(self, input_ids, attention_mask, token_type_ids, input_features=None):
        """
        :param input_ids: torch.Size([batch_size, article_len])
        Indices of input sequence tokens in the vocabulary.
        :param attention_mask: torch.Size([batch_size, article_len])
        Mask to avoid performing attention on padding token indices.
        Mask values selected in `[0, 1]`:
        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.
        :param token_type_ids: torch.Size([batch_size, article_len])
        Segment token indices to indicate first and second portions of the inputs.
        Indices are selected in `[0, 1]`:
        - 0 corresponds to a *sentence A* token,
        - 1 corresponds to a *sentence B* token.
        :return: scores | torch.Size([batch_size, summary_len])
        """

        # The output of [CLS] is inferred by all other words in this sentence.
        # This makes [CLS] a good representation for sentence-level classification.
        cls_mask = (input_ids == get_token_id(self.tokenizer, self.tokenizer.BOS))
        print(f'cls_mask {cls_mask.shape}')

        # Indices of positions of each input sequence tokens in the position embeddings.
        # position ids | torch.Size([batch_size, article_len])
        pos_ids = torch.arange(
            0,
            self.article_len,
            dtype=torch.long,
            device=input_ids.device
        ).unsqueeze(0).repeat(len(input_ids), 1)
        print(f'pos_ids {pos_ids.shape}')
        # extract bert embeddings | torch.Size([batch_size, article_len, d_bert])
        # for each word in the input, the BERT base internally creates a 768-dimensional output,
        # but for tasks like classification, we do not actually require the output for all the embeddings.
        # So by default, BERT considers only the output corresponding to the first token [CLS]
        # and drops the output vectors corresponding to all the other tokens.
        enc_output = self.encoder(input_ids, attention_mask, token_type_ids, pos_ids)

        if self.features:
            out_features = self.features(input_features)
            scores = self.decoder(torch.cat([enc_output[cls_mask], out_features], dim=-1))
        else:
            print('enc_output', enc_output.shape)
            print('enc_output[cls_mask]', enc_output[cls_mask].shape)
            scores = self.decoder(enc_output[cls_mask])
            print('scores', scores.shape)

        return scores

    def evaluate(self, input_ids, attention_mask, token_type_ids, input_features=None):
        """See forward for parameters and output description"""

        # The output of [CLS] is inferred by all other words in this sentence.
        # This makes [CLS] a good representation for sentence-level classification.
        cls_mask = (input_ids == get_token_id(self.tokenizer, self.tokenizer.BOS))

        # position ids | torch.Size([batch_size, article_len])
        pos_ids = torch.arange(
            0,
            self.article_len,
            dtype=torch.long,
            device=input_ids.device
        ).unsqueeze(0).repeat(len(input_ids), 1)

        # extract bert embeddings | torch.Size([batch_size, article_len, d_bert])
        enc_output = self.encoder(input_ids, attention_mask, token_type_ids, pos_ids)

        scores = []
        for eo, cm in zip(enc_output, cls_mask):
            if self.features:
                out_features = self.features(input_features)
                score = self.decoder.evaluate(torch.cat([eo[cm], out_features], dim=-1))
            else:
                score = self.decoder.evaluate(eo[cm])
            scores.append(score)
        return scores


class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(DECODER_DROPOUT)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(self.dropout(x)).squeeze(-1))

    def evaluate(self, x):
        return self.sigmoid(self.linear(self.dropout(x)).squeeze(-1))
