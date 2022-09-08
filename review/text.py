import nltk


def split_text(text):
    return nltk.tokenize.sent_tokenize(text)


def get_token_id(tokenizer, tkn):
    return tokenizer.convert_tokens_to_ids([tkn])[0]


def preprocess_text(text, max_len, tokenizer):
    """
    Preprocess text for BERT / ROBERTA model.
    NOTE: not all the text can be processed because of max_len.
    IMPORTANT: preprocessed text may contain more than one [CLS] token!!! and more than two sentences!!!
    :param text: list(list(str))
    :param max_len: maximum length of preprocessing
    :param tokenizer: BERT or ROBERTA tokenizer
    :return:
        ids | tokenized ids of length max_len, 0 if padding
        attention_mask | list(str) 1 if real token, not padding
        token_type_ids | 0-1 for different sentences
        n_sents | number of actual sentences encoded
    """
    sents = [
        [tokenizer.BOS] + tokenizer.tokenize(sent) + [tokenizer.EOS] for sent in text
    ]

    ids, token_type_ids, segment_signature = [], [], 0
    n_sents = 0
    for i, s in enumerate(sents):
        if len(ids) + len(s) <= max_len:
            n_sents += 1
            ids.extend(tokenizer.convert_tokens_to_ids(s))
            token_type_ids.extend([segment_signature] * len(s))
            segment_signature = (segment_signature + 1) % 2
        else:
            break
    attention_mask = [1] * len(ids)

    pad_len = max(0, max_len - len(ids))
    ids += [get_token_id(tokenizer, tokenizer.PAD)] * pad_len
    attention_mask += [0] * pad_len
    token_type_ids += [segment_signature] * pad_len
    assert len(ids) == len(attention_mask)
    assert len(ids) == len(token_type_ids)
    return ids, attention_mask, token_type_ids, n_sents


# Overlap helps to keep context and connect different parts of abstract
OVERLAP = 5


def text_to_data(text, max_len, tokenizer):
    """
    This is the main entry point, which will be called from PubTrends review feature.
    """
    text = split_text(text)
    total_sents = 0
    data = []
    while total_sents < len(text):
        offset = max(0, total_sents - OVERLAP)
        # Preprocessing BERT cannot encode all the text,
        # only limited number of sentences per single model run is supported.
        ids, attention_mask, token_type_ids, n_sents = \
            preprocess_text(text[offset:], max_len, tokenizer)
        if offset + n_sents <= total_sents:
            total_sents += 1
            continue
        data.append((ids, attention_mask, token_type_ids, offset, text[offset: offset + n_sents]))
        total_sents = offset + n_sents
    return data
