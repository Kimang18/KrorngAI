# Author: KrorngAI org.
# Date: February 2026


class CharTokenizer:
    def __init__(self, chars, special_tokens=['<s>', '</s>', '<pad>', '<unk>']):
        self.special_tokens = special_tokens
        # Unique characters + special tokens
        self.vocab = tuple(special_tokens[:1]) + tuple(chars) + tuple(special_tokens[1:])
        self.str_to_int = {s: i for i, s in enumerate(self.vocab)}
        self.int_to_str = {i: s for i, s in enumerate(self.vocab)}
        self.bos_id = self.str_to_int['<s>']
        self.eos_id = self.str_to_int['</s>']
        self.pad_id = self.str_to_int['<pad>']
        self.unk_id = self.str_to_int['<unk>']

    def __len__(self):
        return len(self.vocab)

    def encode(self, text, add_special_tokens=False):
        tokens = []
        i = 0
        while i < len(text):
            matched_special = False
            # Check for existing special tokens in the input string
            for spec in self.special_tokens:
                if text.startswith(spec, i):
                    tokens.append(self.str_to_int[spec])
                    i += len(spec)
                    matched_special = True
                    break

            if not matched_special:
                char = text[i]
                tokens.append(self.str_to_int.get(char, self.str_to_int['<unk>']))
                i += 1

        # Wrap with <s> and </s> if requested
        if add_special_tokens:
            tokens = [self.str_to_int['<s>']] + tokens + [self.str_to_int['</s>']]

        return tokens

    def decode(self, ids, ignore_special_tokens=False):
        if ignore_special_tokens:
            # Filter out any ID that belongs to the special_tokens list
            return "".join([self.int_to_str[i] for i in ids if self.int_to_str[i] not in self.special_tokens])

        return "".join([self.int_to_str.get(i, '<unk>') for i in ids])


def get_tokenizer(charset: str=None):
    if charset is None:
        kh_charset = "០១២៣៤៥៦៧៨៩កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវសហឡអឥឧឳឪឱឫឬឭឮឦឰឯាិីឹឺុូួើឿៀេែៃោៅំះៈ់៉៊៍័៏៌្ ។៕៖ៗ"
        en_charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
        charset = en_charset + kh_charset
    return CharTokenizer(charset)
