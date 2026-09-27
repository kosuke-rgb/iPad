"""文字単位のトークナイザ。日本語でも英語でもそのまま扱える。"""
import json


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        unknown = [c for c in text if c not in self.stoi]
        if unknown:
            raise ValueError(f"学習データに無い文字です: {''.join(sorted(set(unknown)))}")
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def to_json(self):
        return json.dumps(self.chars, ensure_ascii=False)

    @classmethod
    def from_json(cls, s):
        return cls(json.loads(s))
