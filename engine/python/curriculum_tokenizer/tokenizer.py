"""
Python wrapper around the C++ tokenizer with Hugging Face PreTrainedTokenizer compatibility.
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Union, Optional, Dict, Any

try:
    from ._cpp_tokenizer import Tokenizer as _CTokenizer
except ImportError:
    raise ImportError(
        "C++ extension not built. Run: "
        "pip install pybind11 && python setup.py build_ext --inplace"
    )


class CurriculumTokenizer:
    """
    Fast BPE tokenizer with curriculum training support.

    Compatible with Hugging Face Transformers tokenization interface.
    """

    def __init__(self, vocab_file: str, merges_file: str, **kwargs):
        """
        Args:
            vocab_file: Path to vocab.json
            merges_file: Path to merges.txt
            **kwargs: Additional config (special_tokens, etc.)
        """
        self._tok = _CTokenizer()

        vocab_path = os.path.abspath(vocab_file)
        merges_path = os.path.abspath(merges_file)
        self._vocab_path = vocab_path
        self._merges_path = merges_path
        self._config = dict(kwargs)

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
        if not os.path.exists(merges_path):
            raise FileNotFoundError(f"Merges file not found: {merges_path}")

        ok = self._tok.load(vocab_path, merges_path)
        if not ok:
            raise RuntimeError("Failed to load tokenizer. Check vocab.json and merges.txt format.")

        native_vocab_size = self._tok.vocab_size()
        config_vocab_size = kwargs.get("vocab_size")
        self.special_tokens = list(kwargs.get("special_tokens", ["<|endoftext|>"]))
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("Duplicate special-token names")
        declared_ids = kwargs.get("special_token_ids", kwargs.get("reserved_token_ids", {}))
        if set(declared_ids) - set(self.special_tokens):
            raise ValueError("Configured control IDs contain an undeclared special token")
        self._special_token_ids = {
            token: declared_ids.get(token, self._tok.token_to_id(token))
            for token in self.special_tokens
        }
        self._special_aliases = {}
        for token, token_id in kwargs.get("compatibility_reserved_ids", {}).items():
            if token_id in self._special_aliases:
                raise ValueError("Duplicate decode-only alias ID")
            self._special_aliases[token_id] = token
        inferred_size = max([native_vocab_size, *[i + 1 for i in self._special_token_ids.values()], *[i + 1 for i in self._special_aliases]])
        expected_size = int(config_vocab_size) if config_vocab_size is not None else inferred_size
        self._tok.configure_special_tokens(self._special_token_ids, self._special_aliases, expected_size)
        self.vocab_size = self._tok.vocab_size()
        self._skip_special_ids = set(self._special_token_ids.values()) | set(self._special_aliases)

        # Build vocab mapping for Python access
        self._vocab = {}
        self._id_to_token = {}
        for i in range(self.vocab_size):
            token = self._tok.id_to_token(i)
            if token:
                if i not in self._special_aliases:
                    self._vocab[token] = i
                self._id_to_token[i] = token

        # Special token IDs
        self.eos_token = kwargs.get("eos_token", "<|endoftext|>")
        eos_id = self._tok.token_to_id(self.eos_token) if self.eos_token is not None else -1
        self.eos_token_id = eos_id if eos_id >= 0 else None
        self.pad_token = kwargs.get("pad_token", "<|endoftext|>")
        pad_id = self._tok.token_to_id(self.pad_token) if self.pad_token is not None else -1
        self.pad_token_id = pad_id if pad_id >= 0 else None

    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs):
        """Load tokenizer from a directory containing vocab.json and merges.txt."""
        model_dir = Path(model_dir)
        vocab_file = model_dir / "vocab.json"
        merges_file = model_dir / "merges.txt"

        # Load config if present
        config_file = model_dir / "tokenizer_config.json"
        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)
            kwargs = {**config, **kwargs}

        return cls(str(vocab_file), str(merges_file), **kwargs)

    def encode(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        **kwargs
    ) -> Union[List[int], Dict[str, List[int]]]:
        """
        Encode text to token IDs.

        Args:
            text: String or list of strings to encode
            add_special_tokens: Whether to add EOS token

        Returns:
            List of token IDs, or dict with "input_ids" key (HF style)
        """
        if isinstance(text, list):
            return self.encode_batch(text, add_special_tokens=add_special_tokens, **kwargs)

        ids = self._tok.encode(text)

        if add_special_tokens and self.eos_token_id is not None:
            ids.append(self.eos_token_id)

        return ids

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        num_threads: int = 0,
        **kwargs
    ) -> Dict[str, List[List[int]]]:
        """Encode in input order; 0 selects up to four native workers for large batches."""
        results = self._tok.encode_batch(texts, num_threads=num_threads)

        if add_special_tokens and self.eos_token_id is not None:
            for ids in results:
                ids.append(self.eos_token_id)

        return {"input_ids": results}

    def set_cache_capacity(self, capacity: int) -> None:
        """Bound cached exact pieces; zero disables caching."""
        self._tok.set_cache_capacity(capacity)

    def clear_cache(self) -> None:
        """Clear cached encodings and reset hit/miss counters."""
        self._tok.clear_cache()

    def cache_info(self) -> Dict[str, int]:
        """Return piece-cache statistics; payload bytes exclude container overhead."""
        return self._tok.cache_info()

    def decode(
        self,
        ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Decode token IDs to text.

        Args:
            ids: List of token IDs or batch of lists
            skip_special_tokens: Whether to remove special tokens

        Returns:
            Decoded string or list of strings
        """
        if ids and isinstance(ids[0], list):
            return [self.decode_single(seq, skip_special_tokens) for seq in ids]

        return self.decode_single(ids, skip_special_tokens)

    def decode_single(self, ids: List[int], skip_special_tokens: bool) -> str:
        """Decode a single sequence."""
        if skip_special_tokens and self._skip_special_ids:
            ids = [id for id in ids if id not in self._skip_special_ids]

        return self._tok.decode(ids)

    def __call__(self, text: Union[str, List[str]], **kwargs):
        """Hugging Face style __call__ method."""
        result = self.encode(text, **kwargs)

        if isinstance(result, dict):
            return result

        return {"input_ids": result}

    def convert_tokens_to_ids(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert token strings to IDs."""
        fallback = self.eos_token_id if self.eos_token_id is not None else 0
        if isinstance(tokens, str):
            native_id = self._tok.token_to_id(tokens)
            if native_id >= 0:
                return native_id
            return self._vocab.get(tokens, fallback)
        return [self.convert_tokens_to_ids(token) for token in tokens]

    def convert_ids_to_tokens(self, ids: Union[int, List[int]]) -> Union[str, List[str]]:
        """Convert IDs to token strings."""
        if isinstance(ids, int):
            return self._id_to_token.get(ids, "")
        return [self._id_to_token.get(i, "") for i in ids]

    def save_pretrained(self, save_dir: str):
        """Save tokenizer files to a directory."""
        os.makedirs(save_dir, exist_ok=True)

        config = {**self._config,
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "special_token_ids": self._special_token_ids,
            "eos_token": self.eos_token,
            "pad_token": self.pad_token,
        }

        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        for source, name in [(self._vocab_path, "vocab.json"), (self._merges_path, "merges.txt")]:
            destination = os.path.abspath(os.path.join(save_dir, name))
            if source != destination:
                shutil.copyfile(source, destination)

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return f"CurriculumTokenizer(vocab_size={self.vocab_size})"
