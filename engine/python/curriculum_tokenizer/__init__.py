"""
Curriculum Tokenizer - Fast C++ BPE inference with Hugging Face compatibility.

Usage:
    from curriculum_tokenizer import CurriculumTokenizer

    tok = CurriculumTokenizer.from_pretrained("./artifacts")
    ids = tok.encode("Hello world!")
    text = tok.decode(ids)
"""

from .tokenizer import CurriculumTokenizer

__version__ = "0.3.0"
__all__ = ["CurriculumTokenizer"]
