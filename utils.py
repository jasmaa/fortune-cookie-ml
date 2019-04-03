"""Sentence processing utilities
"""

import re

def clean_sentence(data):
    """Cleans sentence for indexing
    """
    data = data.split(" ")
    data = list(map(lambda x: x.lower(), data))
    data = list(map(lambda x: re.sub(r'\W+', '', x), data))
    return data