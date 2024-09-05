from functools import lru_cache
import os

ENDOFTEXT = "<|endoftext|>"
FIM_PREFIX = "<|fim_prefix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_SUFFIX = "<|fim_suffix|>"
ENDOFPROMPT = "<|endofprompt|>"

@lru_cache(maxsize=100)
def split_tokens(
        text: str, tokenizer_name: str = "cl100k_base", encoder=None, decoder=None
):
    """
    Split a given text into subword tokens and their corresponding token IDs.

    Args:
        text (str): The input text to be tokenized.
        tokenizer_name (str, optional): The name of the tokenizer to use. If provided, the function
            will use the tiktoken library for tokenization. Default is `cl100k_base`.
        encoder (callable, optional): A custom encoder function that takes a string and returns a list
            of token IDs. Required if tokenizer_name is not provided.
        decoder (callable, optional): A custom decoder function that takes a list of token IDs and
            returns the corresponding string. Required if tokenizer_name is not provided.

    Returns:
        tuple: A tuple containing two lists:
            - fragments (list): A list of strings representing the subword tokens.
            - tokenids (list): A list of integers representing the token IDs.

    Raises:
        ValueError: If neither tokenizer_name nor custom encoder and decoder functions are provided.
    """

    if tokenizer_name is not None:
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe

        current_dir = os.path.dirname(os.path.abspath(__file__))
        tokenizer_file_path = os.path.join(current_dir, 'cl100k_base.tiktoken')

        mergeable_ranks = load_tiktoken_bpe(
            tokenizer_file_path,
            expected_hash="223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        )

        special_tokens = {
            ENDOFTEXT: 100257,
            FIM_PREFIX: 100258,
            ENDOFPROMPT: 100276,
        }

        tokenizer = tiktoken.Encoding(
            name='cl100k_base',
            pat_str= r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens
        )

        encoder = tokenizer.encode
        decoder = tokenizer.decode
    elif encoder is None and decoder is None:
        raise ValueError(
            "Either tokenizer_name or custom encoder and decoder functions must be provided."
        )

    tokenids = tokenizer.encode(text)
    fragments = [
        repr(tokenizer.decode([t]).replace(" ", "␣")).strip("'") for t in tokenids
    ]

    return fragments, tokenids
