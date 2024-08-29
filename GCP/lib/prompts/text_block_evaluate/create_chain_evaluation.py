from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from .__TextEvaluation import TextEvaluation
from .__template import TEMPLATE
from typing import (Dict)


def create_chain_evaluation(
        model: BaseLLM,
        meta: Dict
):
    """
    Create a chain for evaluating a text block.

    :param model:
    :param meta:

    :return: chain
    """
    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=TextEvaluation)

    prompt = PromptTemplate(
        template=TEMPLATE,
        input_variables=["text_to_evaluate", "old_type", "next_block"],
        partial_variables={
            "meta": meta,
            "format_instructions": parser.get_format_instructions(),
        },
    )

    # Assuming 'model' is an instance of a language model from langchain
    chain = prompt | model | parser

    return chain
