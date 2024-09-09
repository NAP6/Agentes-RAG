from langchain_core.pydantic_v1 import BaseModel, Field
class TextEvaluation(BaseModel):
    makes_sense: bool = Field(
        default=False,
        description="Indicates whether the text is coherent and understandable."
    )
    block_type: str = Field(
        default="",
        description="Specifies the category of the text block, such as 'Title', 'Author Names', 'Bibliography', etc."
    )
    description: str = Field(
        default="",
        description="Provides a concise summary or explanation of the text block's content."
    )
