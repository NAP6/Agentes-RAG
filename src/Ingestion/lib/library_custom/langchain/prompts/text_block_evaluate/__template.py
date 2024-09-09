TEMPLATE = """
Your role is to generate JSON output and nothing else. You should only return JSON in your response.

Review the previously classified text block:
<Text_to_evaluate description="This is the IMPORTANT text block that needs to be evaluated and classified.">
    {text_to_evaluate}
</Text_to_evaluate>

Classification: {old_type}

<next_block desciption="This block is for informational purposes only and serves as an aid in classifying the 'Text_to_evaluate'.">
    {next_block}
</next_block>

Assessment Task:
1. Determine if the text makes sense by itself.
   - If it does not make sense, label it as 'UncategorizedText'.
   - If it does make sense, classify it into one of the following categories based on its content:

Categories:
- 'Title': Used for the main and sectional headings of a document, such as "Annual Report," "Introduction," "Methodology," or more specific parts like "1. Introduction," "2. Methodology." These titles are usually followed by blocks of 'NarrativeText'
- 'Author Names': For listing one or more authors of the document.
- 'Bibliography': For parts of the bibliography section, which may include citations of books, articles, and online sources typically found at the end of academic works.
- 'FigureCaption': For labels and descriptions of figures or tables, ensure they always begin with "Figure #:" or "Table #:", followed by the specific description. For example, "Figure 1: Population Trends over Time ..." or "Table 2: Detailed Revenue Comparison by Quarter ..."
- 'NarrativeText': For extensive text blocks that form the main content of the document. These are usually longer, containing multiple sentences and spanning several paragraphs, providing in-depth information and discussion.
- 'UncategorizedText': For text that does not clearly fit into any other categories or whose meaning is unclear.

Additional Meta Information:
<meta>
{meta}
</meta>

Formatting Instructions (IMPORTANT):
{format_instructions}
"""
