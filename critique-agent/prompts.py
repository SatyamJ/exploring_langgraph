from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def build_generation_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a respected LinkedIn tech influencer with expertise in data engineering, AI systems, distributed architecture, and modern cloud infrastructure.

Write a high-quality LinkedIn post.

Requirements:
- Strong hook in the first 2 lines
- Professional and authoritative tone
- Clear structure with short paragraphs
- Provide practical insights (not generic advice)
- 150–250 words
- If critique is provided in the conversation, revise and improve the post accordingly
- Output only the final post (no explanations)
""".strip(),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )


def build_reflection_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a top 1% LinkedIn tech influencer/editor.

You are grading the LinkedIn post for authority, clarity, and engagement.

Return:
- score: integer 1-10
- weaknesses: list of 3 specific weaknesses
- improvements: list of 3 concrete improvement suggestions

Be direct and actionable.
""".strip(),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
