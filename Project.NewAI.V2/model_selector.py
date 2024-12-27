# model_selector.py

"""
This file contains helper functions that decide which OpenAI model to use
for various tasks (e.g., embedding, summarization, or advanced analysis).
It allows you to centralize and evolve your model-selection logic without
changing the load-balancer code.
"""

def choose_model_for_task(task_type: str, message_length: int = 0) -> str:
    """
    Decides which GPT model to use based on the given task type and optional message length.

    Arguments:
    ----------
    task_type : str
        A string identifying the type of task. For example:
        "embedding", "complex_deep_analysis", or any summarization label
        such as "short_summarization", "long_summarization", etc.
    message_length : int
        (Optional) The approximate number of tokens or length of the user message.
        You could use this to apply different logic if needed. Default is 0.

    Returns:
    --------
    str
        The name of the model to be used (e.g., "gpt-3.5-turbo", "gpt-4",
        "text-embedding-ada-002", etc.).

    Updated Logic:
    -------------
    - "embedding" -> Always "text-embedding-ada-002".
    - "complex_deep_analysis" (or any advanced agentic tasks) -> "gpt-4".
    - All other cases (including short or long summaries) -> "gpt-3.5-turbo".

    This ensures GPT-3.5 is the default for typical summarization, while GPT-4
    is reserved for specialized tasks.
    """

    # Use text-embedding-ada-002 for embeddings
    if task_type == "embedding":
        return "text-embedding-ada-002"

    # Reserve gpt-4 for complex or deep analysis tasks
    elif task_type == "complex_deep_analysis":
        return "gpt-4"

    # Fallback to gpt-3.5-turbo for everything else (e.g., summarization)
    return "gpt-3.5-turbo"


def choose_embedding_model(task_type: str) -> str:
    """
    Decides which embedding model to use based on the task type or other criteria.

    Arguments:
    ----------
    task_type : str
        A string indicating the type of embedding task. For example:
        "large_document", "basic_embedding", etc.

    Returns:
    --------
    str
        The name of the embedding model to be used (e.g., "text-embedding-ada-002",
        "text-embedding-babbage-001", etc.).

    Default Behavior:
    ----------------
    - If task_type == "large_document": Use "text-embedding-babbage-001".
    - Else: Use "text-embedding-ada-002".
    """

    if task_type == "large_document":
        return "text-embedding-babbage-001"

    # Default if no special case is matched
    return "text-embedding-ada-002"
