# model_selector.py

"""
This file contains helper functions that decide which OpenAI model to use
for various tasks (e.g., short summarization, long summarization, embeddings).
It allows you to centralize and evolve your model-selection logic without
changing the load-balancer code.
"""

def choose_model_for_task(task_type: str, message_length: int = 0) -> str:
    """
    Decides which GPT model to use based on the given task type and optional message length.

    Arguments:
    ----------
    task_type : str
        A string identifying the type of task. Examples might include:
        "embedding", "short_summarization", "long_research_summarization", etc.
    message_length : int
        (Optional) The approximate number of tokens or length of the user message.
        You can use this to switch to GPT-4 if the message is very long, or stay
        on GPT-3.5 if it's short. Default is 0.

    Returns:
    --------
    str
        The name of the model to be used (e.g., "gpt-3.5-turbo", "gpt-4",
        "text-embedding-ada-002", etc.).
    """

    # Example logic below. Adjust according to your needs:
    if task_type == "embedding":
        # Use the standard embedding model
        return "text-embedding-ada-002"

    elif task_type == "long_research_summarization":
        # For complex or long tasks, choose GPT-4
        return "gpt-4"

    elif task_type == "short_summarization":
        # For shorter or simpler tasks, choose GPT-3.5-turbo
        return "gpt-3.5-turbo"

    else:
        # Fallback or default to GPT-3.5-turbo
        return "gpt-3.5-turbo"


def choose_embedding_model(task_type: str) -> str:
    """
    Decides which embedding model to use based on the task type or other criteria.

    Arguments:
    ----------
    task_type : str
        A string that indicates the type of embedding task. You might categorize
        tasks as "large_document", "basic_embedding", etc.

    Returns:
    --------
    str
        The name of the embedding model to be used (e.g., "text-embedding-ada-002",
        "text-embedding-babbage-001", etc.).
    """

    # Example logic below. Adjust as needed:
    if task_type == "large_document":
        # Possibly use a different model if dealing with very large content
        return "text-embedding-babbage-001"

    # Default if no special case is matched
    return "text-embedding-ada-002"
