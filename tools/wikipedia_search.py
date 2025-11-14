"""Wikipedia search tool for general knowledge queries."""

import wikipedia


def search_wikipedia(query: str, sentences: int = 3) -> str:
    """Search Wikipedia for general knowledge and factual information.

    Args:
        query: Search query or topic
        sentences: Number of sentences to return from the summary

    Returns:
        Formatted string containing the Wikipedia summary
    """
    try:
        wikipedia.set_lang("en")
        search_results = wikipedia.search(query, results=3)

        if not search_results:
            return f"No Wikipedia articles found for '{query}'."

        page_title = search_results[0]
        summary = wikipedia.summary(page_title, sentences=sentences, auto_suggest=False)

        return f"Wikipedia - {page_title}:\n\n{summary}"

    except wikipedia.exceptions.DisambiguationError as e:
        options = e.options[:5]
        return (
            f"Multiple topics found for '{query}'. Please be more specific. "
            f"Options include: {', '.join(options)}"
        )

    except wikipedia.exceptions.PageError:
        return (
            f"No Wikipedia page found for '{query}'. "
            "Please try a different search term."
        )

    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"
