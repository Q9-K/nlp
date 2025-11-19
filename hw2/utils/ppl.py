def compute_perplexity(log_likelihood, num_tokens):
    """
    Compute the perplexity given the log likelihood and number of tokens.

    Args:
        log_likelihood (float): The log likelihood of the sequence.
        num_tokens (int): The number of tokens in the sequence.

    Returns:
        float: The computed perplexity.
    """
    if num_tokens <= 0:
        raise ValueError("Number of tokens must be greater than zero.")
    
    perplexity = 2 ** (-log_likelihood / num_tokens)
    return perplexity
if __name__ == "__main__":
    # Example usage
    log_likelihood = -10.0  # Example log likelihood
    num_tokens = 5          # Example number of tokens
    ppl = compute_perplexity(log_likelihood, num_tokens)
    print(f"Perplexity: {ppl}")