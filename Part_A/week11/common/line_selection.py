def select_top_k_lines(
    line_scores,
    line_to_tokens,
    line_keep_ratio
):
    valid_line_indices = [
        line_idx
        for line_idx, token_indices in enumerate(line_to_tokens)
        if len(token_indices) > 0
    ]

    num_lines_to_keep = int(
        len(valid_line_indices) * line_keep_ratio
    )

    ranked_lines = sorted(
        valid_line_indices,
        key=lambda line_idx: (
            -line_scores[line_idx],
            line_idx
        )
    )

    top_k_lines = ranked_lines[:num_lines_to_keep]

    return top_k_lines