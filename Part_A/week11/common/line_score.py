def calculate_line_scores(survival_indices, line_to_tokens):

    line_scores = [0.0] * len(line_to_tokens)
    slot_count = 0

    # 모든 layer 순회
    for layer_idx, layer_indices in survival_indices.items():

        num_kv_heads = layer_indices.shape[1]

        # 현재 layer의 모든 KV head 순회
        for kv_head_idx in range(num_kv_heads):

            kept_tokens = set(
                layer_indices[0, kv_head_idx].tolist()
            )

            # 각 line의 token survival ratio 계산

            for line_idx, token_indices in enumerate(line_to_tokens):

                if len(token_indices) == 0:
                    continue

                kept_count = sum(
                    token_idx in kept_tokens
                    for token_idx in token_indices
                )

                survival_ratio = (
                    kept_count / len(token_indices)
                )

                line_scores[line_idx] += survival_ratio

            slot_count += 1

    if slot_count == 0:
        raise RuntimeError(
            "압축된 layer x KV head slot이 없습니다."
        )

    # 모든 slot의 평균 계산
    for line_idx, token_indices in enumerate(line_to_tokens):

        if len(token_indices) == 0:
            continue

        line_scores[line_idx] /= slot_count

    return line_scores