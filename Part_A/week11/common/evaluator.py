from .line_score import calculate_line_scores
from .line_selection import select_top_k_lines
from .retention import calculate_gt_retention


def evaluate_line_retention(
        survival_indices,
        line_to_tokens,
        gt_lines,
        line_keep_ratio
):
    # 1. GT 범위 검증
    invalid_gt_lines = [
        gt_line
        for gt_line in gt_lines
        if gt_line < 0 or gt_line >= len(line_to_tokens)
    ]

    if invalid_gt_lines:
        raise ValueError(
            f"현재 입력 번위를 벗어난 GT line: {invalid_gt_lines}"
        )

    # 2. KV 생존 토큰 → LineScore
    line_scores = calculate_line_scores(
        survival_indices,
        line_to_tokens
    )

    # 3. LineScore → Top-K Lines
    top_k_lines = select_top_k_lines(
        line_scores,
        line_to_tokens,
        line_keep_ratio
    )

    # 4. Top-K Lines → GT Retention
    retained_gt_lines, retention = calculate_gt_retention(
        top_k_lines,
        gt_lines
    )

    return (
        line_scores,
        top_k_lines,
        retained_gt_lines,
        retention
    )