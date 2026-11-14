def build_token_line_mapping(lines, offset_mapping):

    # 각 로그 라인의 문자열 시작/끝 위치 계산
    cursor = 0
    line_spans = []

    for line in lines:
        line_start = cursor
        cursor += len(line)
        line_end = cursor

        line_spans.append([line_start, line_end])

    # Token → Line 매핑
    token_to_line = []
    line_chk = 0

    for token in offset_mapping[0]:
        token_start = token[0].item()

        while token_start >= line_spans[line_chk][1]:
            line_chk += 1

        token_to_line.append(line_chk)

    # Line → Token 매핑
    line_to_tokens = [[] for _ in range(len(lines))]

    for token_idx, line_idx in enumerate(token_to_line):
        line_to_tokens[line_idx].append(token_idx)

    return token_to_line, line_to_tokens