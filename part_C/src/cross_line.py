import torch


def convert_to_line_attention(
    layer_attentions,
    token_map
):

    # ---------------------------------
    # token_map에서 실제 line_id 목록 생성
    #
    # token이 등장한 순서를 그대로 유지
    # ---------------------------------
    line_ids = []

    for token_info in token_map:
        line_id = token_info.line_id

        if line_id is None:
            continue

        if line_id not in line_ids:
            line_ids.append(line_id)

    if not line_ids:
        raise ValueError(
            "token_map에 유효한 line_id가 없습니다."
        )

    # line_id -> matrix index
    #
    # 예:
    # line_ids = [10, 11, 12]
    #
    # 10 -> 0
    # 11 -> 1
    # 12 -> 2
    line_to_index = {
        line_id: line_idx
        for line_idx, line_id in enumerate(line_ids)
    }

    num_tokens = len(token_map)
    num_lines = len(line_ids)

    device = layer_attentions[0].device
    dtype = layer_attentions[0].dtype

    # ---------------------------------
    # token -> line membership matrix
    #
    # shape:
    # [token, line]
    #
    # 예:
    #
    #          L0  L1
    # token0    1   0
    # token1    1   0
    # token2    0   1
    # token3    0   1
    # ---------------------------------
    member = torch.zeros(
        (num_tokens, num_lines),
        dtype=dtype,
        device=device
    )

    for token_info in token_map:
        token_idx = token_info.token_idx
        line_id = token_info.line_id

        if line_id is None:
            continue

        line_idx = line_to_index[line_id]
        member[token_idx, line_idx] = 1

    # ---------------------------------
    # 각 query line에 속한 token 개수
    #
    # Q token들을 평균낼 때 사용
    # ---------------------------------
    query_token_counts = member.sum(dim=0)

    query_token_counts = (
        query_token_counts.clamp(min=1)
    )

    line_attention_list = []

    # ---------------------------------
    # layer별 attention을
    # token level -> line level로 변환
    # ---------------------------------
    for attention in layer_attentions:

        # 기존 shape:
        # [batch, head, query_token, key_token]
        #
        # batch size = 1이므로 제거
        #
        # 결과:
        # [head, query_token, key_token]
        attention = attention.squeeze(0)

        # [수정 1]
        # key token 수뿐 아니라
        # query token 수도 token_map 길이와 맞는지 확인
        if (
            attention.shape[-1] != num_tokens
            or attention.shape[-2] != num_tokens
        ):
            raise ValueError(
                "attention 토큰 수와 "
                "token_map 길이가 다릅니다. "
                f"query={attention.shape[-2]}, "
                f"key={attention.shape[-1]}, "
                f"token_map={num_tokens}"
            )

        # ---------------------------------
        # 1단계
        # key token -> key line 합산
        #
        # [head, Q_token, K_token]
        #
        #             ↓
        #
        # [head, Q_token, K_line]
        #
        # 같은 K line에 속한 token들의
        # attention mass를 모두 합산
        # ---------------------------------
        attention_to_key_lines = torch.matmul(
            attention,
            member
        )

        # ---------------------------------
        # 2단계
        # query token -> query line 합산
        #
        # [head, Q_token, K_line]
        #
        #             ↓
        #
        # [head, Q_line, K_line]
        # ---------------------------------
        line_attention = torch.matmul(
            member.transpose(0, 1),
            attention_to_key_lines
        )

        # ---------------------------------
        # 3단계
        # query line에 속한 token 수로 나눔
        #
        # K 방향:
        #     SUM
        #
        # Q 방향:
        #     MEAN
        #
        # 최종 수식:
        #
        # A(Lq, Lk)
        # =
        # 1 / |Lq|
        # Σ(q ∈ Lq)
        # Σ(k ∈ Lk)
        # A(q, k)
        # ---------------------------------
        line_attention = (
            line_attention
            / query_token_counts.view(1, -1, 1)
        )

        # GPU/XPU 메모리를 계속 들고 있지 않도록
        # CPU float tensor로 저장
        line_attention_list.append(
            line_attention
            .detach()
            .float()
            .cpu()
        )

    # ---------------------------------
    # 모든 layer 결합
    #
    # 최종 shape:
    #
    # [layer, head, query_line, key_line]
    # ---------------------------------
    line_attentions = torch.stack(
        line_attention_list,
        dim=0
    )

    return line_attentions, line_ids