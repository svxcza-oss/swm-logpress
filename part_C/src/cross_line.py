import torch

def convert_to_line_attention(layer_attentions,token_map):

    line_ids = []

    for token_info in token_map:
        line_id = token_info["line_id"]

        if line_id is None:
            continue

        if line_id not in line_ids:
            line_ids.append(line_id)

    if not line_ids:
        raise ValueError(
            "token_map에 유효한 line_id가 없습니다."
        )
    line_to_index = {
        line_id: line_idx
        for line_idx, line_id in enumerate(line_ids)
    }
    num_tokens = len(token_map)
    num_lines = len(line_ids)

    device = layer_attentions[0].device
    dtype = layer_attentions[0].dtype

    member  = torch.zeros(
        (num_tokens,num_lines),
        dtype = dtype,
        device = device
    )

    for token_info in token_map:
        token_idx = token_info["token_idx"]
        line_id = token_info["line_id"]

        if line_id is None:
            continue

        line_idx = line_to_index[line_id]
        member[token_idx,line_idx] = 1 

    query_token_counts = member.sum(dim=0)

    query_token_counts = query_token_counts.clamp(min =1) #

    line_attention_list = []

    for attention in layer_attentions:
        attention = attention.squeeze(0)

        if attention.shape[-1] != num_tokens:
            raise ValueError(
                "attention 토큰 수와 token_map 길이가 다릅니다. "
                f"attention={attention.shape[-1]}, "
                f"token_map={num_tokens}"
            )
        attention_to_key_lines = torch.matmul(
            attention,
            member
        )

       
        line_attention = torch.matmul(
            member.transpose(0, 1),
            attention_to_key_lines
        )

        
        line_attention = (
            line_attention
            / query_token_counts.view(1, -1, 1)
        )

        line_attention_list.append(
            line_attention.detach().float().cpu()
        )

    line_attentions = torch.stack(
        line_attention_list,
        dim=0
    )

    return line_attentions, line_ids
