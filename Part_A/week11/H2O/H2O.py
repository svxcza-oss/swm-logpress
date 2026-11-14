import torch

class H2OKVCache_LayerWise:
    def __init__(
        self,
        hh_size=4,
        recent_size=4,
        k_seq_dim=2,
        v_seq_dim=2,
    ):
        # print(f"H2OKVCache-LayerWise: {hh_size}, {recent_size}")
        self.hh_size = hh_size
        self.recent_size = recent_size
        self.cache_size = hh_size + recent_size
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.hh_score = None
        self.kept_indices = None
        
    def _update_hh_score(self, attn_score_cache, num_kv_heads):

        batch_size, num_attn_heads, q_len, k_len = (
            attn_score_cache.shape
        )

        # 현재 실험은 batch_size = 1 기준
        assert batch_size == 1

        # Query Head를 KV Head 그룹으로 나누기
        assert num_attn_heads % num_kv_heads == 0

        group_size = num_attn_heads // num_kv_heads

        # [B, Query Heads, Q, K]
        # → [B, KV Heads, Group, Q, K]
        grouped_attn = attn_score_cache.reshape(
            batch_size,
            num_kv_heads,
            group_size,
            q_len,
            k_len
        )

        # Group과 Query 차원을 합산
        # → [KV Heads, K]
        new_scores = grouped_attn.sum(dim=(2, 3)).squeeze(0)

        if self.hh_score is None:
            self.hh_score = new_scores

        else:
            # 이전 Cache 점수와 새 Attention 점수 누적
            previous_length = k_len - q_len

            assert self.hh_score.shape[-1] == previous_length

            new_scores[:, :previous_length] += self.hh_score

            self.hh_score = new_scores



    def __call__(self, past_key_values, attn_score_cache):

        if past_key_values is None:
            return None

        num_kv_heads = past_key_values[0].shape[1]
        
        self._update_hh_score(
            attn_score_cache,
            num_kv_heads
            )
        
        seq_len = past_key_values[0].size(self.k_seq_dim)

        if seq_len <= self.cache_size:
            return past_key_values
        
        bsz, num_heads, _, head_dim = past_key_values[0].shape
        
        select_hh_scores = self.hh_score[:, :seq_len - self.recent_size]
        _, keep_topk = torch.topk(select_hh_scores, self.hh_size, dim=-1)
        keep_topk = keep_topk.sort().values
        
        keep_recent = torch.arange(seq_len - self.recent_size, seq_len, device=keep_topk.device).repeat(keep_topk.shape[0], 1)
        keep_idx = torch.cat([keep_topk, keep_recent], dim=-1)
        
        # 현재 구현은 [B, KVH, S, D], batch_size=1 기준
        assert bsz == 1
        assert self.k_seq_dim == 2 and self.v_seq_dim == 2

        # 선택된 토큰 인덱스 기록
        # shape: [KV Heads, Cache Size]
        self.kept_indices = keep_idx.detach().cpu()

        # gather를 위한 shape 확장
        # [KVH, Cache Size] → [B, KVH, Cache Size, Head Dim]
        gather_indices = keep_idx.unsqueeze(0).unsqueeze(-1).expand(
            bsz,
            num_heads,
            self.cache_size,
            head_dim
        )

        # 실제 KV Cahce 압축
        k_hh_recent = past_key_values[0].gather(
            2, gather_indices
        ).contiguous()

        v_hh_recent = past_key_values[1].gather(
            2, gather_indices
        ).contiguous()

        # 남은 토큰의 HH Score도 동일한 인덱스로 압축
        self.hh_score = self.hh_score.gather(1, keep_idx)

        return (k_hh_recent, v_hh_recent)

if __name__ == "__main__":

    keys = torch.rand(1, 2, 14, 128)
    values = torch.rand(1, 2, 14, 128)

    attentions = torch.rand(1, 16, 14, 14)

    h2o = H2OKVCache_LayerWise(
        hh_size=4,
        recent_size=4
    )

    compressed_keys, compressed_values = h2o(
        (keys, values),
        attentions
    )

    print("Compressed Keys:", compressed_keys.shape)
    print("Compressed Values:", compressed_values.shape)
    print("HH Scores:", h2o.hh_score.shape)
    print("Kept Indices:", h2o.kept_indices)
    print("Kept Indices Shape:", h2o.kept_indices.shape)

    assert h2o.kept_indices.shape == (2, 8)

    expected_recent = torch.arange(10, 14).repeat(2, 1)

    assert torch.equal(
        h2o.kept_indices[:, -4:],
        expected_recent
    )

    print("H2O HH/Recent Index Tracking 검증 성공")