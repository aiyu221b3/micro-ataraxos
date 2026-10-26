import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from src.env import device, NUM_CELLS, NUM_PIECE_TYPES


class SetupNetwork(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.start_token_id = NUM_PIECE_TYPES
        self.token_emb = nn.Embedding(NUM_PIECE_TYPES + 1, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=0.0, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.policy_head = nn.Linear(d_model, NUM_PIECE_TYPES)
        self.wld_head = nn.Linear(d_model, 3)

    @torch.no_grad()
    def sample_setups(self, batch_size):
        inp = torch.full((batch_size, 1), self.start_token_id, dtype=torch.long, device=device)
        used_mask = torch.zeros((batch_size, NUM_PIECE_TYPES), dtype=torch.bool, device=device)
        placed = []

        for step in range(4):
            x = self.token_emb(inp) + self.pos_emb[:, : step + 1, :]
            causal_mask = nn.Transformer.generate_square_subsequent_mask(step + 1).to(device)
            h = self.transformer(x, mask=causal_mask)[:, -1, :]

            logits = self.policy_head(h).masked_fill(used_mask, -1e9)
            if step == 1:
                # force flag on left half before random l-r symmetry flip
                flag_unplaced = ~used_mask[:, 0]
                if flag_unplaced.any():
                    force_mask = torch.zeros_like(used_mask)
                    force_mask[:, 1:] = True
                    logits = torch.where(flag_unplaced.unsqueeze(1), logits.masked_fill(force_mask, -1e9), logits)

            piece = Categorical(logits=logits).sample()
            placed.append(piece)
            used_mask = used_mask.scatter(1, piece.unsqueeze(1), True)
            inp = torch.cat([inp, piece.unsqueeze(1)], dim=1)

        raw_setups = torch.stack(placed, dim=1)
        flip_mask = torch.rand((batch_size, 1), device=device) < 0.5
        board_setups = torch.where(flip_mask, torch.flip(raw_setups, dims=[1]), raw_setups)
        return raw_setups, board_setups

    def evaluate_setups(self, raw_setups):
        # single pass causal eval for training
        B = raw_setups.size(0)
        start = torch.full((B, 1), self.start_token_id, dtype=torch.long, device=device)
        inp = torch.cat([start, raw_setups[:, :-1]], dim=1)

        x = self.token_emb(inp) + self.pos_emb
        causal_mask = nn.Transformer.generate_square_subsequent_mask(4).to(device)
        h = self.transformer(x, mask=causal_mask)
        raw_logits = self.policy_head(h)

        used = torch.zeros((B, NUM_PIECE_TYPES), dtype=torch.bool, device=device)
        step_log_probs, step_entropies = [], []
        for s in range(4):
            step_logits = raw_logits[:, s, :].masked_fill(used, -1e9)
            if s == 1:
                flag_unplaced = ~used[:, 0]
                force_mask = torch.zeros_like(used)
                force_mask[:, 1:] = True
                step_logits = torch.where(
                    flag_unplaced.unsqueeze(1), step_logits.masked_fill(force_mask, -1e9), step_logits
                )
            dist = Categorical(logits=step_logits)
            chosen = raw_setups[:, s]
            step_log_probs.append(dist.log_prob(chosen))
            step_entropies.append(dist.entropy())
            used = used.scatter(1, chosen.unsqueeze(1), True)

        wld_logits = self.wld_head(h)
        wld_probs = F.softmax(wld_logits, dim=-1)
        expected_val = wld_probs[:, :, 0] - wld_probs[:, :, 1]

        return (
            torch.stack(step_log_probs, dim=1),
            torch.stack(step_entropies, dim=1),
            expected_val,
            wld_logits,
        )


class MoveNetwork(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.cell_proj = nn.Linear(13, d_model)
        self.val_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_emb = nn.Parameter(torch.randn(1, NUM_CELLS + 1, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)
        self.wld_head = nn.Linear(d_model, 3)

    def forward(self, cell_tokens, legal_mask):
        # ip -> board tokens, op -> key-query move logits + wld value
        B = cell_tokens.size(0)
        x_cells = self.cell_proj(cell_tokens)
        v_tok = self.val_token.expand(B, -1, -1)
        x = torch.cat([v_tok, x_cells], dim=1) + self.pos_emb

        h = self.encoder(x)
        h_val = h[:, 0, :]
        h_cells = h[:, 1:, :]

        Q = self.W_Q(h_cells)
        K = self.W_K(h_cells)
        logits_2d = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        logits = logits_2d.view(B, NUM_CELLS * NUM_CELLS).masked_fill(~legal_mask, -1e9)

        wld_logits = self.wld_head(h_val)
        wld_probs = F.softmax(wld_logits, dim=-1)
        expected_val = wld_probs[:, 0] - wld_probs[:, 1]

        return logits, expected_val, wld_logits


class BeliefNetwork(nn.Module):
    """most important belief module. encode board, drop all tokens except hidden opp squares, decode row-major."""

    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.start_token_id = NUM_PIECE_TYPES
        self.cell_proj = nn.Linear(13, d_model)
        self.enc_pos_emb = nn.Parameter(torch.randn(1, NUM_CELLS, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.dec_emb = nn.Embedding(NUM_PIECE_TYPES + 1, d_model)
        self.dec_pos_emb = nn.Parameter(torch.randn(1, 4, d_model) * 0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2, dropout=0.1, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_head = nn.Linear(d_model, NUM_PIECE_TYPES)

    def _filter_hidden_tokens(self, enc_out, hidden_cell_mask):
        B, _, D = enc_out.shape
        filtered_mem = torch.zeros((B, 4, D), dtype=enc_out.dtype, device=device)
        mem_pad_mask = torch.ones((B, 4), dtype=torch.bool, device=device)

        for i in range(B):
            idx = torch.nonzero(hidden_cell_mask[i], as_tuple=False).squeeze(-1)
            m = len(idx)
            if m > 0:
                filtered_mem[i, :m, :] = enc_out[i, idx, :]
                mem_pad_mask[i, :m] = False
        return filtered_mem, mem_pad_mask

    def forward_teacher_forced(self, cell_tokens, hidden_cell_mask, target_pieces):
        B = cell_tokens.size(0)
        enc_in = self.cell_proj(cell_tokens) + self.enc_pos_emb
        enc_out = self.encoder(enc_in)

        filtered_mem, mem_pad_mask = self._filter_hidden_tokens(enc_out, hidden_cell_mask)

        start = torch.full((B, 1), self.start_token_id, dtype=torch.long, device=device)
        dec_tokens = torch.cat([start, target_pieces[:, :-1]], dim=1)
        dec_in = self.dec_emb(dec_tokens) + self.dec_pos_emb

        causal_mask = nn.Transformer.generate_square_subsequent_mask(4).to(device)
        dec_out = self.decoder(
            tgt=dec_in, memory=filtered_mem, tgt_mask=causal_mask, memory_key_padding_mask=mem_pad_mask
        )
        return self.out_head(dec_out)

    @torch.no_grad()
    def sample_worlds(self, single_sim, num_samples, use_uniform_belief=False):
        from src.env import StrategoRolloutSimulator

        assert single_sim.B == 1
        worlds = StrategoRolloutSimulator(num_samples)
        for attr in [
            "owner", "piece", "revealed", "alive_pool",
            "last_src", "last_dst", "repeat_count",
            "turn", "moves_since_battle", "total_moves", "done", "winner",
        ]:
            t = getattr(single_sim, attr)
            setattr(worlds, attr, t.expand(num_samples, *([-1] * (t.dim() - 1))).clone())

        player = single_sim.turn[0].item()
        opp = -player
        opp_pool_idx = 1 if player == 1 else 0

        opp_cells = single_sim.owner[0] == opp
        rev_opp_cells = opp_cells & single_sim.revealed[0]
        unrev_opp_cells = opp_cells & (~single_sim.revealed[0])

        num_hidden = int(unrev_opp_cells.sum().item())
        if num_hidden == 0:
            return worlds

        # conserve piece counts: dead or already revealed pieces cannot be hallucinated
        available = single_sim.alive_pool[0, opp_pool_idx].clone()
        for pt in single_sim.piece[0, rev_opp_cells]:
            available[pt] = False
        unavailable = (~available).unsqueeze(0).expand(num_samples, -1).clone()

        tokens = single_sim.get_infoset_tokens().expand(num_samples, -1, -1)
        hidden_mask = tokens[:, :, 5] > 0.5

        enc_out = self.encoder(self.cell_proj(tokens) + self.enc_pos_emb)
        filtered_mem, mem_pad_mask = self._filter_hidden_tokens(enc_out, hidden_mask)

        dec_tokens = torch.full((num_samples, 1), self.start_token_id, dtype=torch.long, device=device)
        sampled_types = []

        for step in range(num_hidden):
            if use_uniform_belief:
                logits = torch.zeros((num_samples, NUM_PIECE_TYPES), device=device)
            else:
                dec_in = self.dec_emb(dec_tokens) + self.dec_pos_emb[:, : step + 1, :]
                causal_mask = nn.Transformer.generate_square_subsequent_mask(step + 1).to(device)
                dec_out = self.decoder(
                    tgt=dec_in, memory=filtered_mem, tgt_mask=causal_mask, memory_key_padding_mask=mem_pad_mask
                )[:, -1, :]
                logits = self.out_head(dec_out)

            logits = logits.masked_fill(unavailable, -1e9)
            sampled_pt = Categorical(logits=logits).sample()
            sampled_types.append(sampled_pt)
            unavailable.scatter_(1, sampled_pt.unsqueeze(1), True)
            dec_tokens = torch.cat([dec_tokens, sampled_pt.unsqueeze(1)], dim=1)

        sampled_types = torch.stack(sampled_types, dim=1)

        grid = torch.arange(NUM_CELLS, device=device).unsqueeze(0)
        canon_grid = single_sim._canonical_idx(grid, single_sim.turn)[0]
        canon_unrev = torch.nonzero(unrev_opp_cells[canon_grid], as_tuple=False).squeeze(-1)
        global_unrev = canon_grid[canon_unrev]

        worlds.piece[:, global_unrev] = sampled_types
        return worlds


def scalar_return_to_wld_target(scalar_ret):
    p_w = torch.clamp(scalar_ret, min=0.0, max=1.0)
    p_l = torch.clamp(-scalar_ret, min=0.0, max=1.0)
    p_d = 1.0 - p_w - p_l
    return torch.stack([p_w, p_l, p_d], dim=-1)