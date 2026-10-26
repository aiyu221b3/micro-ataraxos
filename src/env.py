import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0: flag (immovable), 1: spy (rank 1, slays marshal on attack), 2: miner (rank 2), 3: marshal (rank 3)
NUM_CELLS = 16
BOARD_SIZE = 4
NUM_PIECE_TYPES = 4
MAX_MOVES_WITHOUT_BATTLE = 24


class StrategoRolloutSimulator:
    def __init__(self, batch_size: int):
        self.B = batch_size
        self.reset()

    def reset(self, setups_p0=None, setups_p1=None):
        self.owner = torch.zeros((self.B, NUM_CELLS), dtype=torch.long, device=device)
        self.piece = torch.zeros((self.B, NUM_CELLS), dtype=torch.long, device=device)
        self.revealed = torch.zeros((self.B, NUM_CELLS), dtype=torch.bool, device=device)
        self.alive_pool = torch.ones((self.B, 2, NUM_PIECE_TYPES), dtype=torch.bool, device=device)

        # track last 2 squares per player to enforce the two-square anti-shuffling rule
        self.last_src = torch.full((self.B, 2), -1, dtype=torch.long, device=device)
        self.last_dst = torch.full((self.B, 2), -1, dtype=torch.long, device=device)
        self.repeat_count = torch.zeros((self.B, 2), dtype=torch.long, device=device)

        self.turn = torch.ones(self.B, dtype=torch.long, device=device)
        self.moves_since_battle = torch.zeros(self.B, dtype=torch.long, device=device)
        self.total_moves = torch.zeros(self.B, dtype=torch.long, device=device)
        self.done = torch.zeros(self.B, dtype=torch.bool, device=device)
        self.winner = torch.zeros(self.B, dtype=torch.float, device=device)

        if setups_p0 is not None and setups_p1 is not None:
            self.apply_setups(setups_p0, setups_p1)
        return self

    def apply_setups(self, setups_p0, setups_p1):
        self.owner[:, 0:4] = 1
        self.piece[:, 0:4] = setups_p0
        p1_cells = torch.tensor([15, 14, 13, 12], device=device)
        self.owner[:, p1_cells] = -1
        self.piece[:, p1_cells] = setups_p1

    def clone(self):
        sim = StrategoRolloutSimulator(self.B)
        for attr in [
            "owner", "piece", "revealed", "alive_pool",
            "last_src", "last_dst", "repeat_count",
            "turn", "moves_since_battle", "total_moves", "done", "winner",
        ]:
            setattr(sim, attr, getattr(self, attr).clone())
        return sim

    def _canonical_idx(self, cells, player):
        # flip ze board 180 deg for p1 so net always plays from bottom
        return torch.where(player.unsqueeze(1) == 1, cells, (NUM_CELLS - 1) - cells)

    def get_infoset_tokens(self):
        # ip -> board state, op -> (B, 16, 13) canonical tokens. zero leakage of hidden opp pieces
        B = self.B
        tokens = torch.zeros((B, NUM_CELLS, 13), dtype=torch.float, device=device)
        grid = torch.arange(NUM_CELLS, device=device).unsqueeze(0).expand(B, -1)
        canon_grid = self._canonical_idx(grid, self.turn)

        c_owner = torch.gather(self.owner, 1, canon_grid) * self.turn.unsqueeze(1)
        c_piece = torch.gather(self.piece, 1, canon_grid)
        c_rev = torch.gather(self.revealed, 1, canon_grid)

        tokens[:, :, 0] = (c_owner == 0).float()
        for pt in range(4):
            tokens[:, :, 1 + pt] = ((c_owner == 1) & (c_piece == pt)).float()
        tokens[:, :, 5] = ((c_owner == -1) & (~c_rev)).float()
        for pt in range(1, 4):
            tokens[:, :, 5 + pt] = ((c_owner == -1) & c_rev & (c_piece == pt)).float()

        opp_idx = torch.where(self.turn == 1, 1, 0)
        opp_alive = self.alive_pool[torch.arange(B, device=device), opp_idx].float()
        tokens[:, :, 9:13] = opp_alive.unsqueeze(1).expand(B, NUM_CELLS, 4)
        return tokens

    def get_legal_action_mask(self):
        B = self.B
        grid = torch.arange(NUM_CELLS, device=device).unsqueeze(0).expand(B, -1)
        canon_grid = self._canonical_idx(grid, self.turn)

        c_owner = torch.gather(self.owner, 1, canon_grid) * self.turn.unsqueeze(1)
        c_piece = torch.gather(self.piece, 1, canon_grid)

        can_move_src = (c_owner == 1) & (c_piece > 0)
        valid_dst = c_owner != 1

        r = torch.arange(NUM_CELLS, device=device) // BOARD_SIZE
        c = torch.arange(NUM_CELLS, device=device) % BOARD_SIZE
        adj = (torch.abs(r.unsqueeze(1) - r.unsqueeze(0)) + torch.abs(c.unsqueeze(1) - c.unsqueeze(0))) == 1

        mask = can_move_src.unsqueeze(2) & valid_dst.unsqueeze(1) & adj.unsqueeze(0)
        flat_mask = mask.view(B, NUM_CELLS * NUM_CELLS)

        # two-square rule: prohibit back-and-forth oscillation on >2 consecutive turns
        p_idx = torch.where(self.turn == 1, 0, 1)
        b_arange = torch.arange(B, device=device)
        rep_cnt = self.repeat_count[b_arange, p_idx]
        l_src = self.last_src[b_arange, p_idx]
        l_dst = self.last_dst[b_arange, p_idx]

        ban = (rep_cnt >= 2) & (l_src >= 0) & (l_dst >= 0)
        if ban.any():
            ban_b = b_arange[ban]
            # reversing last move means moving from l_dst back to l_src
            rev_act = l_dst[ban] * NUM_CELLS + l_src[ban]
            temp_mask = flat_mask.clone()
            temp_mask[ban_b, rev_act] = False
            # only apply ban if player still has at least one other legal action
            has_other = temp_mask.any(dim=1)
            flat_mask = torch.where(has_other.unsqueeze(1), temp_mask, flat_mask)

        return flat_mask

    def get_magnet_policy(self, legal_mask):
        # myopic uniform prior: pick a movable piece uniformly, then pick a legal move uniformly
        B = legal_mask.size(0)
        mask_2d = legal_mask.view(B, NUM_CELLS, NUM_CELLS).float()
        src_deg = mask_2d.sum(dim=2, keepdim=True)
        movable = (src_deg > 0).float()
        num_movable = movable.sum(dim=1, keepdim=True).clamp(min=1.0)

        magnet_2d = torch.where(
            src_deg > 0,
            (1.0 / num_movable) * (mask_2d / src_deg.clamp(min=1.0)),
            torch.zeros_like(mask_2d),
        )
        return magnet_2d.view(B, NUM_CELLS * NUM_CELLS)

    def step(self, canon_actions):
        B = self.B
        active = ~self.done
        if not active.any():
            return

        c_src = canon_actions // NUM_CELLS
        c_dst = canon_actions % NUM_CELLS

        # update two-square rule state machine in canonical coordinates
        p_idx = torch.where(self.turn == 1, 0, 1)
        b_all = torch.arange(B, device=device)
        is_rev = (c_src == self.last_dst[b_all, p_idx]) & (c_dst == self.last_src[b_all, p_idx])
        new_rep = torch.where(is_rev, self.repeat_count[b_all, p_idx] + 1, torch.ones_like(p_idx))
        self.repeat_count[active, p_idx[active]] = new_rep[active]
        self.last_src[active, p_idx[active]] = c_src[active]
        self.last_dst[active, p_idx[active]] = c_dst[active]

        src = torch.where(self.turn == 1, c_src, (NUM_CELLS - 1) - c_src)
        dst = torch.where(self.turn == 1, c_dst, (NUM_CELLS - 1) - c_dst)

        b_idx = b_all[active]
        s_idx = src[active]
        d_idx = dst[active]
        p_turn = self.turn[active]

        att_piece = self.piece[b_idx, s_idx]
        att_rev = self.revealed[b_idx, s_idx]  # save revealed state before wiping src square
        def_owner = self.owner[b_idx, d_idx]
        def_piece = self.piece[b_idx, d_idx]

        self.owner[b_idx, s_idx] = 0
        self.piece[b_idx, s_idx] = 0
        self.revealed[b_idx, s_idx] = False

        empty_mask = def_owner == 0
        if empty_mask.any():
            eb, ed = b_idx[empty_mask], d_idx[empty_mask]
            self.owner[eb, ed] = p_turn[empty_mask]
            self.piece[eb, ed] = att_piece[empty_mask]
            self.revealed[eb, ed] = att_rev[empty_mask]
            self.moves_since_battle[eb] += 1

        battle_mask = def_owner != 0
        if battle_mask.any():
            bb, bd = b_idx[battle_mask], d_idx[battle_mask]
            ap, dp = att_piece[battle_mask], def_piece[battle_mask]
            pt = p_turn[battle_mask]
            self.moves_since_battle[bb] = 0

            att_player_idx = torch.where(pt == 1, 0, 1)
            def_player_idx = 1 - att_player_idx

            flag_cap = dp == 0
            if flag_cap.any():
                fb = bb[flag_cap]
                self.done[fb] = True
                self.winner[fb] = pt[flag_cap].float()

            att_wins = flag_cap | ((ap == 1) & (dp == 3)) | (ap > dp)
            tie = ap == dp
            def_wins = (~att_wins) & (~tie)

            if att_wins.any():
                aw_b, aw_d = bb[att_wins], bd[att_wins]
                self.owner[aw_b, aw_d] = pt[att_wins]
                self.piece[aw_b, aw_d] = ap[att_wins]
                self.revealed[aw_b, aw_d] = True
                self.alive_pool[aw_b, def_player_idx[att_wins], dp[att_wins]] = False

            if tie.any():
                t_b, t_d = bb[tie], bd[tie]
                self.owner[t_b, t_d] = 0
                self.piece[t_b, t_d] = 0
                self.revealed[t_b, t_d] = False
                self.alive_pool[t_b, att_player_idx[tie], ap[tie]] = False
                self.alive_pool[t_b, def_player_idx[tie], dp[tie]] = False

            if def_wins.any():
                dw_b, dw_d = bb[def_wins], bd[def_wins]
                self.revealed[dw_b, dw_d] = True
                self.alive_pool[dw_b, att_player_idx[def_wins], ap[def_wins]] = False

        self.turn[active] = -self.turn[active]
        self.total_moves[active] += 1

        next_legal = self.get_legal_action_mask().any(dim=1)
        no_moves = (~self.done) & (~next_legal)
        if no_moves.any():
            self.turn[no_moves] = -self.turn[no_moves]
            other_legal = self.get_legal_action_mask().any(dim=1)
            self.turn[no_moves] = -self.turn[no_moves]

            both_stuck = no_moves & (~other_legal)
            only_curr_stuck = no_moves & other_legal

            if both_stuck.any():
                self.done[both_stuck] = True
                self.winner[both_stuck] = 0.0
            if only_curr_stuck.any():
                self.done[only_curr_stuck] = True
                self.winner[only_curr_stuck] = -self.turn[only_curr_stuck].float()

        stalled = (~self.done) & (self.moves_since_battle >= MAX_MOVES_WITHOUT_BATTLE)
        if stalled.any():
            self.done[stalled] = True
            self.winner[stalled] = 0.0