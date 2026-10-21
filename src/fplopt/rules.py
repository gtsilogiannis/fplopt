"""FPL transfer and chip rules (2026/27). The scoring rules are in xp_model.py, the only
module that uses them.

Transfers, verified 2026-10-19 (notes_06b §2):
  - 1 free transfer (FT) per GW, unused FTs roll over, at most 5 stored.
  - Every transfer beyond the FTs costs HIT_COST points in that GW.
  - Wildcard / Free Hit: unlimited free transfers in that GW; the FTs held before the chip
    are kept for the next GW, but no new one is added (F[g+1] = F[g]). A Free Hit squad
    reverts at the next deadline.
  - Captain and vice-captain come from the XI; if the captain plays 0 minutes the multiplier
    (x2, or x3 with Triple Captain) moves to the vice-captain.
  - Autosubs: a starter with 0 minutes is replaced by the first bench player (in bench order)
    who played and keeps a valid formation; GK only by GK.

Chips, verified 2026-10-20 (premierleague.com "What's happening with FPL chips in 2026/27";
onefpl.com; notes_07 §1):
  - two sets of WC, FH, BB, TC; set 1 must be played by the GW19 deadline, set 2 GW20-38
  - one chip per GW; WC and FH not in GW1; no FH in two consecutive GWs (GW19 -> GW20)
  - WC/FH keep the banked free transfers (F[g+1] = F[g])
The API (bootstrap-static 'chips') lists each chip with start_event / stop_event; the
windows below are used only if that field is missing.
"""
HIT_COST = 4  # points deducted per transfer beyond the free transfers
MAX_FT = 5    # maximum number of stored free transfers

CHIPS = ('WC', 'FH', 'BB', 'TC')
CHIP_API_NAME = {'WC': 'wildcard', 'FH': 'freehit', 'BB': 'bboost', 'TC': '3xc'}   # names in the API
CHIP_RULE_WINDOWS = {'WC': [(2, 19), (20, 38)], 'FH': [(2, 19), (20, 38)],
                     'BB': [(1, 19), (20, 38)], 'TC': [(1, 19), (20, 38)]}
