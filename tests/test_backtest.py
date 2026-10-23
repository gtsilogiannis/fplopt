"""W1: the real-points scoring of the rolling-horizon backtest (scripts/backtest_season.py),
checked on hand-made gameweeks against the game's rules (autosubs, formation, vice-captain,
Triple Captain, Bench Boost, hits)."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from backtest_season import score_gw   # noqa: E402

# squad: GK 1-2, DEF 3-7, MID 8-12, FWD 13-15
POS = {i: 1 if i <= 2 else 2 if i <= 7 else 3 if i <= 12 else 4 for i in range(1, 16)}
XI = [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15]          # 3-4-3
BENCH = [2, 12, 6, 7]                                  # GK slot, then MID 12, DEF 6, DEF 7


def pts(zero=()):
    """Player i scores i points if he plays, 0 if not."""
    return {i: 0 if i in zero else i for i in range(1, 16)}


def gw(**kw):
    base = dict(squad=list(range(1, 16)), xi=XI, bench=BENCH, captain_id=15, vice_id=14, chip=None, hits=0)
    base.update(kw)
    return SimpleNamespace(**base)


def mins(zero=()):
    return {i: 0 if i in zero else 90 for i in range(1, 16)}


def test_everyone_plays():
    total, info = score_gw(gw(), POS, mins(), pts())
    assert total == sum(XI) + 15                       # captain counted twice
    assert info['captain_used'] == 15


def test_captain_out_vice_doubles():
    total, info = score_gw(gw(), POS, mins(zero=[15]), pts(zero=[15]))
    # 15 is replaced by the first bench outfielder who keeps the formation: MID 12 (3-5-2)
    assert info['xi'] == [12 if i == 15 else i for i in XI]
    assert total == sum(XI) - 15 + 12 + 14
    assert info['captain_used'] == 14


def test_formation_skips_a_sub():
    # DEF 3 out: MID 12 would leave 2 DEF (invalid), so DEF 6 comes on
    total, info = score_gw(gw(), POS, mins(zero=[3]), pts(zero=[3]))
    assert 6 in info['xi'] and 12 not in info['xi'] and 3 not in info['xi']
    assert total == sum(XI) - 3 + 6 + 15


def test_goalkeeper_only_by_goalkeeper():
    total, info = score_gw(gw(), POS, mins(zero=[1]), pts(zero=[1]))
    assert info['xi'][0] == 2
    total2, info2 = score_gw(gw(), POS, mins(zero=[1, 2]), pts(zero=[1, 2]))   # both GKs out: nobody comes on
    assert 1 in info2['xi'] and total2 == sum(XI) - 1 + 15


def test_bench_player_who_did_not_play_is_skipped():
    total, info = score_gw(gw(), POS, mins(zero=[8, 12]), pts(zero=[8, 12]))     # MID 8 out, first sub MID 12 out too
    assert 6 in info['xi'] and 12 not in info['xi']                 # DEF 6 next (4-3-3 is valid)
    assert total == sum(XI) - 8 + 6 + 15


def test_triple_captain_bench_boost_hits():
    assert score_gw(gw(chip='TC'), POS, mins(), pts())[0] == sum(XI) + 2 * 15
    assert score_gw(gw(chip='BB'), POS, mins(zero=[3]), pts(zero=[3]))[0] == sum(range(1, 16)) - 3 + 15
    assert score_gw(gw(hits=2), POS, mins(), pts())[0] == sum(XI) + 15 - 8


def test_captain_and_vice_out():
    total, info = score_gw(gw(), POS, mins(zero=[14, 15]), pts(zero=[14, 15]))
    assert info['captain_used'] is None                 # no multiplier for anyone
    assert total == sum(XI) - 14 - 15 + 12 + 6   # MID 12 for 14, then DEF 6 for 15 (4-5-1)


def test_bench_slots_used():
    _, info = score_gw(gw(), POS, mins(zero=[1, 3, 8]), pts(zero=[1, 3, 8]))
    # GK 2 for GK 1; MID 12 for MID 8 (for DEF 3 it would leave 2 DEF); DEF 6 for DEF 3
    assert info["subs"] == [0, 1, 2]
