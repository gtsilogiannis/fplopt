"""Squad, bank, selling prices and free transfers rebuilt from the public API
(chats 8b and 6b; reference: harness_offline.py and harness_6b.py ft in baseline_09a.txt)."""
import pytest

from fpl_fake import CUR, sell_rule
from fplopt.squad import free_transfers_from_history, load_current_squad, selling_price


@pytest.mark.parametrize("purchase, now, expected", [
    (70, 73, 71), (70, 74, 72), (70, 70, 70), (70, 68, 68), (70, 71, 70), (55, 56, 55), (55, 57, 56)])
def test_selling_price(purchase, now, expected):
    # a rise is shared 50/50 and rounded down to 0.1m; a fall is passed on in full
    assert selling_price(purchase, now) == expected


def load(season, client, messages):
    """Runs load_current_squad against the fake API; returns its result, its messages and the truth."""
    squad, bank, sell = load_current_squad(client, 1, CUR, {'elements': season.elements})
    t_sq, t_pur, t_bank = client.api.manager['truth']
    truth_sell = {p: sell_rule(t_pur[p], season.price(p, CUR)) / 10 for p in t_sq}
    return squad, bank, sell, messages(), (t_sq, t_pur, t_bank, truth_sell)


FH_MSG = "[*] Free Hit played in GW7: squad and bank taken from GW6."
UNSURE_MSG = ("[*] Selling value of the squad is between £96.2m and £96.3m (purchase price not exactly "
              "known: P57). Using the lower value; --override-budget sets the exact one from the app.")


@pytest.mark.parametrize("scenario, manager_kw, expected_messages, bank, sum_sell, prices_match, n_requests", [
    ('fh', {}, [FH_MSG], 2.9, 96.2, True, 4),
    ('nofh', {'fh_last': False}, [], 4.9, 94.4, True, 4),
    ('midseason', {'first_gw': 2}, [FH_MSG, UNSURE_MSG], 2.9, 96.2, True, 20),
    ('midmix', {'first_gw': 2, 'mix': True}, [FH_MSG, UNSURE_MSG], 2.9, 96.2, False, 20),
    ('corrupt', {'corrupt': True},
     [FH_MSG, "[WARNING] Rebuilt bank £3.2m != API bank £2.9m: selling prices may be off. "
              "Check them in the app; use --override-budget if needed."], 2.9, 96.2, True, 4),
])
def test_load_current_squad(season, fake_client, messages, scenario, manager_kw, expected_messages, bank,
                            sum_sell, prices_match, n_requests):
    client = fake_client(**manager_kw)
    squad, got_bank, sell, got_messages, (t_sq, t_pur, t_bank, truth_sell) = load(season, client, messages)
    assert got_messages == expected_messages
    assert sorted(squad) == sorted(t_sq)                    # Free Hit squad reverted
    assert got_bank == bank == t_bank / 10
    assert (sell == truth_sell) is prices_match             # midmix: the lower bound is used
    assert round(sum(sell.values()), 1) == sum_sell
    assert all(sell[p] <= truth_sell[p] + 1e-9 for p in t_sq)   # never above the real selling price
    assert len(client.api.calls) == n_requests


def test_midmix_purchase_prices(season, fake_client, messages):
    # two players were bought between two price changes; the true sum is 96.3, the code uses 96.2
    client = fake_client(first_gw=2, mix=True)
    assert client.api.manager['mixed'] == ([39, 57], [(70, 69, 69), (45, 46, 46)])
    *_, (_, _, _, truth_sell) = load(season, client, messages)
    assert round(sum(truth_sell.values()), 1) == 96.3


def test_selling_price_examples(season, fake_client, messages):
    client = fake_client()
    squad, bank, sell, _, (t_sq, t_pur, _, _) = load(season, client, messages)
    gain = [(p, t_pur[p], season.price(p, CUR), sell[p]) for p in t_sq if season.price(p, CUR) != t_pur[p]][:4]
    assert gain == [(39, 70, 66, 6.6), (57, 45, 52, 4.8), (111, 70, 66, 6.6), (297, 84, 87, 8.5)]


def history(first, rows, chips=()):
    """entry/{id}/history/ with {gw: (transfers, cost)} up to GW7."""
    return {'current': [{'event': g, 'event_transfers': rows.get(g, (0, 0))[0],
                         'event_transfers_cost': rows.get(g, (0, 0))[1]} for g in range(first, 8)],
            'chips': [{'name': n, 'event': g} for n, g in chips]}


@pytest.mark.parametrize("hist, expected", [
    # test manager: F2=1 F3=2 F4=2 F5=3 F6=3 (WC) F7=3 F8=3 (FH)
    (history(1, {3: (1, 0), 5: (2, 0), 6: (1, 0), 7: (3, 0)}, [('wildcard', 5), ('freehit', 7)]), (3, [])),
    # user's team (first GW3), three allocations allowed by its bank history -> 2 in each
    (history(3, {4: (1, 0), 5: (1, 0), 7: (1, 0)}), (2, [])),
    (history(3, {5: (2, 0), 7: (1, 0)}), (2, [])),
    (history(3, {5: (2, 0), 6: (1, 0)}), (2, [])),
    (history(1, {}), (5, [])),                                   # cap of 5 stored FTs
    (history(1, {3: (2, 0), 4: (2, 4)}), (4, [])),               # a hit taken in GW4
    (history(1, {3: (3, 0)}), (5, ['GW3: 3 transfers with 2 FT should cost 4, API says 0'])),
    ({'current': [], 'chips': []}, (None, [])),                   # first deadline not passed
])
def test_free_transfers_from_history(hist, expected):
    assert free_transfers_from_history(hist, 8) == expected
