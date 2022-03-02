import string
import numpy as np

from election import Election
import pytest


def test_run_election_from_popularity():
    seed = 123456
    np.random.seed(seed)

    nb_candidate = 5
    candidates = list(string.ascii_uppercase)[:nb_candidate]
    proba_ranked = np.random.rand(nb_candidate)
    popularity = np.random.rand(nb_candidate)
    nb_voter = int(1e3)
    elect = Election(
        nb_candidate=nb_candidate,
        nb_voter=nb_voter,
        candidates=candidates,
        proba_ranked=proba_ranked,
        popularity=popularity,
    )
    elect.overview_candidates()
    elect.run_election_from_popularity(seed=seed)
    elect.build_table_duels()
    result = np.array(
        [
            [0, 40, 4, 80, 15],
            [62, 0, 57, 755, 158],
            [23, 190, 0, 210, 77],
            [17, 122, 13, 0, 38],
            [24, 209, 26, 296, 0],
        ]
    )
    assert np.array_equal(elect.df_duels, result)
    assert elect.check_table_duels() == 2416
    elect.build_table_payoff()
    result = np.array(
        [
            [0, -1, -1, 1, -1],
            [1, 0, -1, 1, -1],
            [1, 1, 0, 1, 1],
            [-1, -1, -1, 0, -1],
            [1, 1, -1, 1, 0],
        ]
    )
    assert np.array_equal(elect.df_payoffs, result)
    elect.get_best_lottery()
    assert elect.best_lottery == {"A": 0.0, "B": 0.0, "C": 1.0, "D": 0.0, "E": 0.0}


def test_ballot1():
    ballot = np.array([[2, 0, 0, 0, 0], [3, 2, 4, 0, 0], [1, 2, 3, 0, 0]])
    elect = Election.run_election_from_ballot(ballot, nb_candidate=5)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(0, 12),
        "C": round(0, 12),
        "D": round(1 / 3, 12),
        "E": round(1 / 3, 12),
    }


def test_ballot2():
    ballot = np.array([[1], [2], [3]])
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(1 / 3, 12),
        "C": round(1 / 3, 12),
    }


def test_ballot3():
    ballot = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(1 / 3, 12),
        "C": round(1 / 3, 12),
    }


def test_ballot4():
    ballot = np.array(
        [[1, 0, 0], [2, 0, 0], [3, 0, 0]]
    )  # pb should be equal to previous line
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(1 / 3, 12),
        "C": round(1 / 3, 12),
    }


def test_ballot5():
    ballot = np.array(
        [[1, [2, 3]], [2, [1, 3]], [3, [1, 2]]], dtype=object
    )  # should be 1/3 * 3
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(1 / 3, 12),
        "C": round(1 / 3, 12),
    }


def test_ballot6():
    ballot = [[2, 0, 0, 0, 0], [3, 2, 4, 0, 0], [[1, 2], 3, 0, 0, 0]]
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 2, 12),
        "B": round(1 / 2, 12),
        "C": round(0, 12),
        "D": round(0, 12),
    }


def test_ballot7():
    ballot = ((2,), (1, (2, 3, 4, 5)), (3, 2, 4), ((1, 2), 3))
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1, 12),
        "B": round(0, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(0, 12),
    }


def test_ballot8():
    ballot = ((2,), (1, (2, 3, 4, 5)), (3, 2, 4), ((1, 2), 3), (5, (0, 0), 0))
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1, 12),
        "B": round(0, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(0, 12),
    }


def test_ballot9():
    ballot = ((1, 2, 3, 4, 5), (5, (4, 3), 2, 1), ((4, 5), 1, 2, 3))
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(0, 12),
        "C": round(0, 12),
        "D": round(1 / 2, 12),
        "E": round(1 / 2, 12),
    }


def test_ballot10():
    ballot = ((1, 2, 3, 4, 5), (5, 4, 3, (2, 1)), ((4, 5), 1, 2, 5))
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(0, 12),
        "C": round(1 / 2, 12),
        "D": round(1 / 2, 12),
        "E": round(0, 12),
    }


def test_ballot11():
    ballot = ((1, (2, 3, 4, 5)), (2, (1, 3, 4, 5)))
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 2, 12),
        "B": round(1 / 2, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(0, 12),
    }


def test_ballot12():
    ballot = ((1, 2, 3), (2, 3))
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1, 12),
        "B": round(0, 12),
        "C": round(0, 12),
    }


def test_ballot13():
    ballot = ((1, 2), (2, 1))
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 2, 12),
        "B": round(1 / 2, 12),
    }


def test_ballot14():
    ballot = ((1, 5, (2, 3, 4)), (2, (1, 3, 4, 5)))
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 2, 12),
        "B": round(1 / 2, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(0, 12),
    }


def test_ballot15():
    ballot = ((1, (2, 3, 4, 5)), (2, (1, 3, 4, 5)), (5, 4, 3, 2, 1))
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(1 / 2, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(1 / 2, 12),
    }


def test_ballot16():
    ballot = (
        (1, (2, 3, 4, 5)),
        (2, (1, 3, 4, 5)),
        (3, (4, 5, 2, 1)),
        (4, (5, 3, 2, 1)),
        (5, (4, 3, 2, 1)),
    )
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 5, 12),
        "B": round(1 / 5, 12),
        "C": round(1 / 5, 12),
        "D": round(1 / 5, 12),
        "E": round(1 / 5, 12),
    }


def test_ballot17():
    ballot = [(4, 1, 3), (3, 2, 4), (1, 3, 2), (2, 4, 1)]
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 4, 12),
        "B": round(1 / 4, 12),
        "C": round(1 / 4, 12),
        "D": round(1 / 4, 12),
    }


def test_ballot18():
    ballot = ((1, (2, 3, 4, 5)), (2, (1, 3, 4, 5)), (5, 4, 3, 2, 1), (5, 4, 3, 2, 1))
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(0, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(1, 12),
    }


def test_ballot19():
    ballot = ((1, 2, 3), (2, 3, 1), (3, 1, 2))
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(1 / 3, 12),
        "C": round(1 / 3, 12),
    }


def test_ballot20():
    ballot = ()
    with pytest.raises(AttributeError):
        elect = Election.run_election_from_ballot(ballot)


def test_ballot21():
    ballot = (("A", "B"), ("C", "D"))
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 2, 12),
        "B": round(0, 12),
        "C": round(1 / 2, 12),
        "D": round(0, 12),
    }


def test_ballot22():
    ballot = (("A", "B"), (3, 4))
    with pytest.raises(AttributeError):
        elect = Election.run_election_from_ballot(ballot)
    ballot = (("A", "B"), ("C", 4))
    with pytest.raises(AttributeError):
        elect = Election.run_election_from_ballot(ballot)


def test_ballot23():
    ballot = (
        ("Luc", "Han", "Chewie", "Yoda", "Ben"),
        ("Ben", ("Yoda", "Chewie"), "Han", "Luc"),
        (("Yoda", "Ben"), "Luc", "Han", "Chewie"),
    )
    elect = Election.run_election_from_ballot(ballot)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "Luc": round(0, 12),
        "Han": round(0, 12),
        "Chewie": round(0, 12),
        "Yoda": round(1 / 2, 12),
        "Ben": round(1 / 2, 12),
    }
