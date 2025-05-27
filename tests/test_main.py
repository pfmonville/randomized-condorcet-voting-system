import json
import re
from unittest.mock import patch

import pytest
import numpy as np

from src.condorcet.types import EliminatedInfo, WinnerInfo
from src.condorcet.visualization import GraphVisualizer
from src.condorcet.election import Election


def test_ballots1():
    """Tests lottery output for a single mixed ballots with and without completing votes."""
    ballots = [[[2, 0, 0, 0, 0], [3, 2, 4, 0, 0], [1, 2, 3, 0, 0]]]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(0, 12),
        "C": round(0, 12),
        "D": round(1 / 3, 12),
        "E": round(1 / 3, 12),
    }

    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(1, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(0, 12),
    }


def test_ballots2():
    """Ensures equal probability lottery when all voters rank each candidate distinctly."""
    ballots = [[[1], [2], [3]]]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=3, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(1 / 3, 12),
        "C": round(1 / 3, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=3)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(1 / 3, 12),
        "C": round(1 / 3, 12),
    }


def test_ballots3():
    """Verifies uniform lottery for cyclic three-candidate preferences."""
    ballots = [[[1, 2, 3], [2, 3, 1], [3, 1, 2]]]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=3, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(1 / 3, 12),
        "C": round(1 / 3, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=3)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(1 / 3, 12),
        "C": round(1 / 3, 12),
    }


def test_ballots4():
    """Checks handling of zero placeholders in ballots, without affecting probabilities."""
    ballots = [[[1, 0, 0], [2, 0, 0], [3, 0, 0]]]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=3, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(1 / 3, 12),
        "C": round(1 / 3, 12),
    }
    ballots = [[[1, 0, 0], [2, 0, 0], [3, 0, 0]]]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=3)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(1 / 3, 12),
        "C": round(1 / 3, 12),
    }


def test_ballots5():
    """Validates mixed nested preferences yield correct uniform lottery."""
    ballots = [[[1, [2, 3]], [2, [1, 3]], [3, [1, 2]]]]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=3, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(1 / 3, 12),
        "C": round(1 / 3, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=3)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(1 / 3, 12),
        "C": round(1 / 3, 12),
    }


def test_ballots6():
    """Tests two-way tie resolution in four-candidate ballots for both vote modes."""
    ballots = [[[2, 0, 0, 0, 0], [3, 2, 4, 0, 0], [[1, 2], 3, 0, 0, 0]]]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=4, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 2, 12),
        "B": round(1 / 2, 12),
        "C": round(0, 12),
        "D": round(0, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=4)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(1, 12),
        "C": round(0, 12),
        "D": round(0, 12),
    }


def test_ballots7():
    """Confirms correct winner when a single top-ranked candidate dominates pairwise duels."""
    ballots = [((2,), (1, (2, 3, 4, 5)), (3, 2, 4), ((1, 2), 3))]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1, 12),
        "B": round(0, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(0, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(1, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(0, 12),
    }


def test_ballots8():
    """Handles ballots with zero ranks mixed in tuple structures correctly."""
    ballots = [((2,), (1, (2, 3, 4, 5)), (3, 2, 4), ((1, 2), 3), (5, (0, 0), 0))]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1, 12),
        "B": round(0, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(0, 12),
    }
    elect = Election.run_single_winner_from_ballots(
        ballots,
        nb_candidate=5,
    )
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(1, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(0, 12),
    }


def test_ballots9():
    """Ensures accurate half-half lottery for two top contenders in complex ballots."""
    ballots = [((1, 2, 3, 4, 5), (5, (4, 3), 2, 1), ((4, 5), 1, 2, 3))]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(0, 12),
        "C": round(0, 12),
        "D": round(1 / 2, 12),
        "E": round(1 / 2, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(0, 12),
        "C": round(0, 12),
        "D": round(1 / 2, 12),
        "E": round(1 / 2, 12),
    }


def test_ballots10():
    """Checks consistent lottery outcomes with nested tuple rankings and completion."""
    ballots = [((1, 2, 3, 4, 5), (5, 4, 3, (2, 1)), ((4, 5), 1, 2, 3))]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(0, 12),
        "C": round(0, 12),
        "D": round(1 / 2, 12),
        "E": round(1 / 2, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(0, 12),
        "C": round(0, 12),
        "D": round(1 / 2, 12),
        "E": round(1 / 2, 12),
    }


def test_ballots11():
    """Validates tie breaking between two candidates with nested tie groups."""
    ballots = [((1, (2, 3, 4, 5)), (2, (1, 3, 4, 5)))]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 2, 12),
        "B": round(1 / 2, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(0, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 2, 12),
        "B": round(1 / 2, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(0, 12),
    }


def test_ballots12():
    """Tests complete vs incomplete votes effect on two-level ballots rankings."""
    ballots = [((1, 2, 3), (2, 3))]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=3, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1, 12),
        "B": round(0, 12),
        "C": round(0, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=3)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 2, 12),
        "B": round(1 / 2, 12),
        "C": round(0, 12),
    }


def test_ballots13():
    """Verifies half-half split for two candidates with reciprocal preferences."""
    ballots = [((1, 2), (2, 1))]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=2, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 2, 12),
        "B": round(1 / 2, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=2)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 2, 12),
        "B": round(1 / 2, 12),
    }


def test_ballots14():
    """Checks handling of three-level nested preferences in five-candidate ballots."""
    ballots = [((1, 5, (2, 3, 4)), (2, (1, 3, 4, 5)))]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 2, 12),
        "B": round(1 / 2, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(0, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 2, 12),
        "B": round(1 / 2, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(0, 12),
    }


def test_ballots15():
    """Ensures correct distribution for partial and full ballots with three levels."""
    ballots = [((1, (2, 3, 4, 5)), (2, (1, 3, 4, 5)), (5, 4, 3, 2, 1))]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(1 / 2, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(1 / 2, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(1 / 2, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(1 / 2, 12),
    }


def test_ballots16():
    """Validates uniform lottery when all candidates form a Condorcet cycle."""
    ballots = [
        (
            (1, (2, 3, 4, 5)),
            (2, (1, 3, 4, 5)),
            (3, (4, 5, 2, 1)),
            (4, (5, 3, 2, 1)),
            (5, (4, 3, 2, 1)),
        )
    ]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 5, 12),
        "B": round(1 / 5, 12),
        "C": round(1 / 5, 12),
        "D": round(1 / 5, 12),
        "E": round(1 / 5, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 5, 12),
        "B": round(1 / 5, 12),
        "C": round(1 / 5, 12),
        "D": round(1 / 5, 12),
        "E": round(1 / 5, 12),
    }


def test_ballots17():
    """Tests equal chances for four candidates with symmetric ranked ballots."""
    ballots = [[(4, 1, 3), (3, 2, 4), (1, 3, 2), (2, 4, 1)]]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=4, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 4, 12),
        "B": round(1 / 4, 12),
        "C": round(1 / 4, 12),
        "D": round(1 / 4, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=4)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 4, 12),
        "B": round(1 / 4, 12),
        "C": round(1 / 4, 12),
        "D": round(1 / 4, 12),
    }


def test_ballots18():
    """Confirms single-candidate victory when repeated top choice ballots."""
    ballots = [((1, (2, 3, 4, 5)), (2, (1, 3, 4, 5)), (5, 4, 3, 2, 1), (5, 4, 3, 2, 1))]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(0, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(1, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=5)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(0, 12),
        "B": round(0, 12),
        "C": round(0, 12),
        "D": round(0, 12),
        "E": round(1, 12),
    }


def test_ballots19():
    """Verifies cyclic three-candidate ballots produce uniform lottery."""
    ballots = [((1, 2, 3), (2, 3, 1), (3, 1, 2))]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=3, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(1 / 3, 12),
        "C": round(1 / 3, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=3)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 3, 12),
        "B": round(1 / 3, 12),
        "C": round(1 / 3, 12),
    }


def test_ballots20():
    """Checks that empty ballots raise AttributeError in both modes."""
    ballots = [()]
    with pytest.raises(AttributeError):
        elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=3, complete_votes=False)
    with pytest.raises(AttributeError):
        elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=3)


def test_ballots21():
    """Ensures correct half-half lottery for two disjoint candidate groups."""
    ballots = [(("A", "B"), ("C", "D"))]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=4, complete_votes=False)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 2, 12),
        "B": round(0, 12),
        "C": round(1 / 2, 12),
        "D": round(0, 12),
    }
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=4)
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "A": round(1 / 2, 12),
        "B": round(0, 12),
        "C": round(1 / 2, 12),
        "D": round(0, 12),
    }


def test_ballots22():
    """Validates nb_voter count based on valid vs invalid vote entries."""
    ballots = [(("A", "B"), (3, 4))]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=4, complete_votes=False)
    assert elect.nb_voter == 2
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=4)
    assert elect.nb_voter == 2
    ballots = [(("A", "B"), ("C", 4))]
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=4, complete_votes=False)
    assert elect.nb_voter == 1
    elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=4)
    assert elect.nb_voter == 1
    ballots = [(("A", 2), ("C", 4))]
    with pytest.raises(AttributeError):
        elect = Election.run_single_winner_from_ballots(ballots, nb_candidate=4)


def test_ballots23():
    """Tests named candidates with nested tuples preserve correct lottery probabilities."""
    ballots = [
        (
            ("Luc", "Han", "Chewie", "Yoda", "Ben"),
            ("Ben", ("Yoda", "Chewie"), "Han", "Luc"),
            (("Yoda", "Ben"), "Luc", "Han", "Chewie"),
        )
    ]
    elect = Election.run_single_winner_from_ballots(
        ballots, nb_candidate=5, candidates=["Luc", "Han", "Chewie", "Yoda", "Ben"], complete_votes=False
    )
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "Luc": round(0, 12),
        "Han": round(0, 12),
        "Chewie": round(0, 12),
        "Yoda": round(1 / 2, 12),
        "Ben": round(1 / 2, 12),
    }
    elect = Election.run_single_winner_from_ballots(
        ballots, nb_candidate=5, candidates=["Luc", "Han", "Chewie", "Yoda", "Ben"]
    )
    assert {k: round(v, 12) for k, v in elect.best_lottery.items()} == {
        "Luc": round(0, 12),
        "Han": round(0, 12),
        "Chewie": round(0, 12),
        "Yoda": round(1 / 2, 12),
        "Ben": round(1 / 2, 12),
    }


def test_run_from_popularity_properties():
    """Verifies that popularity-based election sets correct candidate count, voter count, and normalized probabilities."""
    np.random.seed(42)
    popularity = {"A": 1.0, "B": 2.0, "C": 3.0}
    total_voters = 60
    elect = Election.run_from_popularity(popularity, total_voters=total_voters, batch_size=10)
    # Propriétés de base
    assert elect.nb_candidate == 3
    assert elect.nb_voter == total_voters
    assert set(elect.candidates) == set(popularity.keys())
    # best_lottery
    lottery = elect.best_lottery
    assert set(lottery.keys()) == set(popularity.keys())
    total_prob = sum(lottery.values())
    assert pytest.approx(1.0, rel=1e-6) == total_prob
    for p in lottery.values():
        assert 0.0 <= p <= 1.0


def test_run_from_popularity_two_candidates_pure_winner():
    """Checks that the more popular candidate wins with probability one in a two-candidate race."""
    np.random.seed(123)
    popularity = {"X": 10.0, "Y": 20.0}
    elect = Election.run_from_popularity(popularity, total_voters=100, batch_size=10)
    lottery = elect.best_lottery
    # Y has a strict majority (>50%)
    assert lottery["X"] == pytest.approx(0.0)
    assert lottery["Y"] == pytest.approx(1.0)


def test_run_from_popularity_wrong_popularity():
    """Checks that wrong popularity implies an error"""
    with pytest.raises(ValueError):
        Election.run_from_popularity({"A": -1, "B": 1})
    with pytest.raises(ValueError):
        Election.run_from_popularity({"A": 0, "B": 0})


def test_run_from_popularity_without_total_voters():
    """Checks that without total_voters the number of voters is the sum of popularities"""
    pop = {"A": 3.0, "B": 2.0}
    elect = Election.run_from_popularity(pop)
    assert elect.nb_voter == 5  # 3+2


def test_run_multiple_winners_from_ballot_simple_condorcet():
    """Integration test confirming iterative Condorcet winner selection for two rounds."""
    # Ballots resulting in Condorcet winner A then B > C
    ballots = [
        ["A", "B", "C"],
        ["A", "C", "B"],
        ["B", "A", "C"],
    ]
    batch_ballots = [ballots]
    winners_display, discarded = Election.run_multiple_winners_from_ballots(
        batch_ballots=batch_ballots,
        nb_winners=2,
        nb_candidate=3,
        candidates=["A", "B", "C"],
        complete_votes=False,
    )
    # first round: only A (>50%)
    # second round: among {B, C}, B wins
    expected_winners = [WinnerInfo(name="A", round=1, chances=1), WinnerInfo(name="B", round=2, chances=1)]
    assert winners_display == expected_winners
    assert discarded == []


def test_pick_multiple_winners_more_winners_than_candidates():
    """Integration test confirming iterative Condorcet winner selection for two rounds."""
    with pytest.raises(ValueError):
        Election.run_multiple_winners_from_ballots([[["A"]]], nb_winners=3, nb_candidate=2)


def test_pick_multiple_winners_check_discarded_hopefull():
    """Checks that discarded is correctly populated"""
    ballots = [[["A", "B", "C"]]] * 30  # equal cycle → uniform lotery
    winners, discarded = Election.run_multiple_winners_from_ballots(
        batch_ballots=[ballots], nb_winners=2, nb_candidate=3, complete_votes=False
    )
    # it should be 2 winners and 1 discarded with the right format
    assert len(winners) == 2
    assert len(discarded) == 1
    d = discarded[0]
    assert isinstance(d, EliminatedInfo)


def test_graph_html_contains_graph_data_and_options():
    template_dir = "src/condorcet/templates"

    # 1) Prépare une élection simple et calcule les données du graphe
    ballots = [[[2, 0, 0, 0, 0], [3, 2, 4, 0, 0], [1, 2, 3, 5, 0]]]
    elect = Election.run_single_winner_from_ballots(
        ballots, nb_candidate=5, candidates=["Luc", "Han", "Chewie", "Yoda", "Ben"], complete_votes=False
    )
    graph_data = elect.get_graph_data()

    # 2) Force le tag pour le rendre testable
    fixed_tag = 12345
    with patch("random.randint", return_value=fixed_tag):
        # On pointe sur le dossier où se trouve graph_template.html
        viz = GraphVisualizer(graph_data, template_dir=template_dir)
        html_output = viz.to_html(width=800, linkColor="#555")

    # 3) Vérifie que l'ID de la div contient le tag fixé
    assert f'id="graphdiv12345"' in html_output

    # 4) Vérifie que les noms de candidats apparaissent bien (dans le JSON.embeddé)
    for name in elect.candidates:
        assert f'"{name}"' in html_output

    # 5) Extrait le JSON embarqué et compare au graph_data Python
    #    On cherche un objet JSON entouré de crochets/braces
    m = re.search(r"JSON.parse\('(.+?)'\)", html_output)
    assert m, "Impossible de trouver la donnée JSON dans le HTML"
    data = json.loads(m.group(1))
    assert data == graph_data.to_dict()

    # 6) Vérifie que les options passées (width, linkColor) se retrouvent dans le contexte
    assert "var width = 800;" in html_output
    assert 'var linkColor = "#555";' in html_output
