from __future__ import annotations

import string
from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Any, Self, TypeGuard

import numpy as np
import pandas as pd
import toolz as itz
from scipy.optimize import linprog
from tqdm import tqdm

from .types import Ballots, BatchBallots, EliminatedInfo, GraphData, IntVote, LinkData, NodeData, Vote, WinnerInfo


class Election:
    """
    Election class supporting streaming ballots via batch generators.
    """

    def __init__(
        self,
        nb_candidate: int,
        nb_voter: int,
        candidates: list[str],
        duels: np.ndarray,
        complete_votes: bool = True,
    ) -> None:
        self.nb_candidate = nb_candidate
        self.nb_voter = nb_voter
        self.candidates = candidates
        self.duels = duels
        self.complete_votes = complete_votes
        self.payoffs = None
        self.df_payoffs = None
        self.best_lottery = None
        self.not_complete = False

    @staticmethod
    def _is_homogeneous_sequence(seq: Sequence[Any], element_type: type) -> bool:
        """Vrai si seq est un Sequence (sauf str/bytes) et
        tous ses éléments sont soit element_type soit
        un Sequence homogène de element_type."""
        for item in seq:
            if isinstance(item, element_type):
                continue
            # c’est une sous-séquence (attention, pas str/bytes)
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                if not all(isinstance(x, element_type) for x in item):
                    return False
            else:
                return False
        return True

    @classmethod
    def is_valid_vote(cls, obj: Any) -> TypeGuard[Vote]:
        # 1) doit être un Sequence (liste, tuple…) mais pas une str/bytes
        if not isinstance(obj, Sequence) or isinstance(obj, (str, bytes)):
            return False
        # 2) valider soit la version int, soit la version str
        if cls._is_homogeneous_sequence(obj, int):
            return True
        if cls._is_homogeneous_sequence(obj, str):
            return True
        return False

    @staticmethod
    def _normalize_vote(vote: Vote, candidates: list[str]) -> IntVote:
        """Normalize vote by changing each candidate name by its index in candidates list"""
        cleaned: list = []
        for lvl in vote:
            if isinstance(lvl, str):
                cleaned.append(candidates.index(lvl) + 1)
            elif isinstance(lvl, Iterable) and not isinstance(lvl, str):
                for el in lvl:
                    if isinstance(el, str):
                        pass
                    elif isinstance(el, int):
                        pass
                cleaned.append([candidates.index(d) + 1 if isinstance(d, str) else d for d in lvl])
            else:
                cleaned.append(lvl)
        return cleaned

    @staticmethod
    def _remove_zeros_from_vote(vote: IntVote) -> list[int | list[int]]:
        """Remove zeros and clean nested levels in a single vote"""
        cleaned: list = []
        for lvl in vote:
            if isinstance(lvl, Iterable) and not isinstance(lvl, str):
                trimmed = [x for x in lvl if x != 0]
                if trimmed:
                    cleaned.append(trimmed)
            elif lvl != 0:
                cleaned.append(lvl)
        return cleaned

    @staticmethod
    def _mixed_flatten(seq: Sequence[int | Sequence[int] | str]) -> list[int | str]:
        """Flatten mixed sequences"""
        flat: list[int | str] = []
        for el in seq:
            if isinstance(el, Iterable) and not isinstance(el, str):
                flat.extend(el)
            else:
                flat.append(el)
        return flat

    @classmethod
    def _complete_single_vote(cls, vote: IntVote, nb_candidate: int) -> list[int | list[int]]:
        """Complete a single vote by appending all missing candidates at bottom"""
        flat = cls._mixed_flatten(vote)
        missing = [c for c in range(1, nb_candidate + 1) if c not in flat]
        # original levels cleaned
        cleaned = cls._remove_zeros_from_vote(vote)
        if missing:
            cleaned.append(missing)
        return cleaned

    @staticmethod
    def _frac(x: float) -> str:
        """
        Convert float to fraction assuming that number is float representation of fraction

        Args:
            x (float): Description

        Returns:
            str: Description
        """
        f = Fraction(x).limit_denominator().limit_denominator()
        ratio = "{}/{}".format(f.numerator, f.denominator)
        return ratio

    def get_graph_data(self) -> GraphData:
        """
        Build result dictionary
        'nodes' : list of {'name': candidate name}
        'links' : list of {'source': candidate no, 'target': candidate no,
                           'label': fraction candidate source - fraction candidate target}

        Raises:
            AttributeError: raised if duels, payoffs or best_lottery have not yet been computed

        Returns:
            dict[str, list[dict]]: dict representation of graph with nodes and links
        """
        if self.payoffs is None:
            raise AttributeError("payoffs is None")
        if self.best_lottery is None:
            raise AttributeError("best_lottery is None")

        nodes: list[NodeData] = []
        for candidate in self.candidates:
            node_name = candidate
            node_proba = self._frac(self.best_lottery[candidate]) if self.best_lottery[candidate] > 0 else None
            nodes.append(NodeData(name=node_name, proba=node_proba, winner=False))

        duels_normalized = self.duels / self.nb_voter

        links: list[LinkData] = []
        for i in range(self.nb_candidate):
            for j in range(i + 1, self.nb_candidate):
                if self.payoffs[i, j] > self.payoffs[j, i]:
                    label = "{:.2f} - {:.2f}".format(100 * duels_normalized[i, j], 100 * duels_normalized[j, i])
                    links.append(LinkData(source=i, target=j, label=label))
                elif self.payoffs[i, j] < self.payoffs[j, i]:
                    label = "{:.2f} - {:.2f}".format(100 * duels_normalized[j, i], 100 * duels_normalized[i, j])
                    links.append(LinkData(source=j, target=i, label=label))
            is_winner = True if (self.payoffs[i, :].sum() == self.nb_candidate - 1) else False
            nodes[i].winner = is_winner

        return GraphData(nodes=nodes, links=links)

    @classmethod
    def run_single_winner_from_ballots(
        cls,
        batch_ballots: BatchBallots,
        nb_candidate: int,
        candidates: list[str] | None = None,
        complete_votes: bool = True,
    ) -> Self:
        """
        Run an election by streaming ballots in batches.

        Args:
            batch_ballots (BatchBallots): an iterable yielding batches (lists) of ballots.
            nb_candidate (int): total number of candidates.
            candidates (list[str] | None, optional): optional list of candidate names.
            complete_votes (bool, optional): whether to complete each vote with missing candidates.

        Returns:
            Self: computed election with best lottery.

        Raises:
            AttributeError: raised if no vote has been isued
            TypeError: raised if vote is incorrectly formatted
        """
        if candidates is None:
            candidates = list(string.ascii_uppercase)[:nb_candidate]

        # initialize duel counts
        duels = np.zeros((nb_candidate, nb_candidate), dtype=int)
        total_voters = 0
        total_incorrect_votes = 0
        incorrect_votes_reason = set()

        for batch in tqdm(batch_ballots):
            for raw_vote in batch:
                try:
                    # clean vote
                    if not cls.is_valid_vote(raw_vote):
                        raise TypeError("vote is incorrectly formatted")
                    nomalized_vote = cls._normalize_vote(vote=raw_vote, candidates=candidates)
                    vote = cls._remove_zeros_from_vote(vote=nomalized_vote)
                    if complete_votes:
                        vote = cls._complete_single_vote(vote, nb_candidate)
                except Exception as e:
                    total_incorrect_votes += 1
                    incorrect_votes_reason.add(str(e))
                else:
                    # update duels for this vote
                    for i, higher in enumerate(vote[:-1]):
                        higher_set = (higher,) if not isinstance(higher, Iterable) else higher
                        for lower in vote[i + 1 :]:
                            lower_set = (lower,) if not isinstance(lower, Iterable) else lower
                            for w in higher_set:
                                for l in lower_set:
                                    duels[w - 1, l - 1] += 1
                    total_voters += 1
        if total_voters == 0:
            raise AttributeError("no vote issued")

        # instantiate election
        elect = cls(
            nb_candidate=nb_candidate,
            nb_voter=total_voters,
            candidates=candidates,
            duels=duels,
            complete_votes=complete_votes,
        )

        # compute payoffs and best lottery
        elect._build_table_payoff()
        elect.get_best_lottery()
        return elect

    @classmethod
    def run_multiple_winners_from_ballots(
        cls,
        batch_ballots: BatchBallots,
        nb_winners: int,
        nb_candidate: int,
        candidates: list[str] | None = None,
        complete_votes: bool = True,
    ) -> tuple[list[WinnerInfo], list[EliminatedInfo]]:
        """pick nb_winners iteratively after running election one after another

        Args:
            batch_ballots (BatchBallots): an iterable yielding batches (lists) of ballots.
            nb_winners (int): number of winners wanted
            nb_candidate (int): total number of candidates for this election
            candidates (list[str] | None, optional): optional list of candidate names.
            complete_votes (bool, optional): whether to complete each vote with missing candidates.

        Returns:
            tuple[list[WinnerInfo], list[EliminatedInfo]]: list of winners and list of discarded candidates

        Raises:
            ValueError: raised if number of winners asked is more than number of candidates
        """
        if nb_winners > nb_candidate:
            raise ValueError("too many winners asked")
        elect = cls.run_single_winner_from_ballots(
            batch_ballots, nb_candidate=nb_candidate, candidates=candidates, complete_votes=complete_votes
        )
        discarded_hopefull: list[EliminatedInfo] = []
        i = 1
        winners = []
        winners_display: list[WinnerInfo] = []
        while True:
            hopefull = itz.valfilter(lambda x: x > 0, elect.best_lottery)
            final_winners = list(hopefull.keys())
            if len(hopefull) > (k := nb_winners - len(winners)):
                final_winners = list(
                    np.random.choice(list(hopefull.keys()), size=k, replace=False, p=list(hopefull.values()))
                )
                discarded_hopefull.extend(
                    [
                        EliminatedInfo(name=x, round=i, chances=f"{hopefull[x]:.2f}")
                        for x in list(hopefull.keys())
                        if x not in final_winners
                    ]
                )
            winners_display.extend(
                WinnerInfo(
                    name=x, round=i, chances=f"{hopefull[x]:.2f}" if len(hopefull) > (nb_winners - len(winners)) else 1
                )
                for x in final_winners
            )
            winners.extend(final_winners)
            if len(winners) >= nb_winners:
                break
            elect.get_best_lottery(exclude=winners)
            i += 1

        return winners_display, discarded_hopefull

    def _build_table_payoff(self) -> None:
        """Build payoff matrix and DataFrame"""
        payoffs = np.zeros_like(self.duels, dtype=int)
        for i in range(self.nb_candidate):
            for j in range(i + 1, self.nb_candidate):
                if self.duels[i, j] > self.duels[j, i]:
                    payoffs[i, j], payoffs[j, i] = 1, -1
                elif self.duels[i, j] < self.duels[j, i]:
                    payoffs[j, i], payoffs[i, j] = 1, -1
        self.payoffs = payoffs
        self.df_payoffs = pd.DataFrame(data=payoffs, index=self.candidates, columns=self.candidates)
        if not (payoffs + np.eye(self.nb_candidate, dtype=int)).all():
            self.not_complete = True

    def get_best_lottery(self, exclude: list[str] | None = None) -> None:
        """Compute Condorcet winning lottery

        Args:
            exclude (list[str] | None, optional): Description

        Raises:
            AttributeError: Description
            RuntimeError: Description
        """
        if self.payoffs is None:
            raise AttributeError("payoffs is None")
        # indices of remaining candidates
        exclude = exclude or []
        idx = [self.candidates.index(c) for c in self.candidates if c not in exclude]
        # slice payoff
        P = self.payoffs[np.ix_(idx, idx)]
        # setup LP: minimize v
        n = len(idx)
        A_ub = np.hstack([-np.ones((n, 1)), P])
        b_ub = np.zeros(n)
        A_eq = np.array([[0] + [1] * n])
        b_eq = [1]
        c = np.array([1] + [0] * n)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method="highs")
        if not res.success:
            raise RuntimeError("Linear program failed")
        sol = res.x[1:]
        # if incomplete, refine for fairness
        if self.not_complete:
            # second LP to minimize max probability z
            # constraints: P @ p >= -v
            v_val = res.fun
            cons1 = np.hstack([np.zeros((n, 1)), P])
            cons2 = np.hstack([-np.ones((n, 1)), np.eye(n)])
            A_ub2 = np.vstack([cons1, cons2])
            b_ub2 = np.concatenate([np.zeros(n), np.full(n, v_val)])
            A_eq2 = A_eq
            b_eq2 = b_eq
            c2 = c.copy()
            res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, method="highs")
            if not res2.success:
                raise RuntimeError("Second linear program failed")
            sol = res2.x[1:]
        # map to candidate names
        self.best_lottery = {self.candidates[i]: sol[j] for j, i in enumerate(idx)}

    @classmethod
    def run_from_popularity(
        cls,
        popularity: dict[str, float],
        total_voters: int | None = None,
        batch_size: int = 1000,
    ) -> Self:
        """
        Run an election by generating ballots where the first choice is sampled
        proportionally to popularity and the remaining candidates are randomly ordered.

        Args:
            popularity: mapping from candidate name to a non-negative score.
            total_voters: optional total number of voters.
            batch_size: size of batches for streaming ballots.

        Returns:
            Self: computed election with best_lottery.
        """
        # Prepare candidates and probabilities
        candidates = list(popularity.keys())
        scores = np.array([popularity[c] for c in candidates], dtype=float)
        if np.any(scores < 0):
            raise ValueError("Popularity scores must be non-negative")
        sum_scores = scores.sum()
        if sum_scores == 0:
            raise ValueError("At least one candidate must have positive popularity")
        probs = scores / sum_scores
        nb_candidate = len(candidates)
        # Determine number of voters
        n = int(total_voters) if total_voters is not None else int(sum_scores)

        # Generator yielding batches of ballots
        def ballots_batches() -> Iterable[Ballots]:
            generated = 0
            while generated < n:
                size = min(batch_size, n - generated)
                batch: list[Ballots] = []
                for _ in range(size):
                    # sample first choice
                    first = np.random.choice(candidates, p=probs)
                    # randomize the remaining order
                    others = [c for c in candidates if c != first]
                    np.random.shuffle(others)
                    batch.append([first] + others)
                generated += size
                yield batch

        # Run election on generated ballots
        return cls.run_single_winner_from_ballots(
            batch_ballots=ballots_batches(),
            nb_candidate=nb_candidate,
            candidates=candidates,
            complete_votes=True,
        )
