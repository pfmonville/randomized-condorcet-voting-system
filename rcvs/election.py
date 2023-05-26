import os
import json
from collections.abc import Iterable, Sequence
import itertools as it
from typing import Any
import string
from collections import Counter

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from IPython.display import display, clear_output

from scipy.optimize import linprog
from bs4 import BeautifulSoup
from IPython.display import HTML
from fractions import Fraction
from urllib.parse import urlparse
import jinja2 as jj

Vote = Sequence[int | Sequence[int]] | Sequence[str | Sequence[str]]
Ballot = Sequence[Vote] | NDArray


class Election:
    """
    Random election class

    Consisting of ballots an election will be run with the randomized condorcet voting system.
    A ballot is a list of ordered candidates.
    There is one ballot for each voter.
    Each candidate is represented by a number common to all ballots
    A voter can orders only a subset of candidates (beware if a candidate is not specified
    in a vote then no information can be infered so if only one candidate is put in a vote
    then it doesn’t have any impact).
    A voter can group candidates on a same level (same rank) like (1,2,(3,4),5) -> 3 and 4
    are preffered to 5 and not preffered to 1 and 2 but no information is given to rank 3
    compared to 4.
    """

    def __init__(
        self,
        nb_candidate: int,
        nb_voter: int,
        candidates: list[str] | None = None,
        proba_ranked: list[float] | None = None,
        popularity: list[float] | None = None,
    ):
        self.nb_candidate = nb_candidate
        self.nb_voter = nb_voter
        if candidates is None:
            self.candidates = list(string.ascii_uppercase)[:nb_candidate]
        else:
            self.candidates = candidates
        self.proba_ranked = proba_ranked
        self.popularity = popularity
        self.ballot = None
        self.df_ballot = None
        self.duels = None
        self.df_duels = None
        self.payoffs = None
        self.df_payoffs = None
        self.best_lottery = None
        self.graph_data = None
        self.graph_html = None

    @staticmethod
    def run_election_from_ballot(
        ballot: Ballot, nb_candidate: int | None = None
    ) -> "Election":
        """Runs the election from ballot given

        Args:
            ballot (Ballot): list of each vote
            nb_candidate (int | None, optional): Force number of candidates

        No Longer Returned:
            Election: the election with best lottery computed
        """

        def mixed_flatten(seq: Vote) -> list[Any]:
            """Flattens a mixed list of "elements" and "list of elements"

            Args:
                seq (Vote): the list to flatten

            Returns:
                list[Any]: the flatten list
            """
            flatten_list = []
            for el in seq:
                if isinstance(el, Iterable) and not isinstance(el, str):
                    flatten_list.extend(el)
                else:
                    flatten_list.append(el)
            return flatten_list

        def remove_zeros(ballot: Ballot) -> Ballot:
            """Removes all zeros in ballot

            (1,2,3,0,0,0) -> (1,2,3)
            (1,0,2,0,3,0) -> (1,2,3)
            (1,(2,3),(0,0)) -> (1,(2,3))
            (1,(2,0,3),(0,4),(0,5,0),(0,0)) -> (1,(2,3),4,5)

            Args:
                ballot (Iterable[Ballot]): the ballot to remove zeros from

            Returns:
                Iterable[Ballot]: the processed ballot
            """
            result = []
            for vote in ballot:
                v = []
                for level in vote:
                    if level == 0:
                        continue
                    if isinstance(level, Iterable):
                        trimed = [x for x in level if x != 0]
                        if not trimed:
                            continue
                        v.append(trimed)
                    else:
                        v.append(level)
                result.append(v)
            return result

        if len(ballot) == 0:
            raise AttributeError(f"The ballot is empty : {ballot}")
        flat_ballot = mixed_flatten(it.chain.from_iterable(ballot))
        if max(max(Counter(mixed_flatten(x)).values()) for x in ballot) > 1:
            raise AttributeError("Ballot cannot contains same candidate multiple times")
        if len(set(type(c) for c in flat_ballot)) >= 2:
            raise AttributeError(
                "Ballot cannot be a mixed of types, only int or only str"
            )
        if all(isinstance(c, str) for c in flat_ballot):
            candidates = list(set(flat_ballot))
            nb_candidate = len(candidates)
            ballot = [
                [
                    candidates.index(c) + 1
                    if isinstance(c, str)
                    else [candidates.index(d) + 1 for d in c]
                    for c in b
                ]
                for b in ballot
            ]
        else:
            ballot = remove_zeros(ballot)
            if nb_candidate is None:
                nb_candidate = len(set(mixed_flatten(it.chain.from_iterable(ballot))))
            candidates = list(string.ascii_uppercase)[:nb_candidate]
        nb_voter = len(ballot)
        proba_ranked = None
        popularity = None

        elect = Election(
            nb_candidate=nb_candidate,
            candidates=candidates,
            proba_ranked=proba_ranked,
            popularity=popularity,
            nb_voter=nb_voter,
        )

        elect.ballot = ballot

        elect.build_table_duels()
        elect.check_table_duels()
        elect.build_table_payoff()
        elect.get_best_lottery()

        return elect

    def overview_candidates(self):
        """
        Show candidates statistics
        """
        if self.proba_ranked is None:
            raise AttributeError("proba_ranked is None")
        if self.popularity is None:
            raise AttributeError("popularity is None")

        df = pd.DataFrame(
            data=np.vstack([self.proba_ranked, self.popularity]).T,
            index=self.candidates,
            columns=["Ranked Proba", "Popularity"],
        )
        ax = df.plot.bar(
            figsize=(12, 5), title="Overview of Candidates Ranked Proba and Popularity"
        )
        ax.set_ylabel("Value")
        ax.set_xlabel("Candidate")

    def run_election_from_popularity(self, seed: int | None = None):
        """
        Run election

        For a voter:
            the propability of a candidate being ranked is proba_ranked
            the score of a candidate is a random number (between 0 and 1) times its popularity

            The resulting ballot is a 2d np.array of nb_voter x nb_candidate
            Each line is the ballot of a voter
            It contains the candidates identified by index (from 1 to nb_candidate) and decreasing
            order of preference. The ranked candidates come first then as many 0's as there are
            unranked candidates

        Args:
            seed (int | None, optional): Description

        Raises:
            Exception: Description
        """
        if self.proba_ranked is None or self.popularity is None:
            raise Exception(
                "the election must have proba_ranked and popularity attributes set to run election"
            )
        if seed is not None:
            np.random.seed(seed)

        random1 = np.random.rand(self.nb_voter, self.nb_candidate)
        random2 = np.random.rand(self.nb_voter, self.nb_candidate)

        ranked_candidates = random2 < self.proba_ranked
        score_candidates = random1 * self.popularity * ranked_candidates

        idx = list(np.ix_(*[np.arange(i) for i in score_candidates.shape]))
        order = score_candidates.argsort(axis=1)[:, ::-1]
        idx[1] = order

        sorted_ranked_candidates = ranked_candidates[tuple(idx)]

        self.ballot = (1 + order) * sorted_ranked_candidates

    def build_table_duels(self):
        """
        Build np.array of duels and corresponding pandas dataframe
        cell (row, col) is the number of preference candidate(row) > candidate(col)

        Raises:
            AttributeError: Description
        """
        if self.ballot is None:
            raise AttributeError("ballot is None")

        duels = np.zeros([self.nb_candidate, self.nb_candidate])

        for row in self.ballot:
            for c1, v1 in enumerate(row[:-1]):
                v1 = (v1,) if not isinstance(v1, Iterable) else v1
                for v2 in row[c1 + 1 :]:
                    v2 = (v2,) if not isinstance(v2, Iterable) else v2
                    for winner in v1:
                        for loser in v2:
                            if loser > 0:
                                duels[winner - 1, loser - 1] += 1

        df_duels = pd.DataFrame(
            data=duels, index=self.candidates, columns=self.candidates
        )
        df_duels.index.name = "winner"
        df_duels.columns.name = "loser"

        self.duels = duels
        self.df_duels = df_duels

    def check_table_duels(self) -> int | tuple[int, int]:
        """
        Check nb of preferences present in ballots and duels

        Returns:
            int | tuple[int, int]

        Raises:
            AttributeError: Description
        """

        def my_combinations(seq: list[int], n: int) -> tuple[tuple[int]]:
            """Summary

            Args:
                seq (list[int]): Description
                n (int): Description

            Returns:
                list[list[int]]: Description
            """
            result = tuple(it.combinations(seq, n))
            if not result:
                result = ((0,),)
            return result

        if self.ballot is None:
            raise AttributeError("ballot is None")

        if self.duels is None:
            raise AttributeError("duels is None")

        ballot_sizes = [
            [len(x) if isinstance(x, Sequence) else 1 for x in ballot if x != 0]
            for ballot in self.ballot
        ]
        n_ballots = np.sum(
            np.concatenate(
                [
                    np.prod(my_combinations(ballot_size, 2), 1)
                    for ballot_size in ballot_sizes
                ]
            )
        )
        n_duels = int(self.duels.sum())

        if n_ballots != n_duels:
            print("Error")
            return n_ballots, n_duels
        else:
            return n_ballots

    def build_table_payoff(self):
        """
        Build np.array of payoffs and corresponding pandas dataframe
        cell (row, col) = 1 if duels(row, col) > duels(col, row) else -1
        so this is a zero sum game payoff

        Raises:
            AttributeError: Description
        """
        if self.duels is None:
            raise AttributeError("duels is None")

        payoffs = np.zeros_like(self.duels)

        for i in range(self.nb_candidate):
            for j in range(i + 1, self.nb_candidate):
                if self.duels[i, j] > self.duels[j, i]:
                    payoffs[i, j] = 1
                    payoffs[j, i] = -1
                elif self.duels[i, j] < self.duels[j, i]:
                    payoffs[j, i] = 1
                    payoffs[i, j] = -1

        df_payoffs = pd.DataFrame(
            data=payoffs, index=self.candidates, columns=self.candidates
        )
        df_payoffs.index.name = "winner"
        df_payoffs.columns.name = "loser"

        self.payoffs = payoffs
        self.df_payoffs = df_payoffs

        if np.abs(payoffs + np.eye(self.nb_candidate)).min() == 0:
            print("#" * 60)
            print("WARNING: The payoff matrix has zero non-diagonal values.")
            print("Meaning the graph is not complete.")
            print("Consequently the best lottery is not necessarily unique.")
            print("The solver will return the fairest lottery between")
            print("those which share the same minimax objective value")
            print("#" * 60, "\n\n")

    def get_best_lottery(self):
        """
        Determine best solution to problem
        Gets the lexicographic solution for bi-objective
        1: minimize maximum gain from opponent
        2: minimizing the maximum probability to ensure fairness

        Raises:
            RuntimeError: Description

        No Longer Raises:
            Exception: Description
        """

        # ===============
        # Solver 1
        # ===============

        # define the maximum value between each opponent strategy
        # v ⩾ ∑_j(A[i,j]*p[j]) ∀i
        A_ub = np.c_[
            -np.ones((self.nb_candidate)),
            self.payoffs,
        ]
        b_ub = np.zeros((len(A_ub)))

        # défine that the sum of all probabilities for plays is 1
        # ∑_j(p[j]) == 1
        A_eq = np.array([[0, *[1 for _ in range(self.nb_candidate)]]])
        b_eq = 1

        # define objective as minimizing the maximum gain from opponent
        c = np.array([1, *[0 for _ in range(self.nb_candidate)]])
        res_direct = linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method="highs"
        )
        if not res_direct.success:
            raise RuntimeError("Error: Solving Simplex Direct failed")

        # get solution from solver 1
        v = res_direct.fun

        # ===============
        # Solver 2
        # ===============

        # define the maximum value between each opponent strategy
        # v ⩾ ∑_j(A[i,j]*p[j]) ∀i
        cons1 = np.c_[
            np.zeros((self.nb_candidate)),
            self.payoffs,
        ]
        # défine z as the maximum between all probabilities of each play
        # z ⩾ p[j] ∀j
        cons2 = np.c_[
            -np.ones((self.nb_candidate)),
            np.eye((self.nb_candidate)),
        ]
        A_ub2 = np.r_[cons1, cons2]
        b_ub2 = np.r_[np.zeros((len(cons1))), np.zeros((len(cons2))) + v]

        # défine that the sum of all probabilities for plays is 1
        # ∑_j(p[j]) == 1
        A_eq2 = np.array([[0, *[1 for _ in range(self.nb_candidate)]]])
        b_eq2 = 1

        # define objective as minimizing the maximum probability of a play (which cause fairness)
        c2 = np.array([1, *[0 for _ in range(self.nb_candidate)]])
        res_direct2 = linprog(
            c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2, method="highs"
        )
        if not res_direct2.success:
            raise RuntimeError("Error: Solving Simplex Direct failed")

        # retrieve probabilities values (p) from solution
        solution = res_direct2.x[1:]

        # Set best lottery
        self.best_lottery = dict(zip(self.candidates, solution))

    @staticmethod
    def frac(x: float) -> str:
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

    def build_graph_data(self):
        """
        Build result dictionary
        'nodes' : list of {'name': candidate name}
        'links' : list of {'source': candidate no, 'target': candidate no,
                           'label': fraction candidate source - fraction candidate target}

        Raises:
            AttributeError: Description
        """
        if self.payoffs is None:
            raise AttributeError("duels is None")
        if self.best_lottery is None:
            raise AttributeError("best_lottery is None")

        nodes = []
        for c in self.candidates:
            d = {}
            d["name"] = c
            p = self.best_lottery[c]
            d["proba"] = self.frac(p) if p > 0 else None
            nodes.append(d)

        if self.duels is not None:
            duels2 = self.duels / self.nb_voter

        links = []
        for i in range(self.nb_candidate):
            for j in range(i + 1, self.nb_candidate):
                if self.payoffs[i, j] > self.payoffs[j, i]:
                    label = (
                        "{:.2f} - {:.2f}".format(100 * duels2[i, j], 100 * duels2[j, i])
                        if self.duels is not None
                        else None
                    )
                    links.append({"source": i, "target": j, "label": label})
                elif self.payoffs[i, j] < self.payoffs[j, i]:
                    label = (
                        "{:.2f} - {:.2f}".format(100 * duels2[j, i], 100 * duels2[i, j])
                        if self.duels is not None
                        else None
                    )
                    links.append({"source": j, "target": i, "label": label})
            win = True if (self.payoffs[i, :].sum() == self.nb_candidate - 1) else False
            nodes[i]["C-winner"] = win

        self.graph_data = {"nodes": nodes, "links": links}

    def build_graph_html(
        self,
        width=960,
        height=500,
        linkDistance=200,
        linkColor="#121212",
        labelColor="#aaa",
        charge=-300,
        theta=0.1,
        gravity=0.05,
        saved=False,
    ):
        """
        Build html based on d3.js force layout template
        inspired from http://bl.ocks.org/jhb/5955887

        Args:
            width (int, optional): Description
            height (int, optional): Description
            linkDistance (int, optional): Description
            linkColor (str, optional): Description
            labelColor (str, optional): Description
            charge (TYPE, optional): Description
            theta (float, optional): Description
            gravity (float, optional): Description
            saved (bool, optional): Description
        """
        # get template
        env = jj.Environment(
            loader=jj.FileSystemLoader(["./graph"]),
            variable_start_string="__$",
            variable_end_string="$__",
        )
        html_template = env.get_template("graph_template.html")

        # build data to put in template
        random_tag = str(int(np.random.random() * 10000))
        dic_data = {
            "tag": random_tag,
            "json_data": json.dumps(self.graph_data),
            "width": width,
            "height": height,
            "linkDistance": linkDistance,
            "linkColor": linkColor,
            "labelColor": labelColor,
            "Charge": charge,
            "Theta": theta,
            "Gravity": gravity,
        }

        # render template
        html_string = html_template.render(dic_data)

        # save as standalone
        if saved:
            if not os.path.exists("saved"):
                os.makedirs("saved")
            with open("saved/graph.html", "w") as f:
                f.write(html_string)

        # extract pieces from template
        def get_lib_name(url):
            """Summary

            Args:
                url (TYPE): Description

            Returns:
                TYPE: Description
            """
            return urlparse(url).path.split(".")[0][1:]

        soup = BeautifulSoup(html_string, "html.parser")
        js_lib_url_1 = soup.find("head").find_all("script")[0].attrs["src"]
        css = soup.find("head").find("style")
        div = soup.find("body").find_all("div")[0]
        js = soup.find("body").find_all("script")[0].contents[0]
        js_lib_name_1 = get_lib_name(js_lib_url_1)
        js_lib = json.dumps([js_lib_url_1])
        js_lib_name = ", ".join([js_lib_name_1])

        # build output from pieces
        html_output = """
        %s
        %s
        <script type="text/javascript">
        require(%s, function(%s) { %s });
        </script>
        """ % (
            div,
            css,
            js_lib,
            js_lib_name,
            js,
        )

        self.graph_html = html_output

    def plot_graph(self):
        """
        display graph in Jupyter notebook

        Returns:
            TYPE: Description
        """
        clear_output(wait=True)
        return HTML(self.graph_html)

    # def build_graph_html2(
    #     self,
    #     width=960,
    #     height=500,
    #     linkDistance=200,
    #     linkColor="#121212",
    #     labelColor="#aaa",
    #     charge=-300,
    #     theta=0.1,
    #     gravity=0.05,
    #     saved=False,
    # ):
    #     """
    #     Build html based on d3.js force layout template
    #     inspired from http://bl.ocks.org/jhb/5955887
    #     """
    #     with open("./graph/graph_template.html", "r") as f:
    #         html = f.read()

    #     dic_data = {
    #         "__json_data__": json.dumps(self.graph_data),
    #         "__width__": width,
    #         "__height__": height,
    #         "__linkDistance__": linkDistance,
    #         "__linkColor__": '"{}"'.format(linkColor),
    #         "__labelColor__": "{}".format(labelColor),
    #         "__Charge__": charge,
    #         "__Theta__": theta,
    #         "__Gravity__": gravity,
    #     }

    #     for k, v in dic_data.items():
    #         v2 = v if isinstance(v, str) else str(v)
    #         html = html.replace(k, v2)

    #     if saved:
    #         if not os.path.exists("saved"):
    #             os.makedirs("saved")
    #         with open("saved/graph.html", "w") as f:
    #             f.write(html)

    #     soup = BeautifulSoup(html, "html.parser")

    #     js = soup.find("body").find("script").contents[0]
    #     css = soup.find("head").find("style").contents[0]

    #     JS_LIBS = json.dumps(["http://d3js.org/d3.v3.min.js"])

    #     html_output = """
    #         <div id="graphdiv">
    #         </div>

    #         <style type="text/css">
    #             %s
    #         </style>

    #         <script type="text/javascript">
    #         require(%s, function() {
    #             %s
    #         });
    #         </script>

    #         """ % (
    #         css,
    #         JS_LIBS,
    #         js,
    #     )

    #     html_output = html_output.replace(
    #         "graphdiv", "graphdiv" + str(int(np.random.random() * 10000))
    #     )

    #     self.graph_html = html_output


if __name__ == "__main__":
    ballot = np.array(
        [
            [4, 7, [1, 2, 3, 5, 6, 8]],
            [[4, 7], [2, 8], [1, 3, 5]],
            [7, 4, [1, 2, 3, 5, 6, 8]],
        ],
        dtype=object,
    )
    elect = Election.run_election_from_ballot(ballot, nb_candidate=8)
    print(elect.best_lottery)
    elect.build_graph_html()
