
import os
import json
from collections.abc import Iterable

import numpy as np
import pandas as pd

from IPython.display import display, clear_output

from scipy.optimize import linprog
from bs4 import BeautifulSoup
from IPython.display import HTML
from fractions import Fraction
from urllib.parse import urlparse

def is_not_all_zero(element):
    """Check if element is zero or only contains zeros
    """
    if isinstance(element, Iterable):
        return any(sub_element > 0 for sub_element in element)
    else:
        return element > 0

def get_sublists(l):
    """
    Get all sub lists of l
    """
    return (sub_l for sub_l in l if isinstance(sub_l, Iterable))

def count_non_zero_element(l):
    """Count all elements in l which are not zero or do not only contains zeros
    """
    return sum([1 for element in l if is_not_all_zero(element)])

class Election:
    """
    Random election class

    nb_candidate: number of candidates - int
    nb_voter: number of voters - int
    proba_ranked: array of probabilities that a candidate is ranked by a voter - size = nb_candidate
    popularity: array of numbers that determine relative popularity of candidates with voters
    - size = nb_candidate
    """

    def __init__(self, nb_candidate, candidates, proba_ranked, popularity, nb_voter):
        self.nb_candidate = nb_candidate
        self.candidates = candidates
        self.proba_ranked = proba_ranked
        self.popularity = popularity
        self.nb_voter = nb_voter
        self.ballot = None
        self.df_ballot = None
        self.duels = None
        self.df_duels = None
        self.payoffs = None
        self.df_payoffs = None
        self.best_lottery = None
        self.graph_data = None
        self.graph_html = None

    def overview_candidates(self):
        """
        Show candidates statistics
        """
        df = pd.DataFrame(data=np.vstack([self.proba_ranked, self.popularity]).T,
                          index=self.candidates,
                          columns=['Ranked Proba', 'Popularity'])
        ax = df.plot.bar(
            figsize=(12, 5), title='Overview of Candidates Ranked Proba and Popularity')
        ax.set_ylabel('Value')
        ax.set_xlabel('Candidate')

    def run_election(self, seed=None):
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
        """
        if seed is not None:
            np.random.seed(seed)

        random1 = np.random.rand(self.nb_voter, self.nb_candidate)
        random2 = np.random.rand(self.nb_voter, self.nb_candidate)

        ranked_candidates = random2 < self.proba_ranked
        score_candidates = random1 * self.popularity * ranked_candidates

        idx = list(np.ix_(*[np.arange(i) for i in score_candidates.shape]))
        order = score_candidates.argsort(axis=1)[:, ::-1]
        idx[1] = order

        # print(ranked_candidates)
        # print(score_candidates)
        # print(idx)
        # print(order)

        sorted_ranked_candidates = ranked_candidates[idx]

        # print(sorted_ranked_candidates)

        self.ballot = (1 + order) * sorted_ranked_candidates

        # print(ballot)

    def build_table_duels(self):
        """
        Build np.array of duels and corresponding pandas dataframe
        cell (row, col) is the number of preference candidate(row) > candidate(col)
        """

        duels = np.zeros([self.nb_candidate, self.nb_candidate])
        # print(duels)

        for row in self.ballot:
            for c1, v1 in enumerate(row[:-1]):
                v1 = (v1,) if not isinstance(v1, Iterable) else v1
                for v2 in row[c1 + 1 :]:
                    v2 = (v2,) if not isinstance(v2, Iterable) else v2
                    for winner in v1:
                        for loser in v2:
                            if loser > 0:
                                duels[winner - 1, loser - 1] += 1

        # print(duels.sum())
        # print(duels)

        df_duels = pd.DataFrame(
            data=duels, index=self.candidates, columns=self.candidates)
        df_duels.index.name = 'winner'
        df_duels.columns.name = 'loser'

        self.duels = duels
        self.df_duels = df_duels

    def check_table_duels(self):
        """
        Check nb of preferences present in ballots and duels
        """
        n = np.array(list(count_non_zero_element(row) for row in self.ballot))
        coef = np.array([sum([count_non_zero_element(sub_l) for sub_l in get_sublists(row)]) for row in self.ballot])
        coef[coef==0] = 1
        n_ballots = int(((n * (n - 1) / 2)*coef).sum())

        n_duels = int(self.duels.sum())

        if n_ballots != n_duels:
            print('Error')
            return n_ballots, n_duels
        else:
            return n_ballots

    def build_table_payoff(self):
        """
        Build np.array of payoffs and corresponding pandas dataframe
        cell (row, col) = 1 if duels(row, col) > duels(col, row) else -1
        so this is a zero sum game payoff
        """

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
            data=payoffs, index=self.candidates, columns=self.candidates)
        df_payoffs.index.name = 'winner'
        df_payoffs.columns.name = 'loser'

        self.payoffs = payoffs
        self.df_payoffs = df_payoffs

        if np.abs(payoffs + np.eye(self.nb_candidate)).min() == 0:
            print('#'*60)
            print('WARNING: The payoff matrix has zero non-diagonal values.')
            print('Meaning the graph is not complete.')
            print('Consequently the best lottery is not necessarily unique.')
            print('And function "get_best_lottery" may raise an exception')
            print('if the direct/dual solutions are not the same (safety check)')
            print('#'*60, '\n\n')

    def get_best_lottery(self):
        """
        Determine best solution to problem
        Direct and dual problems (Simplex Method - cf notebook)
        Check is same result
        """

        shift = np.abs(self.payoffs.min()) + 2

        A_ub = self.payoffs + shift
        b_ub = np.ones(self.nb_candidate)
        c = -np.ones(self.nb_candidate)
        res_direct = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None))
        if not res_direct.success:
            raise Exception('Error: Solving Simplex Direct failed')
        sol_direct = res_direct.x / res_direct.x.sum()

        A_ub = -(self.payoffs + shift).T
        b_ub = -np.ones(self.nb_candidate)
        c = np.ones(self.nb_candidate)
        res_dual = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None))
        if not res_dual.success:
            raise Exception('Error: Solving Simplex Dual failed')
        sol_dual = res_dual.x / res_dual.x.sum()

        if not np.allclose(sol_direct, sol_dual):
            print('res_direct')
            print(res_direct)
            print('res_dual')
            print(res_dual)
            raise Exception('Error: Direct and Dual solutions are different')

        self.best_lottery = dict(zip(self.candidates, sol_direct))

    @staticmethod
    def frac(x):
        """
        Convert float to fraction assuming that number is float representation of fraction
        """
        f = Fraction(x).limit_denominator().limit_denominator()
        ratio = '{}/{}'.format(f.numerator, f.denominator)
        return ratio

    def build_graph_data(self):
        """
        Build result dictionary
        'nodes' : list of {'name': candidate name}
        'links' : list of {'source': candidate no, 'target': candidate no,
                           'label': fraction candidate source - fraction candidate target}
        """
        nodes = []
        for c in self.candidates:
            d = {}
            d['name'] = c
            p = self.best_lottery[c]
            d['proba'] = self.frac(p) if p > 0 else None
            nodes.append(d)

        is_duels = self.duels is not None

        if is_duels:
            duels2 = self.duels / self.nb_voter

        links = []
        for i in range(self.nb_candidate):
            for j in range(i + 1, self.nb_candidate):
                if self.payoffs[i, j] > self.payoffs[j, i]:
                    label = '{:.2f} - {:.2f}'.format(
                        100 * duels2[i, j], 100 * duels2[j, i]) if is_duels else None
                    links.append({'source': i, 'target': j, 'label': label})
                elif self.payoffs[i, j] < self.payoffs[j, i]:
                    label = '{:.2f} - {:.2f}'.format(
                        100 * duels2[j, i], 100 * duels2[i, j]) if is_duels else None
                    links.append({'source': j, 'target': i, 'label': label})
            win = True if (self.payoffs[i, :].sum()
                           == self.nb_candidate - 1) else False
            nodes[i]['C-winner'] = win

        self.graph_data = {'nodes': nodes, 'links': links}

    

    def build_graph_html(self, width=960, height=500, linkDistance=200, linkColor='#121212',
                         labelColor='#aaa', charge=-300, theta=0.1, gravity=0.05,
                         saved=False):
        """
        Build html based on d3.js force layout template
        inspired from http://bl.ocks.org/jhb/5955887
        """

        import jinja2 as jj

        # get template
        env = jj.Environment(loader=jj.FileSystemLoader(['./graph']),
                             variable_start_string='__$',
                             variable_end_string='$__')
        html_template = env.get_template('graph_template.html')

        # build data to put in template
        random_tag = str(int(np.random.random() * 10000))
        dic_data = {
            'tag': random_tag,
            'json_data': json.dumps(self.graph_data),
            'width': width,
            'height': height,
            'linkDistance': linkDistance,
            'linkColor': linkColor,
            'labelColor': labelColor,
            'Charge': charge,
            'Theta': theta,
            'Gravity': gravity
        }

        # render template
        html_string = html_template.render(dic_data)

        # save as standalone
        if saved:
            if not os.path.exists('saved'):
                os.makedirs('saved')
            with open('saved/graph.html', 'w') as f:
                f.write(html_string)

        # extract pieces from template
        def get_lib_name(url):
            return urlparse(url).path.split('.')[0][1:]

        soup = BeautifulSoup(html_string, 'html.parser')
        js_lib_url_1 = soup.find('head').find_all('script')[0].attrs['src']
        css = soup.find('head').find('style')
        div = soup.find('body').find_all('div')[0]
        js = soup.find('body').find_all('script')[0].contents[0]
        js_lib_name_1 = get_lib_name(js_lib_url_1)
        js_lib = json.dumps([js_lib_url_1])
        js_lib_name = ', '.join([js_lib_name_1])

        # build output from pieces
        html_output = """
        %s
        %s
        <script type="text/javascript">
        require(%s, function(%s) { %s });
        </script>
        """ % (div, css, js_lib, js_lib_name, js)

        self.graph_html = html_output

    def plot_graph(self):
        """
        display graph in Jupyter notebook
        """
        clear_output(wait=True)
        return HTML(self.graph_html)



# def build_graph_html2(self, width=960, height=500, linkDistance=200, linkColor='#121212',
#                          labelColor='#aaa', charge=-300, theta=0.1, gravity=0.05,
#                          saved=False):
#         """
#         Build html based on d3.js force layout template
#         inspired from http://bl.ocks.org/jhb/5955887
#         """
#         with open('./graph/graph_template.html', 'r') as f:
#             html = f.read()

#         dic_data = {
#             '__json_data__': json.dumps(self.graph_data),
#             '__width__': width,
#             '__height__': height,
#             '__linkDistance__': linkDistance,
#             '__linkColor__': '"{}"'.format(linkColor),
#             '__labelColor__': '{}'.format(labelColor),
#             '__Charge__': charge,
#             '__Theta__': theta,
#             '__Gravity__': gravity,
#         }

#         for k, v in dic_data.items():
#             v2 = v if isinstance(v, str) else str(v)
#             html = html.replace(k, v2)

#         if saved:
#             if not os.path.exists('saved'):
#                 os.makedirs('saved')
#             with open('saved/graph.html', 'w') as f:
#                 f.write(html)

#         soup = BeautifulSoup(html, 'html.parser')

#         js = soup.find('body').find('script').contents[0]
#         css = soup.find('head').find('style').contents[0]

#         JS_LIBS = json.dumps(['http://d3js.org/d3.v3.min.js'])

#         html_output = """
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

#         """ % (css, JS_LIBS, js)

#         html_output = html_output.replace(
#             'graphdiv', 'graphdiv' + str(int(np.random.random() * 10000)))

#         self.graph_html = html_output