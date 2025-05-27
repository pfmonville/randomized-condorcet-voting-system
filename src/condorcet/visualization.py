import json
import random
from pathlib import Path

import jinja2 as jj

from .types import GraphData


class GraphVisualizer:
    """
    Utility to render Condorcet election results (duels + lottery) as a force-directed HTML graph.

    Expects to find a Jinja2 template named 'graph_template.html' in a subfolder
    'templates/graph' alongside this module.
    """

    def __init__(
        self,
        graph_data: GraphData,
        template_dir: str | Path | None = None,
    ) -> None:
        """
        Args:
            graph_data: Dictionary with keys 'nodes' and 'links', as produced by Election._compute_graph_data().
            template_dir: Optional override for where to find the Jinja2 template folder.
        """
        self.data = graph_data
        # Determine template directory
        if template_dir:
            tpl_dir = Path(template_dir)
        else:
            tpl_dir = Path(__file__).parent / "templates" / "graph"
        # Configure Jinja2 environment
        self.env = jj.Environment(
            loader=jj.FileSystemLoader(str(tpl_dir)),
            variable_start_string="__$",
            variable_end_string="$__",
            autoescape=False,
        )

    def to_html(
        self,
        width: int = 960,
        height: int = 500,
        linkDistance: int = 200,
        linkColor: str = "#121212",
        labelColor: str = "#aaa",
        charge: int = -300,
        theta: float = 0.1,
        gravity: float = 0.05,
    ) -> str:
        """
        Render the force-directed graph as an HTML snippet.

        Returns:
            A string containing <div>, <style>, and <script> for embedding.
        """
        template = self.env.get_template("graph_template.html")
        # Unique tag to avoid collisions if multiple graphs on one page
        random_tag = str(random.randint(0, 1_000_000))
        context = {
            "tag": random_tag,
            "json_data": json.dumps(self.data.to_dict()),
            "width": width,
            "height": height,
            "linkDistance": linkDistance,
            "linkColor": linkColor,
            "labelColor": labelColor,
            "Charge": charge,
            "Theta": theta,
            "Gravity": gravity,
        }
        return template.render(context)
