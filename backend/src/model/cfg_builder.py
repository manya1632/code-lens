"""
Control Flow Graph (CFG) Builder.

Builds a CFG from AST nodes using networkx.
Each node in the CFG represents a basic block (straight-line code).
Edges represent possible control transfers (branches, loops, exceptions).
CFG features are extracted as a fixed-size embedding for the model.
"""

import networkx as nx
from dataclasses import dataclass
from tree_sitter import Node
from typing import Tuple, List
from src.model.ast_parser import ASTParser



@dataclass
class CFGNode:
    block_id: int
    statements: List[str]
    start_line: int
    end_line: int
    node_type: str   # entry | exit | branch | loop | normal


@dataclass
class CFGFeatures:
    num_nodes: int
    num_edges: int
    cyclomatic_complexity: int
    max_path_length: int
    num_back_edges: int
    has_unreachable_code: bool
    average_block_size: float



class CFGBuilder:
    """
    Builds a Control Flow Graph from source code.
    """

    def __init__(self, language: str = "python"):
        self.language = language
        self.ast_parser = ASTParser(language)
        self._counter = 0
        self.entry_id = None
        self.exit_id = None


    def _new_id(self) -> int:
        self._counter += 1
        return self._counter


    def build(self, code: str) -> Tuple[nx.DiGraph, CFGFeatures]:
        self._counter = 0
        graph = nx.DiGraph()

        tree = self.ast_parser.parser.parse(bytes(code, "utf8"))

        self.entry_id = self._new_id()
        self.exit_id = self._new_id()

        graph.add_node(
            self.entry_id,
            data=CFGNode(self.entry_id, ["ENTRY"], 0, 0, "entry")
        )

        graph.add_node(
            self.exit_id,
            data=CFGNode(self.exit_id, ["EXIT"], -1, -1, "exit")
        )

        last_nodes = self._process_node(
            tree.root_node,
            graph,
            predecessors=[self.entry_id],
            code=code
        )

        for node_id in last_nodes:
            graph.add_edge(node_id, self.exit_id)

        features = self._extract_features(graph)

        return graph, features


    def _process_node(
        self,
        node: Node,
        graph: nx.DiGraph,
        predecessors: List[int],
        code: str,
    ) -> List[int]:

        if node.type in {"if_statement", "if_clause"}:
            return self._handle_branch(node, graph, predecessors, code)

        if node.type in {"for_statement", "while_statement", "for_in_statement"}:
            return self._handle_loop(node, graph, predecessors, code)

     
        if node.type in {"try_statement"}:
            return self._handle_try(node, graph, predecessors, code)

        if node.type in {"return_statement"}:
            block_id = self._create_block(node, graph, predecessors, code)
            graph.add_edge(block_id, self.exit_id)
            return []

    
        if node.type in {
            "module", "block", "suite",
            "function_definition", "function_declaration",
            "class_definition", "class_body"
        }:
            current_preds = predecessors
            for child in node.children:
                if child.is_named:
                    current_preds = self._process_node(child, graph, current_preds, code)
                    if not current_preds:
                        break
            return current_preds

        return [self._create_block(node, graph, predecessors, code)]


    def _create_block(self, node, graph, predecessors, code):
        block_id = self._new_id()

        text = node.text.decode("utf8") if node.text else node.type

        graph.add_node(
            block_id,
            data=CFGNode(
                block_id,
                [text[:200]],
                node.start_point[0],
                node.end_point[0],
                "normal"
            )
        )

        for pred in predecessors:
            graph.add_edge(pred, block_id)

        return block_id


    def _handle_branch(self, node, graph, predecessors, code):
        cond_id = self._new_id()

        graph.add_node(
            cond_id,
            data=CFGNode(cond_id, ["BRANCH"], node.start_point[0], node.start_point[0], "branch")
        )

        for pred in predecessors:
            graph.add_edge(pred, cond_id)

        exits = []

        for child in node.children:
            if child.is_named:
                branch_exit = self._process_node(child, graph, [cond_id], code)
                exits.extend(branch_exit)

        exits.append(cond_id)  

        return exits


    def _handle_loop(self, node, graph, predecessors, code):
        header_id = self._new_id()

        graph.add_node(
            header_id,
            data=CFGNode(header_id, ["LOOP"], node.start_point[0], node.start_point[0], "loop")
        )

        for pred in predecessors:
            graph.add_edge(pred, header_id)

        body_exits = []

        for child in node.children:
            if child.is_named:
                body_exits = self._process_node(child, graph, [header_id], code)

        for ex in body_exits:
            graph.add_edge(ex, header_id)

        return [header_id]


    def _handle_try(self, node, graph, predecessors, code):
        exits = []
        for child in node.children:
            if child.is_named:
                block_exit = self._process_node(child, graph, predecessors, code)
                exits.extend(block_exit)
        return exits


    def _extract_features(self, graph: nx.DiGraph) -> CFGFeatures:

        N = graph.number_of_nodes()
        E = graph.number_of_edges()

        # Cyclomatic complexity
        P = max(1, len(list(nx.weakly_connected_components(graph))))
        cyclomatic = max(1, E - N + 2 * P)

        # Back edges (cycle detection via DFS)
        num_back_edges = 0
        try:
            for u, v in graph.edges():
                if nx.has_path(graph, v, u):
                    num_back_edges += 1
        except Exception:
            num_back_edges = 0

        reachable = nx.descendants(graph, self.entry_id) | {self.entry_id}
        has_unreachable = len(reachable) < N

        try:
            if nx.is_directed_acyclic_graph(graph):
                max_path = nx.dag_longest_path_length(graph)
            else:
                max_path = N
        except Exception:
            max_path = N

        block_sizes = [
            len(graph.nodes[n]["data"].statements)
            for n in graph.nodes
            if "data" in graph.nodes[n]
        ]

        avg_block = sum(block_sizes) / max(len(block_sizes), 1)

        return CFGFeatures(
            num_nodes=N,
            num_edges=E,
            cyclomatic_complexity=cyclomatic,
            max_path_length=max_path,
            num_back_edges=num_back_edges,
            has_unreachable_code=has_unreachable,
            average_block_size=avg_block,
        )


    def features_to_vector(self, features: CFGFeatures) -> List[float]:
        return [
            float(features.num_nodes),
            float(features.num_edges),
            float(features.cyclomatic_complexity),
            float(features.max_path_length),
            float(features.num_back_edges),
            float(features.has_unreachable_code),
            float(features.average_block_size),
        ]
