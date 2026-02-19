"""
AST Parser — converts source code into tree-sitter AST nodes
and extracts structured features (depth, node types, subtrees).
"""
import json
from dataclasses import dataclass, field
from typing import Optional
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript


LANGUAGE_MAP = {
    "python": Language(tspython.language()),
    "javascript": Language(tsjavascript.language()),
}

# Bug-relevant node types per language
SENSITIVE_NODES = {
    "python": {
        "for_statement", "while_statement", "if_statement",
        "call", "binary_operator", "comparison_operator",
        "assignment", "return_statement", "except_clause",
    },
    "javascript": {
        "for_statement", "for_in_statement", "while_statement",
        "if_statement", "call_expression", "binary_expression",
        "assignment_expression", "return_statement", "catch_clause",
    },
}


@dataclass
class ASTNode:
    node_type: str
    text: str
    start_line: int
    end_line: int
    depth: int
    children: list = field(default_factory=list)
    is_sensitive: bool = False


@dataclass
class ASTFeatures:
    max_depth: int
    num_nodes: int
    num_sensitive_nodes: int
    node_type_sequence: list[str]          
    sensitive_node_positions: list[int]    
    loop_nesting_depth: int
    has_exception_handling: bool
    cyclomatic_complexity: int


class ASTParser:
    """
    Parses source code using tree-sitter and extracts structured
    AST features for input to the transformer model.
    """

    def __init__(self, language: str = "python"):
        if language not in LANGUAGE_MAP:
            raise ValueError(f"Unsupported language: {language}. Supported: {list(LANGUAGE_MAP.keys())}")
        self.language = language
        self.parser = Parser(LANGUAGE_MAP[language])
        self.sensitive_types = SENSITIVE_NODES.get(language, set())

    def parse(self, code: str) -> tuple[ASTNode, ASTFeatures]:
        """Parse code and return root ASTNode + extracted features."""
        tree = self.parser.parse(bytes(code, "utf8"))
        root = self._build_ast_node(tree.root_node, depth=0)
        features = self._extract_features(tree.root_node)
        return root, features

    def _build_ast_node(self, node: Node, depth: int) -> ASTNode:
        text = node.text.decode("utf8") if node.text else ""
        ast_node = ASTNode(
            node_type=node.type,
            text=text[:200],        # cap text length
            start_line=node.start_point[0],
            end_line=node.end_point[0],
            depth=depth,
            is_sensitive=node.type in self.sensitive_types,
            children=[
                self._build_ast_node(child, depth + 1)
                for child in node.children
                if not child.is_named or child.type not in {"comment"}
            ],
        )
        return ast_node

    def _extract_features(self, root: Node) -> ASTFeatures:
        node_types = []
        sensitive_positions = []
        max_depth = [0]
        loop_depth = [0]
        max_loop_depth = [0]
        cyclomatic = [1]

        loop_types = {"for_statement", "while_statement", "for_in_statement"}
        branch_types = {"if_statement", "elif_clause", "case_clause", "catch_clause", "except_clause"}

        def dfs(node: Node, depth: int, current_loop_depth: int):
            max_depth[0] = max(max_depth[0], depth)
            node_types.append(node.type)
            idx = len(node_types) - 1

            if node.type in self.sensitive_types:
                sensitive_positions.append(idx)

            if node.type in loop_types:
                current_loop_depth += 1
                max_loop_depth[0] = max(max_loop_depth[0], current_loop_depth)
            if node.type in branch_types or node.type in loop_types:
                cyclomatic[0] += 1
            for child in node.children:
                dfs(child, depth + 1, current_loop_depth)

            if node.type in loop_types:
                current_loop_depth -= 1

        dfs(root, 0, 0)

        has_except = any(t in {"except_clause", "catch_clause"} for t in node_types)

        return ASTFeatures(
            max_depth=max_depth[0],
            num_nodes=len(node_types),
            num_sensitive_nodes=len(sensitive_positions),
            node_type_sequence=node_types,
            sensitive_node_positions=sensitive_positions,
            loop_nesting_depth=max_loop_depth[0],
            has_exception_handling=has_except,
            cyclomatic_complexity=cyclomatic[0],
        )

    def to_flat_token_sequence(self, code: str) -> list[str]:
        """
        Returns DFS-ordered node type tokens for use as auxiliary
        input alongside code tokens in the transformer.
        """
        _, features = self.parse(code)
        return features.node_type_sequence

    def serialize(self, node: ASTNode) -> dict:
        """Serialize AST to JSON-compatible dict (for dataset storage)."""
        return {
            "type": node.node_type,
            "text": node.text,
            "start_line": node.start_line,
            "end_line": node.end_line,
            "depth": node.depth,
            "is_sensitive": node.is_sensitive,
            "children": [self.serialize(c) for c in node.children],
        }
