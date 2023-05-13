# TODO add support for get
# TODO add support for min/max assgn
# TODO add support for apply
# TODO handle scipy.sparse.linalg (only support the linalg part)
# MAYBE support random
# MAYBE support control flow
import ast
from ast import (
    Attribute,
    Call,
    NodeTransformer,
    Load,
    Import,
    Name,
    ImportFrom,
    Assign,
    AugAssign,
    Subscript,
    Constant,
)
from typing import Optional


def get_op_name(op: ast.AST) -> Optional[str]:
    if isinstance(op, ast.Add):
        return "add"
    if isinstance(op, ast.Mul):
        return "multiply"
    if isinstance(op, ast.Div):
        return "divide"
    if isinstance(op, ast.Pow):
        return "power"
    return None


class NumpyPort(NodeTransformer):
    name: str  # name used for numpy in the code
    replace_subscript_assgn: bool  # will cause problems on non-arrays

    def __init__(self, replace_subscript_assgn: bool = False):
        self.replace_subscript_assgn = replace_subscript_assgn
        self.name = "np"

    def visit_Import(self, node: Import) -> Import:
        for idx, alias in enumerate(node.names):
            if alias.name == "numpy":
                self.name = alias.asname or alias.name
                node.names[idx].name = "jax.numpy"
                node.names[idx].asname = "jnp"
            elif "numpy" in alias.name:
                node.names[idx].name = alias.name.replace("numpy","jax.numpy")
        return node

    def visit_ImportFrom(self, node: ImportFrom) -> ImportFrom:
        if node.module == "numpy":
            node.module = "jax.numpy"
        elif "numpy" in node.module:
            node.module = node.module.replace("numpy","jax.numpy")
        return node

    def visit_Assign(self, node: Assign) -> Optional[Assign]:
        if not self.replace_subscript_assgn:
            return None
        if isinstance(node.targets[0], Subscript):
            target = node.targets[0]
            if isinstance(target.slice, Constant) and not isinstance(
                target.slice.value, int
            ):
                return None  # should flag as unmodified
            # x[y] = z -> x = x.at[y].set(z)
            at = Attribute(value=target.value, attr="at", ctx=Load())  # at[y]
            set_at = Attribute(  # at[y].set(z)
                value=Subscript(value=at, slice=target.slice, ctx=Load()),
                attr="set",
                ctx=Load(),
            )
            node = Assign(  # a = a.at[y].set(z)
                targets=[target.value],
                value=Call(
                    func=set_at,
                    args=[node.value],
                    keywords=[],
                ),
            )
        self.generic_visit(node)
        return node

    def visit_AugAssign(self, node: AugAssign) -> Optional[Assign]:
        if not self.replace_subscript_assgn:
            return None
        if isinstance(node.targets[0], Subscript):
            target = node.targets[0]
            if isinstance(target.slice, Constant) and not isinstance(
                target.slice.value, int
            ):
                return None
            at = Attribute(value=target.value, attr="at", ctx=Load())  # at[y]
            if not (op := get_op_name(node.op)):
                return None
            set_at = Attribute(
                value=Subscript(value=at, slice=target.slice, ctx=Load()),
                attr=op,
                ctx=Load(),
            )
            node = Assign(  # a = a.at[y].set(z)
                targets=[target.value],
                value=Call(
                    func=set_at,
                    args=[node.value],
                    keywords=[],
                ),
            )
        self.generic_visit(node)
        return node

    def visit_Name(self, node: Name) -> Name:
        if node.id == self.name:
            node.id = "jnp"
        return node

class ScipyPort(NodeTransformer):
    not_supported_pkgs = [
            "fftpack",
            "cluster",
            "datasets",
            "io",
            "misc",
            "odr",
            "spatial",
            "sparse"
            ]

    def visit_Import(self, node: Import) -> Import:
        for idx, alias in enumerate(node.names):
            if alias.name == "scipy":
                node.names[idx].name = "jax.scipy"
                node.names[idx].asname = None
            elif "scipy" in alias.name and all(pkg not in alias.name for pkg in self.not_supported_pkgs):
                node.names[idx].name = alias.name.replace("scipy", "jax.scipy")
        return node

    def visit_ImportFrom(self, node: ImportFrom) -> ImportFrom:
        if all(alias.name not in self.not_supported_pkgs for alias in node.names):
            if node.module == "scipy":
                node.module = "jax.scipy"
            elif "scipy" in node.module and all(pkg not in node.module for pkg in self.not_supported_pkgs):
                node.module = node.module.replace("scipy", "jax.scipy")
        return node

    def visit_Attribute(self, node: Attribute) -> Attribute:
        if isinstance(node.value, Name) and node.value.id == "scipy" and node.attr not in self.not_supported_pkgs:
            node.value = Attribute(value=Name(id='jax', ctx=Load()), attr='scipy', ctx=Load())
        self.generic_visit(node)
        return node

if __name__ == "__main__":
    from argparse import ArgumentParser
    from ast import parse, unparse, fix_missing_locations

    parser = ArgumentParser()
    parser.add_argument("-i", "--input", help="file to port", required=True)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as input_file:
        tree = parse(input_file.read())
    new_tree = fix_missing_locations(NumpyPort(replace_subscript_assgn=True).visit(tree))
    new_tree = fix_missing_locations(ScipyPort().visit(new_tree))
    print(unparse(new_tree))
