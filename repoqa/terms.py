"""Code-aware tokenization shared by the lexical index, the hash embedder and query parsing.

`getUserName` and `get_user_name` both index as: getusername, get, user, name — so a question
mentioning "user name" still matches identifiers in either style.
"""
from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_CAMEL = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")

STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were be been being this that these those
    it its into than then so such there their them they we you your i me my our do does did done how what where when
    which who whom why can could should would will shall may might must not no yes all any each every some about
    code function functions method methods class classes file files repo repository project implement implemented
    implementation work works working used use uses using call calls called handle handles handled get set make
    show tell explain find happen happens defined define definition""".split()
)

# Language keywords that look like calls (`if (`, `return (`) but never name a user definition.
CALL_KEYWORDS = frozenset(
    "if for while switch catch return sizeof typeof await yield elif with assert print not and or in is new "
    "function def fn func match case super self this".split()
)
_CALL = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def split_ident(tok: str) -> list[str]:
    parts: list[str] = []
    for piece in tok.split("_"):
        if piece:
            parts.extend(_CAMEL.findall(piece))
    return [p.lower() for p in parts]


def code_terms(text: str, drop_stopwords: bool = False) -> list[str]:
    out: list[str] = []
    for tok in _IDENT.findall(text):
        low = tok.lower()
        if len(low) >= 2 and not (drop_stopwords and low in STOPWORDS):
            out.append(low)
        sub = split_ident(tok)
        if len(sub) > 1:
            out.extend(p for p in sub if len(p) >= 2 and not (drop_stopwords and p in STOPWORDS))
    return out


def query_terms(question: str) -> list[str]:
    seen: dict[str, None] = {}
    for t in code_terms(question, drop_stopwords=True):
        seen.setdefault(t, None)
    return list(seen)


def identifiers_in(question: str) -> list[str]:
    """Tokens in a question that look like code identifiers (camelCase, snake_case, dotted, or `quoted`)."""
    found: dict[str, None] = {}
    for m in re.findall(r"`([^`]+)`", question):
        for part in re.split(r"[.\s:()]+", m):
            if part:
                found.setdefault(part, None)
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*[A-Za-z0-9_]", question):
        if "." in tok:
            found.setdefault(tok, None)   # dotted qualified name, e.g. Retriever.search
        for part in tok.split("."):
            mixed = any(c.isupper() for c in part[1:]) and any(c.islower() for c in part)
            pascal = part[:1].isupper() and part.lower() not in STOPWORDS
            if "_" in part or mixed or pascal:
                found.setdefault(part.rstrip("()"), None)
    return [f for f in found if len(f) >= 3]


_DEF_BEFORE = re.compile(r"(?:\bdef|\bfn|\bfunc|\bfunction|\bclass|\bfunc\s*\([^()]*\))\s*\*?\s*$")


def called_names(text: str, exclude: str | None = None) -> set[str]:
    """Names that appear in call position, minus definitions (`def foo(`, `func (r) foo(`) and `exclude`."""
    out = set()
    for m in _CALL.finditer(text):
        n = m.group(1)
        if n in CALL_KEYWORDS or len(n) < 3 or n == exclude:
            continue
        if _DEF_BEFORE.search(text[max(0, m.start() - 40):m.start()]):
            continue
        out.add(n)
    return out
