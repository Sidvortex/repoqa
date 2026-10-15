"""AST-aware chunking.

Each function, method, class, struct, impl block, and so on becomes its own chunk, so a retrieved chunk is a
complete unit of meaning instead of an arbitrary 512-token slice. Oversized classes are split into
a header chunk plus one chunk per method; oversized functions fall back to overlapping windows.
Code between definitions (imports, module constants, top-level scripts) is grouped into "module"
chunks so nothing in the file is lost. Languages without a grammar fall back to heading-aware
(Markdown) or windowed chunking.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

from .terms import called_names

EXT_LANG = {
    ".py": "python", ".pyi": "python", ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".jsx": "javascript", ".ts": "typescript", ".mts": "typescript", ".tsx": "tsx", ".go": "go", ".rs": "rust",
    ".java": "java", ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".hh": "cpp", ".cs": "csharp", ".rb": "ruby", ".php": "php", ".kt": "kotlin", ".kts": "kotlin",
    ".swift": "swift", ".scala": "scala", ".lua": "lua", ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    # text-only (windowed / heading-aware)
    ".md": "markdown", ".mdx": "markdown", ".rst": "text", ".txt": "text", ".toml": "toml", ".yaml": "yaml",
    ".yml": "yaml", ".json": "json", ".ini": "ini", ".cfg": "ini", ".sql": "sql", ".html": "html", ".css": "css",
    ".scss": "css", ".vue": "html", ".svelte": "html", ".proto": "proto", ".gradle": "text", ".cmake": "text",
}
NAME_LANG = {"Dockerfile": "dockerfile", "Makefile": "make", "CMakeLists.txt": "text", "Justfile": "make"}

_CLS = {"class_definition", "class_declaration", "class_specifier", "struct_specifier", "interface_declaration",
        "impl_item", "trait_item", "mod_item", "namespace_definition", "namespace_declaration", "class", "module",
        "object_declaration", "struct_declaration", "enum_declaration", "record_declaration", "struct_item",
        "enum_item", "type_declaration", "abstract_class_declaration", "trait_definition", "object_definition"}

DEF_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "class_definition", "decorated_definition"},
    "javascript": {"function_declaration", "generator_function_declaration", "class_declaration",
                   "method_definition", "lexical_declaration"},
    "go": {"function_declaration", "method_declaration", "type_declaration"},
    "rust": {"function_item", "impl_item", "struct_item", "enum_item", "trait_item", "mod_item", "macro_definition"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration",
             "method_declaration", "constructor_declaration"},
    "c": {"function_definition", "struct_specifier", "enum_specifier"},
    "csharp": {"class_declaration", "interface_declaration", "struct_declaration", "enum_declaration",
               "record_declaration", "method_declaration", "constructor_declaration", "namespace_declaration"},
    "ruby": {"method", "singleton_method", "class", "module"},
    "php": {"function_definition", "class_declaration", "interface_declaration", "trait_declaration",
            "method_declaration"},
    "kotlin": {"class_declaration", "object_declaration", "function_declaration"},
    "swift": {"class_declaration", "protocol_declaration", "function_declaration"},
    "scala": {"class_definition", "object_definition", "trait_definition", "function_definition"},
    "lua": {"function_declaration"},
    "bash": {"function_definition"},
}
DEF_TYPES["typescript"] = DEF_TYPES["javascript"] | {"interface_declaration", "type_alias_declaration",
                                                     "enum_declaration", "abstract_class_declaration"}
DEF_TYPES["tsx"] = DEF_TYPES["typescript"]
DEF_TYPES["cpp"] = DEF_TYPES["c"] | {"class_specifier", "namespace_definition"}

_NAME_NODES = {"identifier", "field_identifier", "type_identifier", "property_identifier", "constant",
               "simple_identifier", "name", "qualified_identifier", "scoped_identifier", "destructor_name",
               "operator_name", "word"}


@dataclass
class Chunk:
    path: str
    lang: str
    start: int            # 1-based, inclusive
    end: int              # 1-based, inclusive
    kind: str             # function | class | module | section | window
    symbol: str | None
    text: str
    calls: set[str] = field(default_factory=set)
    inner: list[str] = field(default_factory=list)   # methods defined inside this chunk (small classes kept whole)

    def embed_text(self, limit: int = 4000) -> str:
        head = f"{self.path}" + (f" :: {self.symbol}" if self.symbol else "")
        return f"{head}\n{self.text[:limit]}"


def detect_lang(path: str) -> str | None:
    base = os.path.basename(path)
    if base in NAME_LANG:
        return NAME_LANG[base]
    return EXT_LANG.get(os.path.splitext(base)[1].lower())


@lru_cache(maxsize=None)
def _parser(lang: str):
    try:
        from tree_sitter_language_pack import get_parser
        return get_parser(lang)
    except Exception:
        return None


def _name_of(node) -> str | None:
    t = node.type
    if t == "decorated_definition":
        d = node.child_by_field_name("definition")
        return _name_of(d) if d else None
    if t == "lexical_declaration":
        for c in node.named_children:
            if c.type == "variable_declarator":
                n = c.child_by_field_name("name")
                return n.text.decode(errors="replace") if n else None
        return None
    if t == "impl_item":
        ty, tr = node.child_by_field_name("type"), node.child_by_field_name("trait")
        if ty is not None:
            ty_s = ty.text.decode(errors="replace")
            return f"{tr.text.decode(errors='replace')} for {ty_s}" if tr is not None else ty_s
    if t == "type_declaration":  # go: `type X struct {...}`
        for c in node.named_children:
            if c.type in ("type_spec", "type_alias"):
                n = c.child_by_field_name("name")
                if n is not None:
                    return n.text.decode(errors="replace")
    n = node.child_by_field_name("name")
    if n is not None:
        return n.text.decode(errors="replace").strip()
    d = node.child_by_field_name("declarator")  # C/C++ function_definition
    while d is not None:
        if d.type in _NAME_NODES:
            return d.text.decode(errors="replace")
        nxt = d.child_by_field_name("declarator")
        if nxt is None:
            for c in d.named_children:
                if c.type in _NAME_NODES:
                    return c.text.decode(errors="replace")
            break
        d = nxt
    for c in node.named_children:
        if c.type in _NAME_NODES:
            return c.text.decode(errors="replace")
    return None


def _is_def(node, lang: str) -> bool:
    if node.type not in DEF_TYPES.get(lang, ()):
        return False
    if node.type == "lexical_declaration":  # only `const foo = () => {...}` / `= function`
        return any(c.type == "variable_declarator" and (v := c.child_by_field_name("value")) is not None
                   and v.type in ("arrow_function", "function_expression", "function", "class")
                   for c in node.named_children)
    if node.type in ("struct_specifier", "enum_specifier", "class_specifier"):  # skip bare `struct foo x;` uses
        return node.child_by_field_name("body") is not None
    return True


def _kind(node) -> str:
    t = node.type
    if t == "decorated_definition":
        d = node.child_by_field_name("definition")
        t = d.type if d is not None else t
    return "class" if t in _CLS else "function"


def _windows(start: int, end: int, size: int, overlap: int):
    """Yield (a, b) inclusive 0-based line ranges covering [start, end]."""
    if end - start + 1 <= size:
        yield start, end
        return
    step = max(1, size - overlap)
    a = start
    while a <= end:
        b = min(end, a + size - 1)
        yield a, b
        if b == end:
            break
        a += step


class _Collector:
    def __init__(self, lang: str, lines: list[str], max_lines: int, win: int, overlap: int):
        self.lang, self.lines, self.max, self.win, self.ov = lang, lines, max_lines, win, overlap
        self.spans: list[tuple[int, int, str, str | None, list[str]]] = []   # 0-based inclusive

    def _nested_defs(self, node) -> list:
        out = []
        stack = list(node.children)
        while stack:
            c = stack.pop(0)
            if _is_def(c, self.lang):
                out.append(c)
            else:
                stack[:0] = list(c.children)
        return out

    def visit(self, node, prefix: str = ""):
        for child in node.children:
            if _is_def(child, self.lang):
                self._emit(child, prefix)
            elif child.child_count:
                self.visit(child, prefix)

    def _emit(self, node, prefix: str):
        name = _name_of(node)
        qual = f"{prefix}.{name}" if prefix and name else (name or prefix or None)
        s, e = node.start_point[0], node.end_point[0]
        if node.end_point[1] == 0 and e > s:
            e -= 1
        # pull in the doc comment / attributes sitting directly above (Go //, Rust ///, JSDoc, Javadoc, #[derive])
        anchor = node.parent if node.parent is not None and node.parent.type == "export_statement" else node
        prev = anchor.prev_sibling
        while prev is not None and ("comment" in prev.type or prev.type in ("attribute_item", "annotation")) \
                and prev.end_point[0] >= s - 1:
            s = prev.start_point[0]
            prev = prev.prev_sibling
        kind = _kind(node)
        if e - s + 1 <= self.max:
            inner = []
            if kind == "class":
                todo = [(n, qual or "") for n in self._nested_defs(node)]
                while todo:
                    n, pre = todo.pop()
                    nm = _name_of(n)
                    if nm:
                        q2 = f"{pre}.{nm}" if pre else nm
                        inner.append(q2)
                        todo += [(m, q2) for m in self._nested_defs(n)]
            self.spans.append((s, e, kind, qual, inner))
            return
        nested = self._nested_defs(node)
        if nested:
            first = nested[0].start_point[0]
            if first > s:  # class header: signature, docstring, fields
                hs, he = s, first - 1
                for a, b in _windows(hs, he, self.win, self.ov):
                    self.spans.append((a, b, kind, qual, []))
            for n in nested:
                self._emit(n, qual or "")
            return
        for i, (a, b) in enumerate(_windows(s, e, self.win, self.ov)):
            self.spans.append((a, b, kind, f"{qual} (part {i + 1})" if qual else None, []))


def _gap_chunks(covered: list[bool], lines: list[str], win: int, ov: int):
    i, n = 0, len(lines)
    while i < n:
        if covered[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not covered[j + 1]:
            j += 1
        if sum(1 for k in range(i, j + 1) if lines[k].strip()) >= 2:
            # trim blank edges
            a, b = i, j
            while a < b and not lines[a].strip():
                a += 1
            while b > a and not lines[b].strip():
                b -= 1
            for wa, wb in _windows(a, b, win, ov):
                yield wa, wb
        i = j + 1


def _markdown_sections(lines: list[str], win: int, ov: int):
    heads = [i for i, ln in enumerate(lines) if re.match(r"^#{1,6}\s", ln)]
    if not heads or heads[0] != 0:
        heads = [0] + heads
    for idx, s in enumerate(heads):
        e = (heads[idx + 1] - 1) if idx + 1 < len(heads) else len(lines) - 1
        title = lines[s].lstrip("#").strip() if lines[s].startswith("#") else None
        if not any(ln.strip() for ln in lines[s:e + 1]):
            continue
        for a, b in _windows(s, e, win, ov):
            yield a, b, title


def chunk_file(path: str, text: str, lang: str | None = None, *, max_lines: int = 120,
               window: int = 60, overlap: int = 10) -> list[Chunk]:
    lang = lang or detect_lang(path) or "text"
    lines = text.splitlines()
    if not lines:
        return []

    def mk(a, b, kind, sym):
        body = "\n".join(lines[a:b + 1])
        short = sym.split(" (part")[0].rsplit(".", 1)[-1] if sym else None
        return Chunk(path, lang, a + 1, b + 1, kind, sym, body, called_names(body, exclude=short))

    chunks: list[Chunk] = []
    parser = _parser(lang) if lang in DEF_TYPES else None
    if parser is not None:
        try:
            tree = parser.parse(text.encode("utf-8", errors="replace"))
            col = _Collector(lang, lines, max_lines, window, overlap)
            col.visit(tree.root_node)
            covered = [False] * len(lines)
            for a, b, kind, sym, inner in col.spans:
                b = min(b, len(lines) - 1)
                c = mk(a, b, kind, sym)
                c.inner = inner
                c.calls -= {i.rsplit(".", 1)[-1] for i in inner}   # self-defined methods aren't outgoing calls
                chunks.append(c)
                for k in range(a, b + 1):
                    covered[k] = True
            for a, b in _gap_chunks(covered, lines, window, overlap):
                chunks.append(mk(a, b, "module", None))
            chunks.sort(key=lambda c: (c.start, c.end))
            return chunks
        except Exception:
            chunks = []  # fall through to text chunking

    if lang == "markdown":
        return [mk(a, b, "section", t) for a, b, t in _markdown_sections(lines, window, overlap)]
    return [mk(a, b, "window", None) for a, b in _windows(0, len(lines) - 1, window, overlap)
            if any(ln.strip() for ln in lines[a:b + 1])]
