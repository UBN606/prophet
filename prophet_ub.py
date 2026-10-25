"""Prophet UB: Urantia Book passages with exact Paper:Section.Paragraph citations, plus your own notebooks on request.

  prophet ub prayer                        top passages for a word or phrase
  prophet ub --ref 180:2.1 180:2.2         the exact paragraph text for a citation
  prophet ub prayer --with-notebooks       the same words also asked of your notebooks (ask)
  prophet ub prayer --json                 machine-readable (for other tools and agents)

All book text comes from the local book search tool (ub-search.js, run with node). Nothing is quoted from memory,
nothing is paraphrased, no network, no model, no GPU. Search results are re-read with --ref so every passage is the
full, exact paragraph, never an excerpt.
"""
import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import prophet_config as pcfg

HOME = Path(__file__).resolve().parent
UB_DIR = pcfg.UB_TOOLS
UB_SEARCH = UB_DIR / "ub-search.js"
UB_VERIFY = UB_DIR / "ub-verify.js"

REF_RE = re.compile(r"^\d{1,3}:\d{1,2}\.\d{1,3}$")
FOUND_RE = re.compile(r"^Found (\d+) results\.")
HIT_RE = re.compile(r"^--- \((\d+:\d+\.\d+)\) Paper (\d+): (.+?) ---\s*$")
# words that make a typed question a sentence rather than a search phrase; dropped only when the question starts with one
QUESTION_START = {"what", "how", "why", "when", "where", "who", "which", "does", "do", "did", "is", "are", "tell",
                  "explain", "describe", "show", "find", "give", "can", "could", "should"}
FILLER = QUESTION_START | {"me", "us", "say", "says", "said", "about", "the", "a", "an", "book", "urantia", "to", "of",
                           "in", "on", "for", "and", "or", "it", "its", "this", "that", "mean", "means", "meant", "regarding"}


class UbError(RuntimeError):
    """The book tool could not run or answered with an error. Never treated as 'nothing found'."""


def find_node():
    found = shutil.which("node") or shutil.which("node.exe")
    if found:
        return found
    program_files = os.environ.get("ProgramFiles")
    fallback = Path(program_files) / "nodejs" / "node.exe" if program_files else None
    if fallback is not None and fallback.exists():
        return str(fallback)
    raise UbError("node was not found, so the book tools cannot run")


def run_node(args, stdin_text=None, timeout=180):
    """Run node with the given arguments. Returns (returncode, stdout, stderr), decoded as UTF-8."""
    try:
        done = subprocess.run([find_node(), *args], input=(stdin_text.encode("utf-8") if stdin_text is not None else None),
                              capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise UbError("the book tool did not answer within " + str(timeout) + " seconds")
    return done.returncode, done.stdout.decode("utf-8", errors="replace"), done.stderr.decode("utf-8", errors="replace")


def tool_files_ok():
    for tool in (UB_SEARCH, UB_VERIFY):
        if not tool.exists():
            raise UbError("book tool missing: " + str(tool))


def parse_ref_output(text):
    """Parse the output of `ub-search.js --ref ...` into one dict per citation (exact text, never trimmed)."""
    blocks, cur = [], None
    for line in text.splitlines():
        if line.startswith("### "):
            if cur:
                blocks.append(cur)
            cur = {"head": line[4:].strip(), "lines": []}
        elif cur is not None:
            cur["lines"].append(line)
    if cur:
        blocks.append(cur)
    out = []
    for block in blocks:
        while block["lines"] and not block["lines"][-1].strip():
            block["lines"].pop()
        body = "\n".join(block["lines"])
        head = block["head"]
        ref = head.split()[0]
        if "[" not in head:
            out.append({"ref": ref, "error": body.strip("*\n ") or "Reference not found"})
            continue
        inner = head[head.index("[") + 1: head.rindex("]")]
        parts = inner.split(" | ")
        if len(parts) < 4 or not parts[0].startswith("page "):
            out.append({"ref": ref, "error": "could not read the tool's header: " + head})
            continue
        paper = re.match(r"Paper (\d+): (.*)$", parts[1])
        out.append({"ref": ref, "page": parts[0][5:], "paper": int(paper.group(1)) if paper else None,
                    "paper_title": paper.group(2) if paper else parts[1], "section": " | ".join(parts[2:-1]),
                    "author": parts[-1], "text": body})
    return out


def ub_get(refs):
    """Exact paragraph text for citations, from `ub-search.js --ref`. Missing citations come back with an 'error' key."""
    tool_files_ok()
    refs = list(dict.fromkeys(refs))
    if not refs:
        return []
    bad = [r for r in refs if not REF_RE.match(r)]
    if bad:
        raise UbError("not a Paper:Section.Paragraph citation: " + ", ".join(bad))
    code, out, err = run_node([str(UB_SEARCH), "--ref", *refs])
    parsed = parse_ref_output(out)
    if code not in (0, 1) or (not parsed and out.strip() == ""):   # exit 1 only means some citation was not found
        raise UbError("ub-search.js --ref failed (exit " + str(code) + "): " + (err.strip() or out.strip())[:300])
    return parsed


def ub_search(query, limit=5, paper=None):
    """Search the book. Returns (total found, [dict per passage with exact full text])."""
    tool_files_ok()
    args = [str(UB_SEARCH), query, "--limit", str(limit)]
    if paper is not None:
        args += ["--paper", str(paper)]
    code, out, err = run_node(args)
    if code != 0:
        raise UbError("ub-search.js failed (exit " + str(code) + "): " + (err.strip() or out.strip())[:300])
    total, refs = 0, []
    for line in out.splitlines():
        m = FOUND_RE.match(line)
        if m:
            total = int(m.group(1))
        h = HIT_RE.match(line)
        if h:
            refs.append(h.group(1))
    if not refs:
        return total, []
    return total, ub_get(refs)


# Used to rank candidate paragraphs for many text windows in ONE node run (the tool's own search
# function, loaded as a library). Only rankings come from here; the words always come from ub_get().
RANK_JS = (
    "const s=require('" + UB_SEARCH.as_posix() + "');"
    "let d='';process.stdin.on('data',c=>d+=c).on('end',()=>{"
    "const qs=JSON.parse(d);const out=qs.map(q=>{const r=s.searchUB({query:q,limit:6});"
    "return r.results.map(x=>({ref:x.ref,score:x.score}));});"
    "process.stdout.write(JSON.stringify(out));});"
)


def ub_rank(queries):
    tool_files_ok()
    code, out, err = run_node(["-e", RANK_JS], stdin_text=json.dumps(queries))
    if code != 0:
        raise UbError("ub-search.js (library) failed (exit " + str(code) + "): " + (err.strip() or out.strip())[:300])
    try:
        return json.loads(out)
    except ValueError:
        raise UbError("ub-search.js (library) returned something that is not JSON: " + out[:200])


def clean_question(question):
    """A typed sentence ('what does the book say about prayer') becomes its search words. Phrases pass through."""
    words = question.split()
    if len(words) >= 3 and words[0].lower().strip("?,.") in QUESTION_START:
        kept = [w for w in words if w.lower().strip("?,.!") not in FILLER]
        if kept:
            return " ".join(w.strip("?,.!") for w in kept)
    return question


def load_ask():
    spec = importlib.util.spec_from_file_location("ask_module", HOME / "ask.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def notebook_hits(query, n):
    """The same words asked of everything Prophet remembers (channel transcripts, notes, findings), via ask in-process."""
    ask = load_ask()
    index = ask.load()
    out = []
    for r in ask.search(index, query, None, n):
        out.append({"channel": r["channel"], "title": r["title"], "kind": r["kind"], "date": r["date"],
                    "at": ask.clock(r["start"]), "link": r["link"], "score": r["score"],
                    "snippet": r["snippet"].replace("\n", " ")})
    return out


def show_passage(p):
    print(p["ref"] + "  Paper " + str(p["paper"]) + ": " + p["paper_title"] + " | " + p["section"] + " | page " + p["page"])
    print("  " + p["text"])
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="prophet ub", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question", nargs="*", help="a word, a phrase or a question (a citation like 180:2.1 is read as --ref)")
    parser.add_argument("--ref", nargs="+", metavar="P:S.P", help="exact text for these citations, e.g. 180:2.1")
    parser.add_argument("--limit", "-n", type=int, default=5, help="how many passages (default 5)")
    parser.add_argument("--paper", type=int, help="search only this paper number")
    parser.add_argument("--with-notebooks", action="store_true", help="also ask your notebooks (harvested sources, transcripts, notes) the same words")
    parser.add_argument("--notebooks-n", type=int, default=5, help="how many notebook results with --with-notebooks (default 5)")
    parser.add_argument("--json", action="store_true", help="JSON output for other tools and agents")
    args = parser.parse_args(argv)

    refs = list(args.ref or [])
    words = []
    for w in args.question:
        (refs if REF_RE.match(w) else words).append(w)
    question = " ".join(words).strip()
    if not refs and not question:
        parser.print_usage()
        print('Example: prophet ub prayer     or     prophet ub --ref 180:2.1')
        return 2

    report = {"question": question or None, "tool": "node ub-search.js (local, exact text)", "refs": refs, "ref_passages": [],
              "search": None, "notebooks": None, "errors": []}
    code = 0
    try:
        if refs:
            report["ref_passages"] = ub_get(refs)
            if any("error" in p for p in report["ref_passages"]):
                code = 1
        if question:
            used = clean_question(question)
            total, found = ub_search(used, args.limit, args.paper)
            report["search"] = {"searched": used, "total_found": total, "shown": len(found), "passages": found}
        if args.with_notebooks and (question or refs):
            report["notebooks"] = notebook_hits(question or " ".join(refs), args.notebooks_n)
    except UbError as error:
        report["errors"].append(str(error))
        code = 1
    except Exception as error:   # a check that cannot run is a failure, said plainly
        report["errors"].append("unexpected failure: " + type(error).__name__ + ": " + str(error))
        code = 1

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return code

    for error in report["errors"]:
        print("FAILED: " + error)
    if refs and report["ref_passages"]:
        print("Exact text for " + ", ".join(refs) + " (from ub-search.js --ref)\n")
        for p in report["ref_passages"]:
            if "error" in p:
                print(p["ref"] + "  NOT FOUND: " + p["error"] + "\n")
            else:
                show_passage(p)
    if report["search"] is not None:
        s = report["search"]
        shown = "searched: " + s["searched"] if s["searched"] != question else "exact words"
        print('UB passages for "' + question + '" (' + str(s["shown"]) + " of " + str(s["total_found"]) + " found, " + shown + ")\n")
        if not s["passages"]:
            print("No passages found. Try fewer or different words.\n")
        for p in s["passages"]:
            if "error" in p:
                print(p["ref"] + "  NOT FOUND: " + p["error"] + "\n")
            else:
                show_passage(p)
    if report["notebooks"] is not None:
        print('Notebook results for "' + (question or " ".join(refs)) + '" (ask)\n')
        if not report["notebooks"]:
            print("Nothing in your notebooks for those words.\n")
        for i, r in enumerate(report["notebooks"], 1):
            at = (" @ " + r["at"]) if r["at"] else ""
            print(str(i) + ". " + r["channel"] + " | " + r["title"] + at)
            print("   " + r["link"])
            print("   " + r["snippet"])
            print()
    return code


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
