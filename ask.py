"""Search everything Prophet has harvested or transcribed: answers with channel, video, timestamp and a link
that opens the video at that moment. Local files only; no network, no model, no GPU.

  ask camera moves for game animation
  ask --channel muska blender mcp
  ask -n 20 seedance prompt
  ask --json "kling vs seedance"      (machine-readable, for other agents)
  ask --stats
"""
import argparse
import json
import math
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import prophet_config as pcfg

KNOWLEDGE = pcfg.NOTEBOOKS
FILM_SEARCH = "*/_scratch/search"          # inside the films folder (prophet.toml [paths] films)
LOCAL = pcfg.LOCAL
INDEX = KNOWLEDGE / ".search-index.pickle"
VERSION = 4
STOP = set("""a an and are as at be but by for from has have he her his i if in into is it its just like me my no not of on
or our so that the their them then there they this to up us was we were what when which who will with you your yeah okay
um uh really very going gonna get got do does did can could would should one also all about out now here how than""".split())
WORD = re.compile(r"[a-z0-9][a-z0-9.+#-]*[a-z0-9+#]|[a-z0-9]")
NOTE_LABEL = "Prophet note:"            # passages that are Prophet's own notes about a source start with this; they are not quotes


def words(text):
    return [w for w in WORD.findall(text.lower()) if w not in STOP]


def parse(file):
    raw = file.read_text(encoding="utf-8", errors="replace")
    meta, body = {}, raw
    if raw.startswith("---\n"):
        head, _, body = raw[4:].partition("\n---\n")
        for line in head.splitlines():
            key, _, value = line.partition(": ")
            value = value.strip()
            if value.startswith('"'):
                try:
                    value = json.loads(value)
                except ValueError:
                    value = value.strip('"')
            meta[key.strip()] = value
    return meta, body


def chunks(meta, body, file):
    """About a minute of speech per chunk; each chunk keeps its [m:ss] marks so a hit points at the exact segment."""
    parts = re.split(r"\n?\[(\d+):(\d\d)\] ", "\n" + body)
    if len(parts) > 1:
        segments = [(int(parts[i]) * 60 + int(parts[i + 1]), parts[i + 2].strip()) for i in range(1, len(parts) - 2, 3)]
        group = []
        for seg in segments + [None]:
            if group and (seg is None or seg[0] - group[0][0] >= 50):
                marks, text = [], ""
                for start, said in group:
                    marks.append((start, len(text)))
                    text += said + " "
                yield group[0][0], text.strip(), marks
                group = []
            if seg:
                group.append(seg)
    else:
        for para in [p.strip() for p in body.split("\n\n") if p.strip()]:
            yield None, para, []


def sources():
    """Channel transcripts and notebook answers, film transcripts, and other local folders."""
    found = list(KNOWLEDGE.glob("*/**/*.md")) + list(LOCAL.glob("*/*.md"))
    found += [f for d in pcfg.FILMS.glob(FILM_SEARCH) for f in d.glob("*.md")]
    return sorted(f for f in found if f.name != "INDEX.md" and ".prev-" not in f.name)


def build():
    files = sources()
    docs, postings, lengths = [], defaultdict(list), []
    for file in files:
        meta, body = parse(file)
        title = meta.get("title", file.stem)
        url = meta.get("url") or meta.get("drive_url", "")
        channel = meta.get("channel") or meta.get("notebook") or file.parent.parent.name
        kind = "video" if "/videos/" in file.as_posix() else ("local media" if meta.get("media_sha256") else "notebook answer")
        title_words = words(title)
        for start, text, marks in chunks(meta, body, file):
            terms = words(text) + title_words
            if not terms:
                continue
            cid = len(docs)
            docs.append({"channel": channel, "title": title, "url": url, "start": start, "text": text, "marks": marks,
                         "file": str(file), "date": meta.get("upload_date") or meta.get("drive_modified", "")[:10], "kind": kind})
            lengths.append(len(terms))
            for term, tf in Counter(terms).items():
                postings[term].append((cid, tf))
    stamp = {str(f): f.stat().st_mtime for f in files}
    index = {"version": VERSION, "stamp": stamp, "docs": docs, "postings": dict(postings), "lengths": lengths,
             "avg": (sum(lengths) / len(lengths)) if lengths else 1}
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_bytes(pickle.dumps(index, protocol=pickle.HIGHEST_PROTOCOL))
    return index


def load():
    files = {str(f): f.stat().st_mtime for f in sources()}
    if INDEX.exists():
        try:
            index = pickle.loads(INDEX.read_bytes())
            if index.get("version") == VERSION and index["stamp"] == files:
                return index
        except Exception:
            pass
    return build()


def link(doc):
    if doc["start"] is not None and "youtube.com/watch" in doc["url"]:
        return doc["url"] + "&t=" + str(doc["start"]) + "s"
    if doc["kind"] == "local media" and doc["start"] is not None:
        return doc["url"] + " @ " + clock(doc["start"])
    return doc["url"]


def clock(seconds):
    return "" if seconds is None else str(seconds // 60) + ":" + format(seconds % 60, "02d")


def snippet(text, terms, width=260):
    low = text.lower()
    hits = [low.find(t) for t in terms if low.find(t) >= 0]
    start = max(0, min(hits) - 60) if hits else 0
    if text.startswith(NOTE_LABEL):         # Prophet's own record about a source (short): always shown from its label, never from the middle
        start = 0
    cut = text[start:start + width].strip()
    return ("..." if start else "") + cut + ("..." if start + width < len(text) else "")


def best_mark(doc, terms, phrase):
    """Time code of the segment that holds the whole phrase, else the most query words (earliest on a tie)."""
    text, marks = doc["text"].lower(), doc["marks"]
    spans = [(start, text[offset:(marks[i + 1][1] if i + 1 < len(marks) else len(text))]) for i, (start, offset) in enumerate(marks)]
    if len(terms) > 1:
        at = text.find(phrase)
        if at >= 0:
            return max((s for s, offset in marks if offset <= at), default=doc["start"])
    best = max(spans, key=lambda span: (sum(1 for t in terms if t in span[1]), -span[0]))
    return best[0]


def search(index, query, channel=None, limit=8, per_video=2):
    terms = list(dict.fromkeys(words(query)))
    if not terms:
        return []
    n, avg, k1, b = len(index["docs"]), index["avg"], 1.4, 0.75
    scores = defaultdict(float)
    for term in terms:
        plist = index["postings"].get(term, [])
        if not plist:
            continue
        idf = math.log(1 + (n - len(plist) + 0.5) / (len(plist) + 0.5))
        for cid, tf in plist:
            length = index["lengths"][cid]
            scores[cid] += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg))
    phrase = query.lower().strip('"')
    for cid in list(scores):
        doc = index["docs"][cid]
        if channel and channel.lower() not in doc["channel"].lower():
            del scores[cid]
            continue
        matched = sum(1 for t in terms if t in doc["text"].lower())
        scores[cid] *= 1 + matched / len(terms)          # reward chunks that cover more of the question
        if len(terms) > 1 and phrase in doc["text"].lower():
            scores[cid] *= 1.5
    results, seen = [], Counter()
    for cid, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        doc = index["docs"][cid]
        if seen[doc["url"]] >= per_video:
            continue
        seen[doc["url"]] += 1
        hit = dict(doc, score=round(score, 2))
        if doc.get("marks"):
            hit["start"] = best_mark(doc, terms, phrase)
        hit["link"], hit["snippet"] = link(hit), snippet(doc["text"], terms)
        results.append(hit)
        if len(results) >= limit:
            break
    return results


def stats(index):
    by = defaultdict(set)
    for d in index["docs"]:
        by[d["channel"]].add(d["url"])
    print(str(sum(len(v) for v in by.values())) + " sources, " + str(len(index["docs"])) + " searchable passages")
    for ch, urls in sorted(by.items()):
        print("  " + ch + ": " + str(len(urls)))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ask", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="*")
    parser.add_argument("--channel", "-c", help="only this channel (part of the name is enough)")
    parser.add_argument("-n", type=int, default=8, help="how many results (default 8)")
    parser.add_argument("--json", action="store_true", help="JSON output for other tools and agents")
    parser.add_argument("--stats", action="store_true", help="what is searchable right now")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the search index")
    args = parser.parse_args(argv)
    index = build() if args.rebuild else load()
    if args.stats or not args.query:
        stats(index)
        if not args.query:
            print('\nUsage: ask <question>    e.g.  ask camera moves for game animation')
        return 0
    query = " ".join(args.query)
    results = search(index, query, args.channel, args.n)
    if args.json:
        print(json.dumps({"query": query, "results": [{k: r[k] for k in ("channel", "title", "date", "kind", "start", "link", "score", "snippet", "file")} for r in results]}, indent=2, ensure_ascii=False))
        return 0
    if not results:
        print('Nothing found for "' + query + '". Try fewer or different words, or: ask --stats'); return 0
    for i, r in enumerate(results, 1):
        when = (" @ " + clock(r["start"])) if r["start"] is not None else ""
        print(str(i) + ". " + r["channel"] + " | " + r["title"] + when)
        print("   " + r["link"])
        print("   " + r["snippet"].replace("\n", " "))
        print()
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
