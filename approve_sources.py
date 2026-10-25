"""Approve the source list before the harvester touches the network.

  python approve_sources.py            show the list by area, then ask for a yes
  python approve_sources.py --yes      record the approval without asking
  python approve_sources.py --status   say whether the current list is approved (exit 0 yes, 1 no)

source_harvest.py refuses to start unless the approval file holds the SHA-256 of the sources/sources.json it is about to read and
says "approved" (a record without a status does not count). Change the list and you must approve it again. The approval is one
small file in your data folder; nothing is sent anywhere.
"""
import argparse
import collections
import getpass
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prophet_config as pcfg  # noqa: E402

REGISTRY = HERE / "sources" / "sources.json"
APPROVAL = pcfg.NOTEBOOKS / "sources" / "APPROVAL.json"


def digest():
    return hashlib.sha256(REGISTRY.read_bytes()).hexdigest()


def recorded():
    try:
        return json.loads(APPROVAL.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(prog="approve_sources", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="record the approval without asking")
    ap.add_argument("--status", action="store_true", help="say whether the current list is approved, then exit")
    args = ap.parse_args(argv)
    if not REGISTRY.exists():
        print("No source list at %s" % REGISTRY, file=sys.stderr)
        return 2
    sha = digest()
    old = recorded()
    approved_now = bool(old) and old.get("sourcesSha256") == sha and old.get("status") == "approved"
    if args.status:
        print("approved: the current list matches %s" % APPROVAL if approved_now else "NOT approved: %s" % (
            "the record says %r, not 'approved'" % old.get("status") if old and old.get("sourcesSha256") == sha else
            "the list changed since the approval" if old else "no approval recorded"))
        return 0 if approved_now else 1
    sources = json.loads(REGISTRY.read_text(encoding="utf-8"))["sources"]
    verified = [s for s in sources if s.get("verified") is True]
    disabled = [s for s in verified if s.get("disabled")]
    areas = collections.OrderedDict()
    for s in sources:
        row = areas.setdefault(s["area"], [0, 0])
        row[0] += 1
        row[1] += 1 if s.get("verified") is True and not s.get("disabled") else 0
    print("Source list %s  (sha256 %s...)" % (REGISTRY, sha[:12]))
    for area, (total, ok) in areas.items():
        print("  %-30s %3d sources, %3d fetched" % (area, total, ok))
    print("%d sources, %d verified, %d of those disabled. Only verified, enabled ones are fetched; none needs a key or an account." % (
        len(sources), len(verified), len(disabled)))
    for s in disabled:
        print("  disabled, kept in the list, never fetched: %s (%s)" % (s["id"], s.get("disabled_reason", "no reason recorded")))
    if approved_now:
        print("Already approved: %s" % APPROVAL)
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("Not a terminal: read the list, then rerun with --yes.", file=sys.stderr)
            return 2
        if input("Type yes to approve fetching the verified, enabled sources: ").strip().lower() != "yes":
            print("Not approved. Nothing recorded.")
            return 1
    try:
        who = getpass.getuser()
    except Exception:
        who = "owner"
    record = {"status": "approved", "at": datetime.now(timezone.utc).strftime("%Y-%m-%d"), "by": who, "answer": "approved the verified sources",
              "scope": "Fetch only the sources in sources/sources.json that are marked verified and not disabled: public, no key, no account.",
              "sourcesSha256": sha}
    APPROVAL.parent.mkdir(parents=True, exist_ok=True)
    APPROVAL.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("Recorded: %s" % APPROVAL)
    return 0


if __name__ == "__main__":
    sys.exit(main())
