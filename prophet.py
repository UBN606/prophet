"""Prophet: finds the exact moment in what you have collected, and calls what comes next with an honest track record.

  prophet                      what Prophet can do, and what is proven
  prophet ask <question>       search everything Prophet knows (time-coded links)
  prophet <skill> [args]       run a skill; `prophet <skill> --help` for its options
"""
import json
import subprocess
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent
PY = sys.executable
STATUS = HOME / "prophet_status.json"

# skill: (script, extra leading args, one-line description)
SKILLS = {
    "ask":       ("ask.py", [], "Search every transcript, note and finding; answers carry a time code or path:line"),
    "ub":        ("prophet_ub.py", [], "Urantia Book passages with exact citations from a local book tool you provide, plus your own notebooks"),
    "lab":       ("prophet_lab.py", [], "Pattern Lab: sealed calls on sports, weather, markets, anything; scored against a baseline"),
}


def installed():
    """The skills whose module is present next to this file. A skill with no module is neither listed nor runnable: the public
    copy ships only the general skills, and its front door must not name the ones it leaves out (project decision)."""
    return {name: spec for name, spec in SKILLS.items() if (HOME / spec[0]).exists()}


def card():
    status = {}
    if STATUS.exists():
        try:
            status = json.loads(STATUS.read_text(encoding="utf-8"))
        except ValueError:
            pass
    print("PROPHET\n")
    for name, (script, _, desc) in installed().items():
        state = status.get(name, {}).get("state", "proven" if name in ("ask",) else "building")
        print(f"  {name:11} {state:13} {desc}")
    print("\n  prophet <skill> --help   for options.  Machine flags are worth a look, never an approval.")


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        card()
        return 0
    name = args[0]
    skills = installed()
    if name not in skills:
        print(f"Unknown skill '{name}'. Skills: {', '.join(skills)}")
        return 2
    script, lead, _ = skills[name]
    return subprocess.call([PY, "-B", str(HOME / script), *lead, *args[1:]])


if __name__ == "__main__":
    sys.exit(main())
