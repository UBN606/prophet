"""Prophet is a receipt book for predictions: make the call, lock it in, get graded.

  prophet                      what Prophet can do
  prophet lab call ...         make a call and lock it in (`prophet lab resolve` grades it, `prophet lab record` shows the record)
  prophet ask <question>       search the notes and transcripts you collected (a link and a snippet for each answer)
  prophet <skill> [args]       run a skill; `prophet <skill> --help` for its options
"""
import subprocess
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent
PY = sys.executable

# skill: (script, extra leading args, one-line description)
SKILLS = {
    "ask":       ("ask.py", [], "Search the transcripts, notes and findings you collected; each answer shows a link and a snippet"),
    "ub":        ("prophet_ub.py", [], "Exact Urantia Book passages with citations, read by the UB tools on your computer, plus your own notebooks"),
    "lab":       ("prophet_lab.py", [], "Make the call, lock it in, get graded: sealed predictions on anything, scored against the obvious guess"),
}


def installed():
    """The skills whose module is present next to this file. A skill with no module is neither listed nor runnable: the public
    copy ships only the general skills, and its front door must not name the ones it leaves out (project decision)."""
    return {name: spec for name, spec in SKILLS.items() if (HOME / spec[0]).exists()}


def card():
    print("PROPHET\n")
    for name, (script, _, desc) in installed().items():
        print(f"  {name:11} {desc}")
    print("\n  prophet <skill> --help   for options.")


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
