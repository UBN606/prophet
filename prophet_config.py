"""Prophet configuration: the one place that says where Prophet keeps its data and where it looks for your material.

Nothing is wired to a machine. Every folder defaults to a place under one data folder.

  data folder   environment variable PROPHET_HOME, else `home` in prophet.toml, else ~/prophet-data
  config file   environment variable PROPHET_CONFIG, else prophet.toml next to this file,
                else <data folder>/prophet.toml   (copy prophet.example.toml and edit)

Any key under [paths] or [tools] can also be set by an environment variable named PROPHET_<KEY>
(PROPHET_NOTEBOOKS, PROPHET_LOCAL, ...). A list uses the operating system's path separator.
A relative path is taken from the data folder. `~` and environment variables are expanded.

  python prophet_config.py        print what Prophet resolved, and whether each folder exists
"""
import os
import shutil
import sys
from pathlib import Path

if sys.version_info < (3, 11):
    raise SystemExit("Prophet needs Python 3.11 or newer (this is Python %s)." % sys.version.split()[0])

CODE = Path(__file__).resolve().parent


def _find_config():
    named = os.environ.get("PROPHET_CONFIG", "").strip()
    if named:
        return Path(os.path.expandvars(named)).expanduser()
    beside = CODE / "prophet.toml"
    if beside.is_file():
        return beside
    home = os.environ.get("PROPHET_HOME", "").strip()
    base = Path(os.path.expandvars(home)).expanduser() if home else Path.home() / "prophet-data"
    inside = base / "prophet.toml"
    return inside if inside.is_file() else None


def _load(path):
    if path is None:
        return {}
    try:
        import tomllib
    except ModuleNotFoundError:
        raise SystemExit("Prophet needs Python 3.11 or newer to read %s (this is Python %s)." % (path, sys.version.split()[0]))
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except (OSError, ValueError) as error:
        raise SystemExit("Cannot read the Prophet config %s: %s" % (path, error))


CONFIG_FILE = _find_config()
_CFG = _load(CONFIG_FILE)


def _setting(section, key):
    env = os.environ.get("PROPHET_" + key.upper(), "").strip()
    if env:
        return env
    value = (_CFG.get(section) or {}).get(key)
    if isinstance(value, str):
        return value.strip() or None
    return value or None


def _expand(text, base):
    path = Path(os.path.expandvars(str(text))).expanduser()
    return path if path.is_absolute() else Path(os.path.abspath(base / path))


def _data_folder():
    env = os.environ.get("PROPHET_HOME", "").strip()
    if env:
        return _expand(env, Path.cwd())
    configured = str(_CFG.get("home") or "").strip()
    if configured:
        return _expand(configured, CONFIG_FILE.parent)
    return _expand("~/prophet-data", Path.cwd())


DATA = _data_folder()


def _path(key, default):
    value = _setting("paths", key)
    return _expand(value, DATA) if isinstance(value, str) else DATA / default


def _paths(key):
    value = _setting("paths", key)
    if not value:
        return []
    items = value.split(os.pathsep) if isinstance(value, str) else list(value)
    return [_expand(item, DATA) for item in items if str(item).strip()]


NOTEBOOKS = _path("notebooks", "notebooks")            # harvested text, notes, every skill's output and the search index (`ask` reads it)
LOCAL = _path("local", "local")                        # Markdown transcripts of your own media (`ask` searches them)
FILMS = _path("films", "films")                        # one folder per project; `ask` also searches each project's _scratch/search folder
UB_TOOLS = _path("ub_tools", "ub-tools")               # folder with ub-search.js and ub-verify.js (`prophet ub`; the book text is not part of this repo)
GODS_EYE_VIEW = _path("gods_eye_view", "repos/gods-eye-view") # optional local clone indexed by source_harvest.py


def tool(name):
    """Command for an external program: [tools] <name> (or PROPHET_<NAME>), else the first one on PATH, else the bare name."""
    value = _setting("tools", name.lower().replace("-", "_"))
    if isinstance(value, str):
        return value if (os.sep not in value and "/" not in value) else str(_expand(value, DATA))
    return shutil.which(name) or name


def describe():
    rows = [("config file", CONFIG_FILE or "(none: defaults)"), ("data folder", DATA)]
    for label, value in [("notebooks", NOTEBOOKS), ("local", LOCAL), ("films", FILMS), ("ub_tools", UB_TOOLS), ("gods_eye_view", GODS_EYE_VIEW)]:
        rows.append((label, "%s%s" % (value, "" if Path(value).exists() else "   (not there yet)")))
    return rows


if __name__ == "__main__":
    for label, value in describe():
        print("%-14s %s" % (label, value))
