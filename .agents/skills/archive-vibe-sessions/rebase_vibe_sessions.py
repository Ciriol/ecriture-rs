#!/usr/bin/env python3
"""Archive Vibe : rebuild de l'arbre `vibe` (tronc général + branches thématiques).

Usage (depuis la racine du clone ecriture-rs) :
    python3 .agents/skills/archive-vibe-sessions/rebase_vibe_sessions.py [EXPORT_DIR]
    python3 .agents/skills/archive-vibe-sessions/rebase_vibe_sessions.py --scan

Structure construite :
  - `vibe`              : tronc général, commit racine = session la plus ancienne,
                          puis sessions "general" par start_time croissant ;
  - `vibe-interface`    : parent = tip(vibe), puis sessions "interface" par start_time ;
  - `vibe-reconnaissance`: parent = tip(vibe), puis sessions "reconnaissance" par start_time.
Un commit = une session (sessions/<slug>/{RESUME.md, journal.jsonl, meta.json,
records/}) + index global sessions/{README.md, index.json} (catégorie + branche).
Dates auteur/commiter = start_time. Secrets masqués (***REDACTED***).

Sources des sessions : branches locales vibe* + <archive-remote>/vibe* (autres
machines) + EXPORT_DIR (défaut /tmp/vibe-sessions-export, produit par
`session-export --export <id> --out EXPORT_DIR`).

Remote d'archivage : `git config vibe.archiveRemote` (défaut : `vibe-archive`
s'il existe, sinon `origin`) — dépôt PRIVÉ dédié (ex. Ciriol/vibe-sessions),
jamais le dépôt public du projet.

Plumbing git uniquement : le worktree n'est jamais modifié.
"""
import subprocess, re, os, sys, json, glob, tempfile
from datetime import datetime

REPO = os.getcwd()
EXPORT_DIR_DEFAULT = "/tmp/vibe-sessions-export"
REDACTED = b"***REDACTED***"
PATTERNS = [
    re.compile(rb"github_pat_[A-Za-z0-9_]{82}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{36,255}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{40,120}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
    re.compile(rb"xox[baprs]-[A-Za-z0-9-]{10,80}"),
    re.compile(rb"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
]

# Secrets littéraux supplémentaires, lus HORS dépôt (jamais commités) :
# ~/.vibe/archive-scrub-local.txt, un secret par ligne, chmod 600.
# Ex. : mot de passe d'une machine du LAN, partagé dans une session.
SCRUB_LOCAL_FILE = os.path.expanduser("~/.vibe/archive-scrub-local.txt")


def _extra_patterns():
    pats = []
    try:
        with open(SCRUB_LOCAL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                s = line.rstrip("\n")
                if s and not s.startswith("#"):
                    pats.append(re.compile(re.escape(s.encode())))
    except FileNotFoundError:
        pass
    return pats


PATTERNS += _extra_patterns()

# classement explicite des sessions connues ; les nouvelles sont classées par mots-clés
CATEGORIES = {
    "20261103-1127-ls": "general",
    "20261124-1719-rust-note-taking-skeleton": "general",
    "20261125-0055-crnn-handwriting-recognition": "reconnaissance",
    "20261125-1247-pylaia-model-weights-sharing": "reconnaissance",
    "20261125-1537-gui-language-model-switch": "interface",
    "20261126-1636-tablette-dessin-trace-affichage": "interface",
    "20261126-1747-session": "general",
    "20261127-0037-session": "general",
}
BRANCH_OF = {"general": "vibe", "interface": "vibe-interface", "reconnaissance": "vibe-reconnaissance"}
BRANCHES = [("vibe", "general"), ("vibe-interface", "interface"), ("vibe-reconnaissance", "reconnaissance")]


def git(args, env_extra=None, binary=False, input=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(["git", "-C", REPO] + args, capture_output=True, env=env, input=input)
    if r.returncode != 0:
        raise RuntimeError(f"git {args}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout if binary else r.stdout.decode().strip()


def _archive_remote():
    """Remote privé d'archivage : git config vibe.archiveRemote, sinon
    `vibe-archive` s'il existe, sinon `origin` (rétrocompatibilité)."""
    r = subprocess.run(["git", "-C", REPO, "config", "--get", "vibe.archiveRemote"],
                       capture_output=True)
    if r.returncode == 0 and r.stdout.decode().strip():
        return r.stdout.decode().strip()
    remotes = git(["remote"]).split()
    return "vibe-archive" if "vibe-archive" in remotes else "origin"


ARCHIVE_REMOTE = _archive_remote()


def scrub(data):
    for rx in PATTERNS:
        data = rx.sub(REDACTED, data)
    return data


def classify(slug, title):
    if slug in CATEGORIES:
        return CATEGORIES[slug]
    t = ((title or "") + " " + slug).lower()
    if any(k in t for k in ["gui", "interface", "ui", "stylet", "tablette", "souris", "hover", "trace", "affichage", "dessin", "fenetre", "fenêtre", "window"]):
        return "interface"
    if any(k in t for k in ["recon", "crnn", "ctc", "ocr", "lexique", "model", "poids", "weights", "pylaia", "handwriting", "digits", "classifieur", "classifier"]):
        return "reconnaissance"
    return "general"


def epoch(iso):
    return str(int(datetime.fromisoformat(iso).timestamp())) + " +0000"


def gen_readme(entries):
    md = ("# Sessions Vibe\n\nSessions exportées depuis `~/.vibe/logs/session/unified/` — "
          "arbre `vibe` : tronc général puis branches thématiques `vibe-interface` / "
          "`vibe-reconnaissance`, un commit par session, ordonnées par start_time.\n\n"
          "| Session | Catégorie | Période | Tokens | Branche |\n|---|---|---|---|---|\n")
    for e in entries:
        md += f"| {e['title']} | {e['category']} | {e['start'][0:10]} → {e['end'][0:10]} | {e['tokens']} | `{e['branch']}` |\n"
    return md


def collect():
    """slug -> {meta:{start,end,title,session_id,tokens}, files:{path: ('blob',sha)|('file',path)}}"""
    sessions = {}

    def add_from_tree(tree, tokens_by_slug):
        for line in git(["ls-tree", "-r", tree]).splitlines():
            meta, path = line.split("\t", 1)
            sha = meta.split()[2]
            m = re.match(r"sessions/([^/]+)/(.+)$", path)
            if not m:
                continue
            slug, rel = m.group(1), m.group(2)
            s = sessions.setdefault(slug, {"meta": {}, "files": {}})
            s["files"][path] = ("blob", sha)
            if rel == "meta.json":
                s["meta"] = json.loads(git(["cat-file", "blob", sha]))
            if rel == "index.json":
                pass
        # index.json du tip : tokens par slug
        try:
            idx = json.loads(git(["show", f"{tree}:sessions/index.json"]))
            for e in idx:
                if e.get("slug") in sessions:
                    sessions[e["slug"]]["meta"]["tokens"] = e.get("tokens", 0)
        except Exception:
            pass

    # branches locales vibe* puis distantes <archive-remote>/vibe* (autres machines)
    refs = [b for b in git(["for-each-ref", "--format=%(refname)", "refs/heads/vibe*"]).splitlines()]
    refs += [b for b in git(["for-each-ref", "--format=%(refname)", f"refs/remotes/{ARCHIVE_REMOTE}/vibe*"]).splitlines()]
    for b in refs:
        if b.endswith("/session-") or "/session-" in b:
            continue
        try:
            add_from_tree(git(["rev-parse", f"{b}^{{tree}}"]), None)
        except Exception:
            pass

    # exports temporaires (nouvelles sessions de cette machine)
    exp = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else EXPORT_DIR_DEFAULT
    if os.path.isdir(exp):
        for slug in sorted(os.listdir(exp)):
            d = os.path.join(exp, slug)
            if not os.path.isdir(d):
                continue
            s = sessions.setdefault(slug, {"meta": {}, "files": {}})
            mj = os.path.join(d, "meta.json")
            if os.path.isfile(mj):
                s["meta"] = json.load(open(mj))
            for root, _, names in os.walk(d):
                for n in names:
                    p = os.path.join(root, n)
                    rel = os.path.join("sessions", os.path.relpath(p, exp))
                    s["files"][rel] = ("file", p)
    return sessions


def build(sessions):
    groups = {"general": [], "interface": [], "reconnaissance": []}
    for slug, s in sessions.items():
        meta = s["meta"]
        s["category"] = classify(slug, meta.get("title", ""))
        s["start"] = meta.get("start_time", "")
        groups[s["category"]].append(slug)
    for g in groups.values():
        g.sort(key=lambda sl: sessions[sl]["start"])

    all_entries = []
    for slug, s in sessions.items():
        m = s["meta"]
        all_entries.append({"slug": slug, "session_id": m.get("session_id", ""),
                            "title": m.get("title") or "", "start": m.get("start_time", ""),
                            "end": m.get("end_time", ""), "tokens": m.get("tokens", 0),
                            "branch": BRANCH_OF[s["category"]], "category": s["category"], "raw": True})
    all_entries.sort(key=lambda e: e["start"])
    index_json = json.dumps(all_entries, indent=2, ensure_ascii=False)
    readme = gen_readme(all_entries)

    tmpidx = tempfile.mktemp(prefix="vibe-tree-index-")
    idx_env = {"GIT_INDEX_FILE": tmpidx}
    cname = git(["config", "user.name"]) or "vibe"
    cmail = git(["config", "user.email"]) or "vibe@localhost"

    def add_file(path, src):
        if src[0] == "blob":
            sha = src[1]
            data = git(["cat-file", "blob", sha], binary=True)
            clean = scrub(data)
            if clean != data:
                sha = git(["hash-object", "-w", "--stdin"], idx_env, input=clean)
        else:
            data = open(src[1], "rb").read()
            clean = scrub(data)
            sha = git(["hash-object", "-w", "--stdin"], idx_env, input=clean)
        git(["update-index", "--add", "--cacheinfo", f"100644,{sha},{path}"], idx_env)

    built = {}
    for branch, cat in BRANCHES:
        if branch == "vibe":
            git(["read-tree", "--empty"], idx_env)
            parent = None
        else:
            git(["read-tree", built["vibe_tree"]], idx_env)
            parent = built["vibe"]
        for slug in groups[cat]:
            s = sessions[slug]
            for path, src in sorted(s["files"].items()):
                add_file(path, src)
            bi = git(["hash-object", "-w", "--stdin"], idx_env, input=index_json.encode())
            git(["update-index", "--add", "--cacheinfo", f"100644,{bi},sessions/index.json"], idx_env)
            br = git(["hash-object", "-w", "--stdin"], idx_env, input=readme.encode())
            git(["update-index", "--add", "--cacheinfo", f"100644,{br},sessions/README.md"], idx_env)
            tree = git(["write-tree"], idx_env)
            date = epoch(s["start"]) if s["start"] else "0 +0000"
            env = {"GIT_AUTHOR_NAME": cname, "GIT_AUTHOR_EMAIL": cmail, "GIT_AUTHOR_DATE": date,
                   "GIT_COMMITTER_NAME": cname, "GIT_COMMITTER_EMAIL": cmail, "GIT_COMMITTER_DATE": date}
            title = s["meta"].get("title") or slug
            args = ["commit-tree", tree, "-m", f"session {slug} — {title} [{cat}] (arbre vibe, sans code)"]
            if parent:
                args += ["-p", parent]
            commit = git(args, env)
            git(["update-ref", f"refs/heads/{branch}", commit])
            print(f"  {branch}: {s['start'][:16]}  {slug}  ->  {commit[:8]}")
            parent = commit
        built[branch] = parent
        built[branch + "_tree"] = git(["rev-parse", f"refs/heads/{branch}^{{tree}}"])
    os.unlink(tmpidx)


def scan():
    import hashlib
    total = 0
    seen = set()
    refs = [b for b in git(["for-each-ref", "--format=%(refname)", "refs/heads/vibe*"]).splitlines()]
    for ref in refs:
        for line in git(["ls-tree", "-r", ref]).splitlines():
            meta, path = line.split("\t", 1)
            sha = meta.split()[2]
            data = git(["cat-file", "blob", sha], binary=True)
            for rx in PATTERNS:
                for m in rx.finditer(data):
                    h = hashlib.sha256(m.group(0)).hexdigest()[:12]
                    if (h, path) not in seen:
                        seen.add((h, path))
                        total += 1
                        print(f"SECRET sha256={h}  ref={ref}  path={path}")
    print(f"scan: {total} secret(s) distinct(s)")
    return total


if __name__ == "__main__":
    if "--scan" in sys.argv:
        sys.exit(1 if scan() else 0)
    sessions = collect()
    print(f"{len(sessions)} session(s) connue(s)")
    build(sessions)
    print("rebuild OK")
    sys.exit(1 if scan() else 0)
