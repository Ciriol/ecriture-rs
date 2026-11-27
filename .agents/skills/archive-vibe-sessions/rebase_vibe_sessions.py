#!/usr/bin/env python3
"""Archive Vibe : rebase chronologique + nettoyage des secrets de l'arbre vibe/session-*.

Usage (depuis la racine du projet ecriture-rs) :
    python3 .agents/skills/archive-vibe-sessions/rebase_vibe_sessions.py           # rebase + clean
    python3 .agents/skills/archive-vibe-sessions/rebase_vibe_sessions.py --scan    # scan secrets seul

- lit chaque branche refs/heads/vibe/session-<slug>, prend son entree dans
  sessions/index.json (start_time = timestamp de la session) ;
- reecrit chaque branche en chaine lineaire ordonnee par start_time : chaque
  commit a pour parent la session precedente et des dates auteur/commiter =
  start_time de la session ;
- masque tout secret (ghp_..., github_pat_..., sk-..., AKIA..., AIza...,
  xox..., JWT) par ***REDACTED*** dans tous les blobs (push protection GH013) ;
- plumbing git uniquement : le worktree n'est jamais modifie.
"""
import subprocess, re, os, json, tempfile, sys
from datetime import datetime

REPO = os.getcwd()
PREFIX = "refs/heads/vibe/session-"
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


def git(args, env_extra=None, binary=False, input=None):
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(["git", "-C", REPO] + args, capture_output=True, env=env, input=input)
    if r.returncode != 0:
        raise RuntimeError(f"git {args}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout if binary else r.stdout.decode().strip()


def scrub(data: bytes) -> bytes:
    for rx in PATTERNS:
        data = rx.sub(REDACTED, data)
    return data


def branches():
    return [b for b in git(["for-each-ref", "--format=%(refname)", "refs/heads/vibe/"]).splitlines()
            if b.startswith(PREFIX)]


def scan():
    import hashlib
    total = 0
    seen = set()
    for ref in branches():
        for line in git(["ls-tree", "-r", ref]).splitlines():
            meta, path = line.split("\t", 1)
            sha = meta.split()[2]
            data = git(["cat-file", "blob", sha], binary=True)
            for rx in PATTERNS:
                for m in rx.finditer(data):
                    h = hashlib.sha256(m.group(0)).hexdigest()[:12]
                    key = (h, path)
                    if key not in seen:
                        seen.add(key)
                        total += 1
                        print(f"SECRET sha256={h}  ref={ref}  path={path}")
    print(f"scan: {total} secret(s) distinct(s)")
    return total


def clean_tree(tree, idx_env):
    git(["read-tree", tree], idx_env)
    changed = 0
    for line in git(["ls-tree", "-r", tree]).splitlines():
        meta, path = line.split("\t", 1)
        sha = meta.split()[2]
        data = git(["cat-file", "blob", sha], binary=True)
        clean = scrub(data)
        if clean != data:
            new = git(["hash-object", "-w", "--stdin"], idx_env, input=clean)
            git(["update-index", "--add", "--cacheinfo", f"100644,{new},{path}"], idx_env)
            changed += 1
    if changed:
        print(f"    nettoye {changed} fichier(s) dans l'arbre {tree[:8]}")
    return git(["write-tree"], idx_env)


def epoch(iso):
    return str(int(datetime.fromisoformat(iso).timestamp())) + " +0000"


def rebase():
    items = []
    for b in branches():
        slug = b[len(PREFIX):]
        idx = json.loads(git(["show", f"{b}:sessions/index.json"]))
        ent = next(e for e in idx if e.get("slug") == slug)
        items.append({"branch": b, "slug": slug, "start": ent["start"],
                      "author": git(["log", "-1", "--format=%an <%ae>", b]),
                      "subject": git(["log", "-1", "--format=%s", b]),
                      "tree": git(["rev-parse", f"{b}^{{tree}}"])})
    items.sort(key=lambda it: it["start"])
    tmpidx = tempfile.mktemp(prefix="vibe-rebase-index-")
    idx_env = {"GIT_INDEX_FILE": tmpidx}
    parent = None
    cname = git(["config", "user.name"]) or "vibe"
    cmail = git(["config", "user.email"]) or "vibe@localhost"
    for it in items:
        tree = clean_tree(it["tree"], idx_env)
        aname, amail = it["author"].split(" <")[0], it["author"].split("<")[1].rstrip(">")
        env = {"GIT_AUTHOR_NAME": aname, "GIT_AUTHOR_EMAIL": amail, "GIT_AUTHOR_DATE": epoch(it["start"]),
               "GIT_COMMITTER_NAME": cname, "GIT_COMMITTER_EMAIL": cmail, "GIT_COMMITTER_DATE": epoch(it["start"])}
        args = ["commit-tree", tree, "-m", it["subject"]]
        if parent:
            args += ["-p", parent]
        commit = git(args, env)
        git(["update-ref", it["branch"], commit])
        print(f"  {it['start']}  {it['slug']}  ->  {commit[:8]}")
        parent = commit
    os.unlink(tmpidx)
    print("rebase OK")


if __name__ == "__main__":
    if "--scan" in sys.argv:
        sys.exit(1 if scan() else 0)
    rebase()
    sys.exit(1 if scan() else 0)
