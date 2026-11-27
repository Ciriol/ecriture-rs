#!/bin/sh
# Hook session_end (projet .vibe/hooks.toml) : archive les sessions Vibe
# journalisées EN RAPPORT AVEC CE PROJET dans l'arbre `vibe` (tronc général
# + branches vibe-interface / vibe-reconnaissance, un commit par session,
# ordonnées par start_time), secrets masqués, puis push --force-with-lease.
# L'export passe par un dossier temporaire (--out), jamais par des branches
# vibe/session-* (git interdit refs/heads/vibe + refs/heads/vibe/...).
# Plumbing git uniquement : le worktree n'est jamais modifié. Portable :
# fonctionne dans tout clone du dépôt ecriture-rs. Lancé détaché (setsid).

REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
git -C "$REPO" remote get-url origin 2>/dev/null | grep -q 'ecriture-rs' || exit 0
SKILL="$REPO/.agents/skills/archive-vibe-sessions/rebase_vibe_sessions.py"
[ -f "$SKILL" ] || exit 0
cd "$REPO" || exit 0
EXPORT_DIR=/tmp/vibe-sessions-export

git fetch --all --prune 2>/dev/null

# export des sessions locales du projet (cwd == REPO ou sous-dossier)
git worktree add /tmp/ecriture-main origin/main 2>/dev/null
if [ -d /tmp/ecriture-main ]; then
  (cd /tmp/ecriture-main && CARGO_TARGET_DIR="$REPO/target" cargo build --bin session-export) 2>/dev/null
fi
BIN="$REPO/target/debug/session-export"
if [ -x "$BIN" ]; then
  IDS=$(python3 -c "
import json, glob, os, sys
repo = sys.argv[1]
for d in sorted(glob.glob(os.path.expanduser('~/.vibe/logs/session/unified/*/meta.json'))):
    m = json.load(open(d))
    cwd = (m.get('environment') or {}).get('working_directory') or ''
    if cwd == repo or cwd.startswith(repo + '/'):
        print(m.get('session_id'))
" "$REPO")
  mkdir -p "$EXPORT_DIR"
  for id in $IDS; do
    "$BIN" --export "$id" --out "$EXPORT_DIR"
  done
fi

# rebuild de l'arbre vibe (catégorisation + chrono + scrub secrets + scan final)
if python3 "$SKILL" "$EXPORT_DIR"; then
  git push --force-with-lease origin vibe vibe-interface vibe-reconnaissance
  # supprime les éventuelles anciennes branches vibe/session-* distantes
  for r in $(git ls-remote --heads origin 'refs/heads/vibe/session-*' | cut -f2); do
    git push origin --delete "$r" 2>/dev/null
  done
fi
