#!/bin/sh
# Hook session_end (projet .vibe/hooks.toml) : archive les sessions Vibe
# journalisées EN RAPPORT AVEC CE PROJET dans les branches vibe/session-*,
# rebase chronologique par start_time, secrets masqués, puis push
# --force-with-lease. Plumbing git uniquement : le worktree n'est jamais
# modifié. Portable : fonctionne dans tout clone du dépôt ecriture-rs.
# Lancé détaché (setsid) par hooks.toml.

REPO="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
# garde-fou : ne s'applique qu'au dépôt ecriture-rs
git -C "$REPO" remote get-url origin 2>/dev/null | grep -q 'ecriture-rs' || exit 0
SKILL="$REPO/.agents/skills/archive-vibe-sessions/rebase_vibe_sessions.py"
[ -f "$SKILL" ] || exit 0
cd "$REPO" || exit 0

git fetch --all --prune 2>/dev/null

# branches vibe/* distantes -> locales (récupère aussi celles de l'autre machine)
for r in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin/vibe/); do
  git branch --force "${r#origin/}" "$r" 2>/dev/null
done

# export des sessions locales en rapport avec ce projet (cwd == REPO ou sous-dossier)
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
  for id in $IDS; do
    "$BIN" --export "$id" --branch
  done
fi

# rebase chronologique par start_time + scrub des secrets + scan final
# (exit non nul si un secret reste -> pas de push)
if python3 "$SKILL"; then
  git push --force-with-lease origin 'refs/heads/vibe/*'
fi
