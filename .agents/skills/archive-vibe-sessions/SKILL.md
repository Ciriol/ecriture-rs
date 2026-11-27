---
name: archive-vibe-sessions
description: MANDATORY for EVERY Vibe session working in this project (ecriture-rs). At session start and again before ending the session, archive the locally journaled Vibe sessions (~/.vibe/logs/session/unified/) into the git branches vibe/session-<slug> on origin — one commit per session, chained strictly in start_time order, secrets scrubbed, never any project code — then push with --force-with-lease. Always load and apply this skill; run the procedure without asking for confirmation.
---

# Archivage obligatoire des sessions Vibe (ecriture-rs)

Toute session Vibe dans ce projet DOIT suivre cette méthode — au démarrage de la
session et avant de la terminer — sans demander de confirmation à l'utilisateur.

## Règles absolues

1. **Aucun fichier du projet** dans `refs/heads/vibe/*` : uniquement
   `sessions/<slug>/{RESUME.md, journal.jsonl, meta.json, records/}` plus l'index
   global `sessions/{README.md, index.json}`. Vérifié avec
   `git ls-tree -r <branche> | grep -v '^sessions/'` → doit être vide.
2. **Le worktree courant n'est jamais modifié** : plumbing git uniquement
   (`hash-object` / `update-index` / `write-tree` / `commit-tree` / `update-ref`
   sur un index temporaire).
3. **Aucun secret dans les commits** : le push protection GH013 de GitHub bloque
   sinon tout le push. Tout token (`ghp_…`, `github_pat_…`, `sk-…`, `AKIA…`,
   `AIza…`, `xox…`, JWT `eyJ…`) doit être remplacé par `***REDACTED***` avant le
   push. Le script ci-dessous le fait automatiquement ; un `--scan` vérifie.
4. **Ordre chronologique** : l'arbre `vibe/` est une chaîne linéaire triée par
   `start_time` de chaque session (timestamp de `meta.json`, pas l'ordre de
   push). Chaque commit a pour parent la session précédente et des dates
   auteur/commiter égales au `start_time` de sa session.
5. **Push** : `git push --force-with-lease origin 'refs/heads/vibe/*'`
   (les branches réécrites par le rebase exigent le force-with-lease).
6. **Filtrage par projet** : seules les sessions **en rapport avec ce projet**
   sont exportées — celles dont le cwd (`meta.json` →
   `environment.working_directory`) est la racine du dépôt **ou un
   sous-dossier**. Une session d'un autre dossier (ex. `cprn_infra`, ansible)
   n'appartient pas à ce dépôt : pour la retirer, supprimer sa branche
   locale et distante (`git branch -D vibe/session-<slug>` +
   `git push origin --delete vibe/session-<slug>`), puis relancer le rebase
   (l'entrée disparaît de `sessions/index.json`/`README.md` de la chaîne).

## Procédure (à exécuter à chaque session)

```bash
cd "$(git rev-parse --show-toplevel)"   # racine du clone ecriture-rs (portable)
git fetch --all --prune

# 1) rapatrier l'arbre vibe/ distant en branches locales
#    (clone neuf : fetch direct ; sinon : branches depuis origin/vibe/*)
git fetch origin '+refs/heads/vibe/*:refs/heads/vibe/*' 2>/dev/null
for r in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin/vibe/); do
  git branch --force "${r#origin/}" "$r" 2>/dev/null || true
done

# 2) exporter UNIQUEMENT les sessions de ce projet (cwd == racine du dépôt) —
#    outil session-export (sur main) ; si la branche courante ne l'a pas,
#    builder via origin/main :
git worktree add /tmp/ecriture-main origin/main 2>/dev/null || true
(cd /tmp/ecriture-main && cargo build --bin session-export)
/tmp/ecriture-main/target/debug/session-export --list
IDS=$(python3 -c "
import json, glob, os, sys
repo = os.getcwd()
for d in sorted(glob.glob(os.path.expanduser('~/.vibe/logs/session/unified/*/meta.json'))):
    m = json.load(open(d))
    cwd = (m.get('environment') or {}).get('working_directory') or ''
    if cwd == repo or cwd.startswith(repo + '/'):   # ce projet uniquement
        print(m.get('session_id'))
")
for id in $IDS; do /tmp/ecriture-main/target/debug/session-export --export "$id" --branch; done   # sans --push
#    (si on est déjà sur main : cargo run --bin session-export -- --export <id> --branch)

# 3) rebase chronologique par start_time + nettoyage des secrets
python3 .agents/skills/archive-vibe-sessions/rebase_vibe_sessions.py

# 4) vérifications : ordre chrono, pas de code projet, 0 secret
LAST=$(git for-each-ref --format='%(refname:short)' refs/heads/vibe/ | sort | tail -1)
git log --format='%h %ad %s' --date=iso "$LAST"          # N sessions, dates croissantes
for b in $(git for-each-ref --format='%(refname)' refs/heads/vibe/); do
  git ls-tree -r --name-only "$b" | grep -v '^sessions/'  # rien
  git ls-tree -r --name-only "$b" | grep -q "sessions/${b#refs/heads/vibe/session-}/" || echo "MANQUE $b"
done
python3 .agents/skills/archive-vibe-sessions/rebase_vibe_sessions.py --scan   # 0 secret

# 5) pousser l'arbre vibe/
git push --force-with-lease origin 'refs/heads/vibe/*'
```

## Fichiers support

- `rebase_vibe_sessions.py` — rebase chronologique (chaîne ordonnée par
  `start_time`, dates = timestamp de session) + masquage des secrets dans tous
  les blobs ; mode `--scan` pour vérifier qu'il ne reste aucun secret.
- `.vibe/hooks.toml` + `.vibe/hooks/archive_vibe_sessions.sh` +
  `.vibe/hooks/archive_vibe_sessions_context.py` — hooks **projet** (actifs
  dès que le clone est trusté) : `session_start` injecte l'obligation de
  suivre ce skill dans le contexte du modèle ; `session_end` lance
  l'export filtré + rebase + scrub + push, détaché via `setsid`.

## Répliquer sur un autre ordinateur (via ce dépôt git)

La méthode voyage avec le dépôt :

```bash
git clone <url-du-dépôt> && cd ecriture-rs
git checkout export-inkml     # branche portant le skill et les hooks projet
# trust le dossier une fois : vibe --trust (ou /trusted-folders dans Vibe)
git fetch origin '+refs/heads/vibe/*:refs/heads/vibe/*'   # arbre vibe/ des autres machines
```

Dès lors, toute session ouverte dans ce clone : le hook `session_start`
injecte l'obligation de suivre ce skill, et le hook `session_end` exporte
les sessions locales **de ce clone** (filtre cwd), les fusionne au rebase
chronologique avec celles de l'autre machine (via `origin/vibe/*`), masque
les secrets et push `--force-with-lease origin 'refs/heads/vibe/*'`.

## Pièges connus

- Un token présent dans un journal de session bloque tout le push (GH013) :
  le révoquer côté GitHub ET le laisser masqué dans les branches vibe/*.
- Ne jamais créer de branche nommée exactement `vibe` : git interdit
  `refs/heads/vibe` et `refs/heads/vibe/...` simultanément. La « branche vibe »
  est le namespace `vibe/session-*`, la tête étant la session la plus récente.
- `session-export` lit `~/.vibe/logs/session/unified/` (métadonnées de CETTE
  machine) : les sessions d'une autre machine arrivent via les branches
  `origin/vibe/*` et participent au même rebase chronologique.
