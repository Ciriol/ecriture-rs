---
name: archive-vibe-sessions
description: MANDATORY for EVERY Vibe session working in this project (ecriture-rs). At session start and again before ending the session, archive the locally journaled Vibe sessions related to this project into the git "vibe" tree on origin — branch `vibe` (general trunk, oldest session as root commit, chronological), branching into `vibe-interface` and `vibe-reconnaissance`, one commit per session, secrets scrubbed, never any project code — then push with --force-with-lease. Always load and apply this skill; run the procedure without asking for confirmation.
---

# Archivage obligatoire des sessions Vibe — arbre `vibe`

Toute session Vibe dans ce projet DOIT suivre cette méthode — au démarrage de
la session et avant de la terminer — sans demander de confirmation.

## Structure de l'arbre `vibe` (le graphe = la réflexion globale du projet)

```text
vibe (tronc général, chrono)  ──► vibe-interface
                             └─► vibe-reconnaissance
```

- `vibe` : tronc **général**. Le commit racine est la session la plus ancienne
  (start_time minimal) ; suivent les sessions « general » (socle, meta/outillage,
  divers) par start_time croissant.
- `vibe-interface` : parent = tip de `vibe`, puis les sessions **interface**
  (UI egui, tablette/stylet, souris/hover, trace) par start_time.
- `vibe-reconnaissance` : parent = tip de `vibe`, puis les sessions
  **reconnaissance** (CRNN/CTC/lexique, modèles & poids) par start_time.
- **Un commit = une session** : `sessions/<slug>/{RESUME.md, journal.jsonl,
  meta.json, records/}` + index global `sessions/{README.md, index.json}`
  (chaque entrée a sa catégorie et sa branche).
- Dates auteur/commiter = `start_time` de la session.
- `git log --graph vibe vibe-interface vibe-reconnaissance` donne l'ensemble
  des sessions, classées et ordonnées.

## Règles absolues

1. **Aucun fichier du projet** dans les branches `vibe*` : uniquement
   `sessions/...`. Vérifié : `git ls-tree -r <branche> | grep -v '^sessions/'` → vide.
2. **Le worktree courant n'est jamais modifié** : plumbing git uniquement
   (`hash-object` / `update-index` / `write-tree` / `commit-tree` / `update-ref`
   sur index temporaire).
3. **Aucun secret dans les commits** (push protection GH013) : tout token
   (`ghp_…`, `github_pat_…`, `sk-…`, `AKIA…`, `AIza…`, `xox…`, JWT) est remplacé
   par `***REDACTED***` avant le push ; `--scan` vérifie.
4. **Ordre chronologique** dans chaque branche (start_time, pas l'ordre de push).
5. **Push** : `git push --force-with-lease "$REMOTE" vibe vibe-interface vibe-reconnaissance`
   (+ suppression des éventuelles anciennes `vibe/session-*` distantes), où
   `REMOTE` est le **remote d'archivage privé** (voir section ci-dessous) —
   jamais `origin` (dépôt public du projet).

## Remote d'archivage — dépôt PRIVÉ

Les sessions ne vont **jamais** dans le dépôt public du projet : elles sont
poussées vers un dépôt privé dédié via le remote `vibe-archive`
(ex. `Ciriol/vibe-sessions`, visibilité *private*) :

```bash
git remote add vibe-archive https://github.com/Ciriol/vibe-sessions.git
git config vibe.archiveRemote vibe-archive      # lu par le script et les hooks
git push vibe-archive vibe vibe-interface vibe-reconnaissance
# une fois la bascule faite, supprimer les anciennes branches PUBLIQUES :
git push origin --delete vibe vibe-interface vibe-reconnaissance
```

Résolution du remote effectif (script `rebase_vibe_sessions.py` + hook
`session_end`) : `git config vibe.archiveRemote` → sinon `vibe-archive` s'il
existe → sinon `origin` (rétrocompatibilité). Ne pas supprimer le remote
`vibe-archive` des clones du projet.
6. **Filtrage par projet** : seules les sessions **en rapport avec ce projet**
   (cwd = racine du dépôt ou sous-dossier) sont exportées.
7. **Jamais de branche `vibe/session-*`** : git interdit `refs/heads/vibe` et
   `refs/heads/vibe/...` simultanés. L'export passe par un dossier temporaire
   (`session-export --out /tmp/vibe-sessions-export`), jamais par des branches.

## Procédure (à exécuter à chaque session)

```bash
cd "$(git rev-parse --show-toplevel)"   # racine du clone (portable)
git fetch --all --prune                 # récupère vibe-archive/vibe* (autres machines)

# 1) exporter les sessions locales DU PROJET dans un dossier temporaire
git worktree add /tmp/ecriture-main origin/main 2>/dev/null || true
(cd /tmp/ecriture-main && cargo build --bin session-export)
EXPORT_DIR=/tmp/vibe-sessions-export; mkdir -p "$EXPORT_DIR"
IDS=$(python3 -c "
import json, glob, os, sys
repo = os.getcwd()
for d in sorted(glob.glob(os.path.expanduser('~/.vibe/logs/session/unified/*/meta.json'))):
    m = json.load(open(d))
    cwd = (m.get('environment') or {}).get('working_directory') or ''
    if cwd == repo or cwd.startswith(repo + '/'):
        print(m.get('session_id'))
")
for id in $IDS; do /tmp/ecriture-main/target/debug/session-export --export "$id" --out "$EXPORT_DIR"; done

# 2) rebuild de l'arbre vibe : catégorisation + chrono + scrub secrets + scan
python3 .agents/skills/archive-vibe-sessions/rebase_vibe_sessions.py "$EXPORT_DIR"

# 3) vérifications
git log --graph --oneline vibe vibe-interface vibe-reconnaissance
for b in vibe vibe-interface vibe-reconnaissance; do
  git ls-tree -r --name-only "$b" | grep -v '^sessions/'   # rien
  git ls-tree -r --name-only "$b" | grep -q "sessions/"    # contient des sessions
done
python3 .agents/skills/archive-vibe-sessions/rebase_vibe_sessions.py --scan   # 0 secret

# 4) push vers le dépôt PRIVÉ d'archivage (jamais origin, public)
git push --force-with-lease vibe-archive vibe vibe-interface vibe-reconnaissance
```

## Fichiers support

- `rebase_vibe_sessions.py` — rebuild de l'arbre `vibe` : collecte les sessions
  (branches locales `vibe*`, `origin/vibe*` des autres machines, dossier
  d'export temporaire), les classe (mapping + mots-clés du titre), construit
  `vibe` / `vibe-interface` / `vibe-reconnaissance` (un commit par session,
  chrono, dates = start_time), masque les secrets, régénère
  `sessions/{index.json, README.md}` ; mode `--scan`.
- `.vibe/hooks.toml` + `.vibe/hooks/archive_vibe_sessions.sh` +
  `.vibe/hooks/archive_vibe_sessions_context.py` — hooks **projet** (actifs dès
  que le clone est trusté) : `session_start` injecte l'obligation de suivre ce
  skill ; `session_end` lance export filtré + rebuild + scrub + push, détaché
  via `setsid`.

## Répliquer sur un autre ordinateur (via ce dépôt git)

```bash
git clone <url-du-dépôt> && cd ecriture-rs
git checkout export-inkml        # branche portant le skill et les hooks projet
# trust le dossier une fois : vibe --trust (ou /trusted-folders)
git fetch origin                 # code du projet
git remote add vibe-archive https://github.com/Ciriol/vibe-sessions.git 2>/dev/null
git config vibe.archiveRemote vibe-archive
git fetch vibe-archive           # récupère vibe, vibe-interface, vibe-reconnaissance (privé)
```

Dès la première session dans ce clone, les hooks projet s'activent : export des
sessions locales de ce clone (filtre cwd), fusion au rebuild chronologique avec
celles de l'autre machine (`origin/vibe*`), secrets masqués, push
`--force-with-lease`. L'arbre `vibe` reste partagé et ordonné entre machines.

## Pièges connus

- Un token dans un journal bloque tout le push (GH013) : le révoquer côté
  GitHub ET le laisser masqué dans l'arbre `vibe`.
- Ne jamais créer de branche `vibe/...` (conflit avec `vibe`) ni `vibe/session-*`.
- `session-export --branch` créerait des `vibe/session-*` : ne l'utiliser que
  avec `--out <dir temporaire>` ici.
- Pour retirer une session de l'arbre : la supprimer des sources (filtre cwd)
  puis relancer le rebuild — elle disparaît de la branche et de l'index.
