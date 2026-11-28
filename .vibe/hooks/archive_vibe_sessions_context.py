#!/usr/bin/env python3
"""Hook session_start (projet .vibe/hooks.toml) : rend obligatoire le skill
archive-vibe-sessions de ce dépôt (arbre vibe : tronc général + branches
vibe-interface / vibe-reconnaissance, un commit par session).

Écrit sur stdout la structured-response attendue par Vibe :
additional_context est injecté (hors vue utilisateur) dans le prochain
message utilisateur ; system_message est affiché dans la conversation.
Portable : chemins relatifs au dépôt.
"""
import json

CTX = (
    "MANDATORY for this session: load and follow the project skill "
    "`archive-vibe-sessions` (.agents/skills/archive-vibe-sessions/SKILL.md "
    "in this repository). It requires archiving every locally journaled Vibe "
    "session related to THIS project (~/.vibe/logs/session/unified/ entries "
    "whose cwd is this repository) into the git tree `vibe` on origin — branch "
    "`vibe` (general trunk, oldest session as root commit, chronological), "
    "branching into `vibe-interface` and `vibe-reconnaissance`, one commit per "
    "session, dates = session start_time, secrets scrubbed to ***REDACTED***, "
    "never any project code — then `git push --force-with-lease origin vibe "
    "vibe-interface vibe-reconnaissance`. Sessions from other machines arrive "
    "via `origin/vibe*`. A session_end hook (.vibe/hooks.toml) runs this "
    "automatically when the session closes; before ending the session, verify "
    "it succeeded, otherwise run `python3 "
    ".agents/skills/archive-vibe-sessions/rebase_vibe_sessions.py` followed by "
    "the push."
)

print(json.dumps({
    "system_message": "archive-vibe-sessions actif : archivage obligatoire des sessions Vibe de ce projet dans l'arbre vibe (tronc + vibe-interface / vibe-reconnaissance, un commit par session, chrono, secrets masqués, push --force-with-lease) — voir le skill du dépôt.",
    "hook_specific_output": {"additional_context": CTX},
}))
