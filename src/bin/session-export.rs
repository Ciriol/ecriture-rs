//! session-export : exporte les sessions Vibe journalisées localement
//! (`~/.vibe/logs/session/unified/<id>/`) dans le projet.
//!
//! Deux modes :
//!   --out DIR (défaut ./sessions) : écrit sessions/<slug>/{RESUME.md, journal.jsonl,
//!                                   meta.json, records/} dans le worktree courant.
//!   --branch : construit l'arbre git sous le namespace `vibe/` SANS toucher au
//!              worktree (git interdit refs/heads/vibe + refs/heads/vibe/... en même
//!              temps, donc pas de branche racine "vibe") :
//!              - une branche `vibe/session-<date>-<slug>` par session, commit RACINE
//!                autonome (elle ne contiendra donc jamais le code du projet),
//!              - chaque branche embarque RESUME.md + journal + records + l'index
//!                global sessions/index.json / sessions/README.md à jour.
//!              --push pousse la branche créée (`git push origin 'refs/heads/vibe/*'`
//!              pour tout pousser).
//!
//! Usage :
//!   cargo run --bin session-export -- --list
//!   cargo run --bin session-export -- --export last
//!   cargo run --bin session-export -- --export <session-id> --branch --push
//!   cargo run --bin session-export -- --export all --branch
//!   options : --no-raw (RESUME.md seul, pas journal.jsonl/meta.json), --out DIR

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Clone, Debug)]
struct SessionInfo {
    id: String,
    dir: PathBuf,
    start: String,
    end: String,
    title: String,
    preview: String,
    tokens: u64,
    journal_files: Vec<PathBuf>,
    journal_bytes: u64,
}

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
struct IndexEntry {
    slug: String,
    session_id: String,
    title: String,
    start: String,
    end: String,
    tokens: u64,
    branch: String,
    raw: bool,
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| ".".into()))
}

fn unified_dir() -> PathBuf {
    home().join(".vibe/logs/session/unified")
}

/// Charge .session-index.json (titres, previews, tokens) -> map id -> entrée.
fn load_index() -> serde_json::Value {
    let p = unified_dir().join(".session-index.json");
    std::fs::read_to_string(&p)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or(serde_json::Value::Null)
}

fn index_entry_for<'a>(index: &'a serde_json::Value, id: &str) -> Option<&'a serde_json::Value> {
    index
        .get("entries")?
        .as_array()?
        .iter()
        .find(|e| e.get("session_id").and_then(|s| s.as_str()) == Some(id))
}

fn list_sessions() -> Vec<SessionInfo> {
    let index = load_index();
    let mut v = Vec::new();
    let rd = match std::fs::read_dir(unified_dir()) {
        Ok(r) => r,
        Err(_) => return v,
    };
    for e in rd.flatten() {
        let dir = e.path();
        if !dir.is_dir() {
            continue;
        }
        let id = match dir.file_name().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        let meta: serde_json::Value = std::fs::read_to_string(dir.join("meta.json"))
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or(serde_json::Value::Null);
        let (title, preview, tokens) = match index_entry_for(&index, &id) {
            Some(en) => {
                let sess = &en["session"];
                (
                    sess.get("title")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                    sess.get("preview")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                    sess
                        .get("tokenUsage")
                        .and_then(|t| t.get("totalTokens"))
                        .and_then(|t| t.as_u64())
                        .unwrap_or(0),
                )
            }
            None => (String::new(), String::new(), 0),
        };
        let journal_dir = dir.join("journal");
        let mut journal_files = Vec::new();
        let mut journal_bytes = 0u64;
        if let Ok(jr) = std::fs::read_dir(&journal_dir) {
            for jf in jr.flatten() {
                if jf.path().extension().and_then(|e| e.to_str()) == Some("jsonl") {
                    journal_bytes += jf.metadata().map(|m| m.len()).unwrap_or(0);
                    journal_files.push(jf.path());
                }
            }
        }
        journal_files.sort();
        v.push(SessionInfo {
            id,
            start: meta
                .get("start_time")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string(),
            end: meta
                .get("end_time")
                .and_then(|t| t.as_str())
                .unwrap_or("")
                .to_string(),
            title,
            preview,
            tokens,
            dir,
            journal_files,
            journal_bytes,
        });
    }
    v.sort_by(|a, b| a.start.cmp(&b.start));
    v
}

/// Translittération simple des accents latins courants pour les slugs.
fn deaccent(c: char) -> &'static str {
    match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => "a",
        'ç' => "c",
        'è' | 'é' | 'ê' | 'ë' => "e",
        'ì' | 'í' | 'î' | 'ï' => "i",
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' => "o",
        'ù' | 'ú' | 'û' | 'ü' => "u",
        'ý' | 'ÿ' => "y",
        'ñ' => "n",
        _ => "",
    }
}

/// "Tablette dessin trace affichage" -> "tablette-dessin-trace-affichage"
fn slugify(s: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = true;
    for c in s.chars() {
        let lower = c.to_ascii_lowercase();
        let deacc = deaccent(c);
        let piece: &str = if deacc.is_empty() { "" } else { deacc };
        let c = if piece.is_empty() { lower } else { piece.chars().next().unwrap() };
        if c.is_ascii_alphanumeric() {
            out.push(c);
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        out.push_str("session");
    }
    out.truncate(60);
    out
}

/// "2026-11-26T16:36:41.525000+00:00" -> "2026-11-26-1636"
fn date_slug(start: &str) -> String {
    let date = start.get(0..10).unwrap_or("").replace('-', "");
    let time = start
        .get(11..16)
        .unwrap_or("")
        .replace(':', "");
    if date.is_empty() {
        "0000-00-00-0000".into()
    } else {
        format!("{date}-{time}")
    }
}

// ---------- extraction du journal pour le RESUME.md ----------

struct JournalExtract {
    user_texts: Vec<String>,
    shell_cmds: Vec<String>,
    tool_counts: BTreeMap<String, usize>,
    record_paths: Vec<String>,
}

fn extract_journal(text: &str) -> JournalExtract {
    let mut ex = JournalExtract {
        user_texts: Vec::new(),
        shell_cmds: Vec::new(),
        tool_counts: BTreeMap::new(),
        record_paths: Vec::new(),
    };
    for line in text.lines() {
        let Ok(o) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let Some(payload) = o.get("payload") else { continue };
        // messages utilisateur : payload.input.type == "user_message"
        if let Some(input) = payload.get("input") {
            if input.get("type").and_then(|t| t.as_str()) == Some("user_message") {
                if let Some(content) = input.pointer("/command/content").and_then(|c| c.as_array()) {
                    for part in content {
                        if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                            ex.user_texts.push(t.to_string());
                        }
                    }
                }
            }
        }
        // tool calls : payload.result.result.parts[] avec arguments_json
        if let Some(parts) = payload
            .pointer("/result/result/parts")
            .and_then(|p| p.as_array())
        {
            for part in parts {
                let name = part
                    .get("name")
                    .or_else(|| part.get("tool_name"))
                    .and_then(|n| n.as_str())
                    .unwrap_or("?")
                    .to_string();
                *ex.tool_counts.entry(name).or_insert(0) += 1;
                if let Some(args) = part.get("arguments_json").and_then(|a| a.as_str()) {
                    if let Ok(av) = serde_json::from_str::<serde_json::Value>(args) {
                        if let Some(cmd) = av.get("command").and_then(|c| c.as_str()) {
                            ex.shell_cmds.push(cmd.to_string());
                        }
                    }
                }
            }
        }
        // chemins --record-input / --replay-input mentionnés dans la ligne brute
        for needle in ["--record-input ", "--replay-input "] {
            let mut rest = line;
            while let Some(pos) = rest.find(needle) {
                let after = &rest[pos + needle.len()..];
                let path: String = after
                    .chars()
                    .take_while(|c| !c.is_whitespace() && *c != '"' && *c != '\'' && *c != '\\')
                    .collect();
                if !path.is_empty() && !ex.record_paths.contains(&path) {
                    ex.record_paths.push(path);
                }
                rest = after;
            }
        }
    }
    ex
}

fn truncate(s: &str, n: usize) -> String {
    let one: String = s.chars().take(n).collect();
    if s.chars().count() > n {
        format!("{one}…")
    } else {
        one
    }
}

fn gen_resume(info: &SessionInfo, slug: &str, ex: &JournalExtract, raw: bool) -> String {
    let mut md = String::new();
    let title = if info.title.is_empty() {
        truncate(&info.preview, 80)
    } else {
        info.title.clone()
    };
    md.push_str(&format!("# {title}\n\n"));
    md.push_str(&format!("- session : `{}`\n", info.id));
    md.push_str(&format!("- branche : `vibe/session-{slug}` (namespace vibe, sans code)\n"));
    md.push_str(&format!("- période : {} → {}\n", info.start, info.end));
    md.push_str(&format!("- tokens : {}\n", info.tokens));
    md.push_str(&format!("- journal brut : {}\n", if raw { "oui (journal.jsonl)" } else { "non (--no-raw)" }));
    md.push_str(&format!("- preview : {}\n", truncate(&info.preview, 200)));
    md.push('\n');
    if !ex.user_texts.is_empty() {
        md.push_str("## Messages de l'utilisateur\n\n");
        for t in ex.user_texts.iter().take(30) {
            md.push_str(&format!("- {}\n", truncate(t, 300)));
        }
        md.push('\n');
    }
    if !ex.shell_cmds.is_empty() {
        md.push_str("## Commandes exécutées (dédupliquées)\n\n");
        let mut seen = Vec::new();
        for c in &ex.shell_cmds {
            if !seen.contains(c) {
                seen.push(c.clone());
            }
        }
        for c in seen.iter().take(60) {
            md.push_str(&format!("- `{}`\n", truncate(c, 200)));
        }
        md.push('\n');
    }
    if !ex.tool_counts.is_empty() {
        md.push_str("## Outils appelés\n\n");
        let mut counts: Vec<_> = ex.tool_counts.iter().collect();
        counts.sort_by(|a, b| b.1.cmp(a.1));
        for (name, n) in counts.iter().take(15) {
            md.push_str(&format!("- {name} × {n}\n"));
        }
        md.push('\n');
    }
    if !ex.record_paths.is_empty() {
        md.push_str("## Enregistrements d'input (--record-input/--replay-input)\n\n");
        for p in &ex.record_paths {
            md.push_str(&format!("- `{p}`\n"));
        }
        md.push('\n');
    }
    md
}

// ---------- git plumbing (ne touche jamais au worktree) ----------

fn git(repo: &Path, args: &[&str], index: Option<&Path>) -> Result<String, String> {
    let mut c = Command::new("git");
    c.arg("-C").arg(repo).args(args);
    if let Some(idx) = index {
        c.env("GIT_INDEX_FILE", idx);
    }
    let out = c.output().map_err(|e| format!("git {args:?}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {args:?} a échoué: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn repo_root() -> Result<PathBuf, String> {
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    Ok(PathBuf::from(git(&cwd, &["rev-parse", "--show-toplevel"], None)?))
}

/// Liste les branches `vibe/session-*` existantes (triées par nom = par date).
fn list_session_branches(repo: &Path) -> Result<Vec<String>, String> {
    let out = git(
        repo,
        &["for-each-ref", "--format=%(refname)", "refs/heads/vibe/"],
        None,
    )?;
    let mut v: Vec<String> = out.lines().map(|l| l.to_string()).collect();
    v.sort();
    Ok(v)
}

/// Construit un commit RACINE (sans parent) contenant `files` + l'index global,
/// et crée la branche `vibe/session-<slug>`. Chaque branche est autonome :
/// elle ne contient que des sessions, jamais le code du projet.
fn commit_session_branch(
    repo: &Path,
    slug: &str,
    files: &[(String, PathBuf)],
    index_entries: &[IndexEntry],
    readme: &str,
) -> Result<String, String> {
    let tmpidx = std::env::temp_dir().join(format!("vibe-index-{}", std::process::id()));
    let _ = std::fs::remove_file(&tmpidx);
    let idx = Some(tmpidx.as_path());
    let mut all: Vec<(String, PathBuf)> = files.to_vec();
    // index.json + README.md globaux à la racine de la branche
    let index_json = serde_json::to_string_pretty(index_entries).map_err(|e| e.to_string())?;
    let tmp_index = std::env::temp_dir().join(format!("vibe-index-json-{}", std::process::id()));
    std::fs::write(&tmp_index, index_json).map_err(|e| e.to_string())?;
    let tmp_readme = std::env::temp_dir().join(format!("vibe-readme-{}", std::process::id()));
    std::fs::write(&tmp_readme, readme).map_err(|e| e.to_string())?;
    all.push(("sessions/index.json".to_string(), tmp_index));
    all.push(("sessions/README.md".to_string(), tmp_readme));
    for (rel, src) in &all {
        let blob = git(repo, &["hash-object", "-w", &src.to_string_lossy()], idx)?;
        git(
            repo,
            &["update-index", "--add", "--cacheinfo", &format!("100644,{blob},{rel}")],
            idx,
        )?;
    }
    let tree = git(repo, &["write-tree"], idx)?;
    let msg = format!("session {slug} (branche vibe, sans code)");
    let commit = git(repo, &["commit-tree", &tree, "-m", &msg], None)?;
    let branch = format!("refs/heads/vibe/session-{slug}");
    git(repo, &["update-ref", &branch, &commit], None)?;
    let _ = std::fs::remove_file(&tmpidx);
    Ok(commit)
}

/// Reconstruit l'index global des sessions exportées : on prend l'index cumulatif
/// de la branche la plus récente (les slugs commencent par la date).
fn read_existing_index(repo: &Path) -> Vec<IndexEntry> {
    let branches = match list_session_branches(repo) {
        Ok(b) => b,
        Err(_) => return Vec::new(),
    };
    if let Some(last) = branches.last() {
        let out = Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(["show", &format!("{last}:sessions/index.json")])
            .output();
        if let Ok(o) = out {
            if o.status.success() {
                return serde_json::from_slice(&o.stdout).unwrap_or_default();
            }
        }
    }
    Vec::new()
}

fn gen_readme(entries: &[IndexEntry]) -> String {
    let mut md = String::from("# Sessions Vibe\n\nSessions exportées depuis `~/.vibe/logs/session/unified/` — une branche `vibe/session-<slug>` par session.\n\n| Session | Période | Tokens | Branche |\n|---|---|---|---|\n");
    for e in entries {
        md.push_str(&format!(
            "| {} | {} → {} | {} | `{}` |\n",
            e.title, e.start.get(0..10).unwrap_or(""), e.end.get(0..10).unwrap_or(""), e.tokens, e.branch
        ));
    }
    md
}

// ---------- export ----------

struct ExportOpts {
    branch: bool,
    push: bool,
    raw: bool,
    out: PathBuf,
}

/// Exporte une session. Retourne le slug.
fn export_session(info: &SessionInfo, opts: &ExportOpts) -> Result<String, String> {
    let slug = format!("{}-{}", date_slug(&info.start), slugify(if info.title.is_empty() { &info.preview } else { &info.title }));
    // 1) préparer les fichiers dans un dossier temporaire
    let tmp = std::env::temp_dir().join(format!("session-export-{slug}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    let mut journal_text = String::new();
    for jf in &info.journal_files {
        if let Ok(t) = std::fs::read_to_string(jf) {
            journal_text.push_str(&t);
            journal_text.push('\n');
        }
    }
    let ex = extract_journal(&journal_text);
    let resume = gen_resume(info, &slug, &ex, opts.raw);
    std::fs::write(tmp.join("RESUME.md"), &resume).map_err(|e| e.to_string())?;

    let mut files: Vec<(String, PathBuf)> = vec![(format!("sessions/{slug}/RESUME.md"), tmp.join("RESUME.md"))];
    if opts.raw {
        let jdst = tmp.join("journal.jsonl");
        std::fs::write(&jdst, &journal_text).map_err(|e| e.to_string())?;
        files.push((format!("sessions/{slug}/journal.jsonl"), jdst));
        let mdst = tmp.join("meta.json");
        let _ = std::fs::copy(info.dir.join("meta.json"), &mdst);
        files.push((format!("sessions/{slug}/meta.json"), mdst));
    }
    // records trouvés dans le journal (--record-input /tmp/xxx.jsonl) : copiés si présents
    for rp in &ex.record_paths {
        let src = PathBuf::from(rp);
        if src.exists() {
            let name = src.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "record.jsonl".into());
            let dst = tmp.join("records").join(&name);
            std::fs::create_dir_all(dst.parent().unwrap()).ok();
            if std::fs::copy(&src, &dst).is_ok() {
                files.push((format!("sessions/{slug}/records/{name}"), dst));
            }
        }
    }

    if !opts.branch {
        // mode worktree : copier dans opts.out/<slug>/
        let dst = opts.out.join(&slug);
        let _ = std::fs::remove_dir_all(&dst);
        std::fs::create_dir_all(&dst).map_err(|e| e.to_string())?;
        for (_, src) in &files {
            let rel = src.strip_prefix(&tmp).unwrap();
            let d = dst.join(rel);
            std::fs::create_dir_all(d.parent().unwrap()).ok();
            std::fs::copy(src, &d).map_err(|e| e.to_string())?;
        }
        eprintln!("exporté dans {}", dst.display());
        return Ok(slug);
    }

    // mode branche : arbre git `vibe/session-*` (namespace, sans branche racine :
    // git interdit refs/heads/vibe et refs/heads/vibe/... simultanément)
    let repo = repo_root()?;
    let mut entries = read_existing_index(&repo);
    entries.retain(|e| e.slug != slug);
    let branch_name = format!("vibe/session-{slug}");
    entries.push(IndexEntry {
        slug: slug.clone(),
        session_id: info.id.clone(),
        title: if info.title.is_empty() { truncate(&info.preview, 60) } else { info.title.clone() },
        start: info.start.clone(),
        end: info.end.clone(),
        tokens: info.tokens,
        branch: branch_name.clone(),
        raw: opts.raw,
    });
    entries.sort_by(|a, b| a.start.cmp(&b.start));
    let readme = gen_readme(&entries);
    let commit = commit_session_branch(&repo, &slug, &files, &entries, &readme)?;
    eprintln!("commit {} sur {branch_name}", &commit[..7.min(commit.len())]);
    if opts.push {
        git(&repo, &["push", "origin", &branch_name], None)?;
        eprintln!("pushé: {branch_name}");
    }
    Ok(slug)
}

fn usage() -> ! {
    eprintln!(
        "usage: session-export --list\n\
         \x20       session-export --export <id|last|all> [--branch] [--push] [--no-raw] [--out DIR]\n\
         --list          liste les sessions trouvées dans ~/.vibe/logs/session/unified/\n\
         --export ...    exporte dans ./sessions (ou --out DIR), ou dans l'arbre git\n\
         \x20               vibe/ avec --branch (branche vibe/session-<slug> par session)\n\
         --push          push la branche (avec --branch)\n\
         --no-raw        exporte seulement RESUME.md (pas journal.jsonl/meta.json)"
    );
    std::process::exit(2)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut list = false;
    let mut export: Option<String> = None;
    let opts = ExportOpts {
        branch: false,
        push: false,
        raw: true,
        out: PathBuf::from("sessions"),
    };
    let mut opts = opts;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--list" => list = true,
            "--export" => {
                i += 1;
                export = Some(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "--branch" => opts.branch = true,
            "--push" => opts.push = true,
            "--no-raw" => opts.raw = false,
            "--out" => {
                i += 1;
                opts.out = PathBuf::from(args.get(i).cloned().unwrap_or_else(|| usage()));
            }
            "-h" | "--help" => usage(),
            "--" => {} // séparateur cargo run --bin ... --
            other => {
                eprintln!("argument inconnu: {other}");
                usage();
            }
        }
        i += 1;
    }
    let sessions = list_sessions();
    if list {
        println!("{} session(s) dans {}", sessions.len(), unified_dir().display());
        for s in &sessions {
            println!(
                "  {}  {}  {}  {:.2} Mo  {}",
                s.id.get(0..8).unwrap_or(""),
                s.start.get(0..16).unwrap_or(""),
                if s.title.is_empty() { truncate(&s.preview, 50) } else { s.title.clone() },
                s.journal_bytes as f64 / 1e6,
                s.id
            );
        }
        return;
    }
    let Some(what) = export else { usage() };
    let selected: Vec<&SessionInfo> = match what.as_str() {
        "all" => sessions.iter().collect(),
        "last" => sessions.last().into_iter().collect(),
        id => {
            let found: Vec<_> = sessions.iter().filter(|s| s.id.starts_with(id)).collect();
            if found.is_empty() {
                eprintln!("session inconnue: {id} (voir --list)");
                std::process::exit(1);
            }
            found
        }
    };
    for s in selected {
        match export_session(s, &opts) {
            Ok(slug) => println!("OK {slug}  ({})", s.id),
            Err(e) => {
                eprintln!("ERREUR pour {}: {e}", s.id);
                std::process::exit(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugify_titres() {
        assert_eq!(slugify("Tablette dessin trace affichage"), "tablette-dessin-trace-affichage");
        assert_eq!(slugify("Fix: hover souris après stylet !"), "fix-hover-souris-apres-stylet");
        assert_eq!(slugify(""), "session");
        assert_eq!(slugify("déjà là"), "deja-la");
    }

    #[test]
    fn date_slug_iso() {
        assert_eq!(date_slug("2026-11-26T16:36:41.525000+00:00"), "20261126-1636");
        assert_eq!(date_slug(""), "0000-00-00-0000");
    }

    const MINI_JOURNAL: &str = r#"{"type":"core_input","payload":{"input":{"type":"user_message","command":{"content":[{"text":"le tracé n'apparaît plus","type":"text"}]}}}}
{"type":"action_result","payload":{"result":{"result":{"parts":[{"name":"bash","arguments_json":"{\"command\": \"cargo run -- --record-input /tmp/bug.jsonl\"}"},{"name":"edit","arguments_json":"{\"file_path\": \"src/main.rs\"}"}]}}}}
"#;

    #[test]
    fn extract_user_cmds_tools_records() {
        let ex = extract_journal(MINI_JOURNAL);
        assert_eq!(ex.user_texts, vec!["le tracé n'apparaît plus".to_string()]);
        assert_eq!(ex.shell_cmds, vec!["cargo run -- --record-input /tmp/bug.jsonl".to_string()]);
        assert_eq!(ex.tool_counts.get("bash"), Some(&1));
        assert_eq!(ex.tool_counts.get("edit"), Some(&1));
        assert_eq!(ex.record_paths, vec!["/tmp/bug.jsonl".to_string()]);
    }

    #[test]
    fn resume_contient_les_sections() {
        let info = SessionInfo {
            id: "abc123".into(),
            dir: PathBuf::from("/tmp"),
            start: "2026-11-26T16:36:41".into(),
            end: "2026-11-26T17:29:36".into(),
            title: "Tablette dessin".into(),
            preview: "preview...".into(),
            tokens: 42,
            journal_files: vec![],
            journal_bytes: 0,
        };
        let ex = extract_journal(MINI_JOURNAL);
        let md = gen_resume(&info, "20261126-1636-tablette-dessin", &ex, true);
        assert!(md.contains("# Tablette dessin"));
        assert!(md.contains("vibe/session-20261126-1636-tablette-dessin"));
        assert!(md.contains("le tracé n'apparaît plus"));
        assert!(md.contains("cargo run -- --record-input /tmp/bug.jsonl"));
        assert!(md.contains("/tmp/bug.jsonl"));
    }
}
