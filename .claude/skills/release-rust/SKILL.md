---
name: release-rust
description: Cut a new release of this Rust crate — bump version, lint, format, commit, tag, push, then monitor CI/Release pipelines and self-heal until green. Use when the user says "release", "ship it", "cut a release", "tag a new version", "push a new feature release", or otherwise asks to publish what's on the working tree.
tools: Bash, Read, Edit, AskUserQuestion, Monitor
---

# Rust release workflow

End-to-end release flow for this crate. Triggered when the user wants to ship pending changes as a new tagged release.

## Inputs you need before starting

- Run `git status` and `git diff --stat` to see what's about to ship.
- Run `git log --oneline $(git describe --tags --abbrev=0)..HEAD` to see commits since the last tag (or full log if no prior tags).
- Read current version: `grep '^version' Cargo.toml`.
- List existing tags: `git tag --sort=-v:refname | head -5`.

## Phase 1 — Decide the bump

Infer the SemVer bump from the diff and commits:

- **major**: breaking API/CLI changes (renamed/removed flags, changed default behavior, removed public functions).
- **minor**: new feature / new subcommand / new flag (backwards-compatible additions).
- **patch**: bug fixes, doc/comment changes, internal refactors, dependency bumps with no behavior change.

If the answer is obvious from the diff, proceed with it. **Ask via AskUserQuestion only when the diff is genuinely ambiguous** (e.g. mixes a feature with a behavior tweak that might be breaking).

## Phase 2 — Prepare the tree

Run in this exact order. Stop and fix on any failure before moving on.

1. **Bump version in `Cargo.toml`** using the Edit tool. Single line:
   ```toml
   version = "X.Y.Z"
   ```
2. **`cargo build`** — this also refreshes `Cargo.lock` to match the new version. If it fails, fix and retry.
3. **`cargo clippy --all-targets -- -D warnings`** — treat warnings as errors. If lints fire, apply `cargo clippy --fix --allow-dirty --allow-staged --all-targets` first, then re-run with `-D warnings`. Hand-fix anything autofix can't handle.
4. **`cargo fmt --all`** — apply, don't just check. CI rejects any diff from `cargo fmt --all -- --check`, including alphabetized imports inside `use foo::{...}` groups. Never skip this step. `cargo build` does NOT run rustfmt.
5. **`cargo fmt --all -- --check`** — verify clean. Should be silent.
6. **`cargo test`** if the repo has any tests (`cargo test` runs zero tests in this repo today, but check `tests/` and Rust unit tests before assuming).

## Phase 3 — Commit, tag, push

```bash
git add -u                                     # stage tracked changes; add specific new files if any
git status                                     # confirm what's staged
git commit -m "<conventional commit message>"  # follow the repo's existing style — see `git log --oneline -10`
git tag -a vX.Y.Z -m "vX.Y.Z — <short summary>"
git push origin <branch>                        # usually main
git push origin vX.Y.Z
```

Notes:
- Use a HEREDOC for the commit message body if it has more than one line — see CLAUDE Code's git-commit guidance.
- Include the `Co-Authored-By: Claude ...` trailer.
- Never use `--no-verify` or `--amend` unless the user explicitly asks.

## Phase 4 — Monitor the pipelines (this is where things go wrong)

Two workflows fire on push: **CI** (on the branch push) and **Release** (on the tag push). Both must succeed. Monitor them in parallel.

```bash
gh run list --limit 5    # find the run IDs for the new CI and Release runs
```

For each run, start a `Monitor` (non-persistent) that polls `gh run view <id> --json status,conclusion` every ~15–30s and exits when status becomes `completed`. Keep working in parallel — events arrive as notifications.

### When CI fails

Pull the failed logs:
```bash
gh run view <id> --log-failed 2>&1 | tail -100
```

Common failures and fixes:

| Symptom in logs | Fix |
|---|---|
| `Diff in .../src/...rs` from `cargo fmt --all -- --check` | Run `cargo fmt --all`, commit as `chore: apply rustfmt`, push to same branch. |
| `error:` from `cargo clippy` | Apply the suggested fix or `cargo clippy --fix`, commit as `chore: appease clippy`, push. |
| Compile error only on a target you didn't test locally (e.g. windows) | Read the error, gate the offending code with `#[cfg(...)]`, push. |
| Network/transient (e.g. registry timeout) | Re-run via `gh run rerun <id>` rather than committing again. |

After fixing, commit on the same branch and push. **Do NOT re-tag** unless the broken thing affects the release binary itself — the existing tag's Release workflow may already be building correctly even if CI on `main` failed (release.yml in this repo runs `cargo build --release` only, not fmt/clippy). Confirm by reading `.github/workflows/release.yml` before deciding.

If the Release workflow itself failed and you need to retag:
```bash
git tag -d vX.Y.Z
git push origin :refs/tags/vX.Y.Z
git tag -a vX.Y.Z <new-sha> -m "..."
git push origin vX.Y.Z
```
Force-retagging is destructive — confirm with the user before doing it if the bad tag has already produced a published artifact.

### Loop until green

Re-monitor after every fix push. Don't declare success until **both** the latest CI run for the branch AND the Release run for the tag have `conclusion=success`.

## Phase 5 — Report

When both are green, report to the user:
- New version + tag.
- Commit SHAs (release commit, plus any follow-up fix commits).
- Link to the Release run (`gh run view <id>` shows the URL).
- Anything notable that was auto-fixed mid-loop (so they know what's in the tree beyond the original change).

## Reusable checks before invoking this skill

If any of these are dirty, surface them and ask before proceeding:
- Uncommitted changes unrelated to the release (`git status`).
- Currently on a non-default branch.
- Existing tag with the proposed version (`git rev-parse vX.Y.Z 2>/dev/null`).
- Previous CI run for `HEAD` is already red.
