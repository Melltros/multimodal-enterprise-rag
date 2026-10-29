"""
Script: daily_feature_runner.py
Description: Automated pipeline utility for testing, attributing, committing,
             and publishing daily features strictly under Melltros's identity.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Enforce strict attribution constants
ALLOWED_AUTHOR_NAME = "Melltros"
ALLOWED_AUTHOR_EMAIL = "minecraftkelodia@gmail.com"


def run_cmd(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    """Executes a command and returns the completed process."""
    cmd_str = " ".join(cmd)
    print(f"⚙️ Running: {cmd_str}")
    res = subprocess.run(
        cmd,
        cwd=str(cwd or Path.cwd()),
        text=True,
        capture_output=True,
        env=env
    )
    if res.returncode != 0:
        print(f"❌ Command failed (code {res.returncode}):\n{res.stderr.strip()}")
    else:
        if res.stdout.strip():
            print(res.stdout.strip())
    return res


def verify_attribution() -> bool:
    """Verifies that Git identity is strictly set to Melltros."""
    print("🔍 Verifying git identity and attribution safeguards...")

    # Check local git config
    name_res = run_cmd(["git", "config", "user.name"])
    email_res = run_cmd(["git", "config", "user.email"])

    author_name = name_res.stdout.strip()
    author_email = email_res.stdout.strip()

    print(f"👤 Git user.name:  {author_name}")
    print(f"📧 Git user.email: {author_email}")

    if author_name != ALLOWED_AUTHOR_NAME or author_email != ALLOWED_AUTHOR_EMAIL:
        print("⚠️ Current git config does not match target. Automatically configuring...")
        run_cmd(["git", "config", "user.name", ALLOWED_AUTHOR_NAME])
        run_cmd(["git", "config", "user.email", ALLOWED_AUTHOR_EMAIL])

    print("✅ Git identity verified: 100% attributed to Melltros.")
    return True


def run_tests() -> bool:
    """Runs full unit test discovery."""
    print("🧪 Running unit and integration test suite...")
    res = run_cmd([sys.executable, "-m", "unittest", "discover", "tests"])
    return res.returncode == 0


def run_smoke_test() -> bool:
    """Runs CLI smoke tests in mock mode across retrieval and reranking configurations."""
    print("🚀 Running CLI pipeline smoke tests...")
    modes = [
        ["main.py", "--mock", "--retrieval-mode", "hybrid", "--rerank"],
        ["main.py", "--mock", "--retrieval-mode", "dense"],
        ["main.py", "--mock", "--retrieval-mode", "bm25"],
    ]
    for mode_args in modes:
        res = run_cmd([sys.executable] + mode_args)
        if res.returncode != 0:
            return False
    return True


def build_env() -> dict:
    """Creates execution environment strictly forcing Melltros author & committer attribution."""
    env = os.environ.copy()
    env["GIT_AUTHOR_NAME"] = ALLOWED_AUTHOR_NAME
    env["GIT_AUTHOR_EMAIL"] = ALLOWED_AUTHOR_EMAIL
    env["GIT_COMMITTER_NAME"] = ALLOWED_AUTHOR_NAME
    env["GIT_COMMITTER_EMAIL"] = ALLOWED_AUTHOR_EMAIL
    return env


def commit_and_push(commit_msg: str, push: bool = False) -> bool:
    """Stages all changes, creates a verified commit, and optionally pushes."""
    if not verify_attribution():
        return False

    if not run_tests():
        print("❌ Tests failed. Commit aborted.")
        return False

    if not run_smoke_test():
        print("❌ Smoke tests failed. Commit aborted.")
        return False

    env = build_env()

    # Stage changes
    stage_res = run_cmd(["git", "add", "."], env=env)
    if stage_res.returncode != 0:
        return False

    # Check status
    status_res = run_cmd(["git", "status", "--porcelain"], env=env)
    if not status_res.stdout.strip():
        print("ℹ️ No changes detected to commit.")
        return True

    # Commit with explicit author flag
    author_flag = f"{ALLOWED_AUTHOR_NAME} <{ALLOWED_AUTHOR_EMAIL}>"
    commit_res = run_cmd([
        "git", "commit",
        f"--author={author_flag}",
        "-m", commit_msg
    ], env=env)

    if commit_res.returncode != 0:
        return False

    print(f"🎉 Commit created successfully: '{commit_msg}'")

    if push:
        print("🚀 Pushing to origin main as Melltros...")
        push_res = run_cmd(["git", "push", "origin", "main"], env=env)
        if push_res.returncode != 0:
            print("⚠️ Push failed. Please ensure Melltros has write permissions on the remote repository.")
            return False
        print("✅ Successfully pushed to remote origin as Melltros.")
        return True

    return True


def main():
    parser = argparse.ArgumentParser(description="Daily Feature Pipeline Runner")
    parser.add_argument("--verify", action="store_true", help="Verify git identity and attribution safeguards")
    parser.add_argument("--test", action="store_true", help="Run tests and smoke tests")
    parser.add_argument("--commit", type=str, help="Commit message to use for staged changes")
    parser.add_argument("--push", action="store_true", help="Push commit to remote origin")

    args = parser.parse_args()

    if args.verify:
        if not verify_attribution():
            sys.exit(1)

    if args.test:
        if not (run_tests() and run_smoke_test()):
            sys.exit(1)

    if args.commit:
        success = commit_and_push(args.commit, push=args.push)
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
