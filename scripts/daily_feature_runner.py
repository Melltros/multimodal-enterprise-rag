"""
Script: daily_feature_runner.py
Description: Automated pipeline utility for testing, attributing, committing,
             and publishing daily features strictly under Aanshmita's identity.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Enforce strict attribution constants
ALLOWED_AUTHOR_NAME = "Aanshmita"
ALLOWED_AUTHOR_EMAIL = "129788305+Aanshmita@users.noreply.github.com"
DISALLOWED_SUBSTRINGS = ["melltros", "madhav", "kelodiamadhav"]


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
    """Verifies that Git identity is strictly set to Aanshmita with zero Melltros traces."""
    print("🔍 Verifying git identity and attribution safeguards...")

    # Check local git config
    name_res = run_cmd(["git", "config", "user.name"])
    email_res = run_cmd(["git", "config", "user.email"])

    author_name = name_res.stdout.strip()
    author_email = email_res.stdout.strip()

    print(f"👤 Git user.name:  {author_name}")
    print(f"📧 Git user.email: {author_email}")

    for banned in DISALLOWED_SUBSTRINGS:
        if banned in author_name.lower() or banned in author_email.lower():
            print(f"🚨 FATAL: Disallowed substring '{banned}' detected in git author settings!")
            return False

    if author_name != ALLOWED_AUTHOR_NAME or author_email != ALLOWED_AUTHOR_EMAIL:
        print("⚠️ Current git config does not match target. Automatically configuring...")
        run_cmd(["git", "config", "user.name", ALLOWED_AUTHOR_NAME])
        run_cmd(["git", "config", "user.email", ALLOWED_AUTHOR_EMAIL])

    print("✅ Git identity verified: 100% attributed to Aanshmita.")
    return True


def run_tests() -> bool:
    """Runs full unit test discovery."""
    print("🧪 Running test suite...")
    res = run_cmd([sys.executable, "-m", "unittest", "discover", "tests"])
    return res.returncode == 0


def run_smoke_test() -> bool:
    """Runs CLI smoke test in mock mode."""
    print("🚀 Running CLI pipeline smoke test...")
    res = run_cmd([sys.executable, "main.py", "--mock", "--retrieval-mode", "hybrid"])
    return res.returncode == 0


def build_env() -> dict:
    """Creates environment with forced author attribution."""
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
        print("❌ Smoke test failed. Commit aborted.")
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
        print("🚀 Pushing to origin main...")
        push_res = run_cmd(["git", "push", "origin", "main"], env=env)
        return push_res.returncode == 0

    return True


def main():
    parser = argparse.ArgumentParser(description="Daily Feature Pipeline Runner")
    parser.add_argument("--verify", action="store_true", help="Verify git identity")
    parser.add_argument("--test", action="store_true", help="Run tests and smoke tests")
    parser.add_argument("--commit", type=str, help="Commit message to use")
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
