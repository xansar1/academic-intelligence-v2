#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO='xansar1/academic-intelligence-v2'
REMOTE="https://github.com/$REPO.git"
if ! git ls-remote "$REMOTE" >/dev/null 2>&1; then
  echo "First create an EMPTY PUBLIC GitHub repository at https://github.com/new"
  echo "Owner: xansar1 | Name: academic-intelligence-v2 | Do not initialize README"
  echo "Then rerun this script."
  exit 2
fi
if [ ! -d .git ]; then git init -b main; fi
OLD="$(git remote get-url origin 2>/dev/null || true)"
case "$OLD" in
  '' ) git remote add origin "$REMOTE";;
  "$REMOTE"|"git@github.com:$REPO.git") ;;
  * ) echo "Refusing to push: unexpected origin $OLD"; exit 3;;
esac
git add .github .gitignore .streamlit/config.toml academic_v2_demo.py academic_v2/ requirements.txt requirements-dev.txt scripts/ tests/ README.md
if ! git diff --cached --quiet; then git commit -m 'Ship Academic Intelligence V2.1 safe synthetic demo'; fi
git push -u origin main
printf '
Published: https://github.com/%s
' "$REPO"
