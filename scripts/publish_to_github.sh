#!/usr/bin/env bash
# Helper: stage, commit, and push selected project files to a remote GitHub repo via SSH.
# Usage: ./scripts/publish_to_github.sh
# The script will prompt for the remote SSH URL (e.g. git@github.com:ameerhamzarashid/dissertation-fl-offloading-ns3.git)
# and the branch name. It will create a local commit "publish: scripts + results + artifacts"
# and push to the remote. This script does NOT include the virtual environment or large NS-3 build.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT=$(pwd)

read -rp "Remote SSH repo URL (ssh) [git@github.com:ameerhamzarashid/dissertation-fl-offloading-ns3.git]: " REMOTE_URL
REMOTE_URL=${REMOTE_URL:-git@github.com:ameerhamzarashid/dissertation-fl-offloading-ns3.git}
read -rp "Remote branch [main]: " BRANCH
BRANCH=${BRANCH:-main}

echo "Preparing to push to ${REMOTE_URL} on branch ${BRANCH}"

# Create a temporary worktree so we don't disturb current branch state (safe staging).
TMP_BRANCH="publish-scripts-$(date +%s)"

echo "Creating temporary branch ${TMP_BRANCH}..."
# Ensure repo exists
if ! git rev-parse --git-dir > /dev/null 2>&1; then
  echo "Error: not a git repository. Initialize one first with 'git init' and add a remote.'"
  exit 1
fi

git checkout -b "${TMP_BRANCH}"

# Ensure .gitignore covers large items
cat >> .gitignore <<'GITIGNORE'
# Local large artifacts - auto-added by publish helper
.venv/
.venv
venv/
ns-allinone-*/
ns-allinone-*
ns3_allinone*/
ns3_allinone*/
ns3_module/build/
data/raw_logs/
GITIGNORE

# Stage desired folders and files. This list can be adjusted.
# Exclude venv and NS-3 builds explicitly.

echo "Staging files..."
# Always include scripts/, python_fl/, configs/, results/, data/plots/, experiments/, ns3_module include/src (but exclude builds)

git add --all --ignore-errors scripts/ python_fl/ configs/ results/ data/plots/ experiments/ docs/ ns3_module/include/ ns3_module/src/ tests/ || true

# Remove any accidentally staged virtualenv or build files
git reset -- scripts/.venv || true

# Commit
COMMIT_MSG="publish: scripts, results, plots, and code artifacts"

git commit -m "$COMMIT_MSG" || {
  echo "Nothing to commit. Cleaning up temporary branch and exiting."
  git checkout -
  git branch -D "${TMP_BRANCH}" || true
  exit 0
}

# Add remote if not present
if ! git remote | grep -q origin; then
  git remote add origin "$REMOTE_URL"
fi

echo "Pushing to remote..."
# Push the temporary branch to the requested branch on remote
git push -u origin "${TMP_BRANCH}:${BRANCH}"

# Cleanup local temporary branch and return to previous branch (if possible)
PREV_BRANCH=$(git reflog | awk 'NR==2{print $NF}') || true

echo "Switching back to previous branch"
git checkout - || true

echo "Deleting local temporary branch ${TMP_BRANCH}"
git branch -D "${TMP_BRANCH}" || true

echo "Publish complete. Verify the remote repository to confirm files were pushed."

# Exit cleanly
exit 0
