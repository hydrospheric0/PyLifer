#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/hydrospheric0/PyLifer.git"
SOURCE_BRANCH="main"
SOURCE_REMOTE="origin"
DEFAULT_UPDATE_MESSAGE="Update PyLifer"
BUMP_KIND="patch"
RELEASE_VERSION=""
STAGE_MODE="all"

cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
  cat <<'EOF'
Usage:
  ./pushit.sh [message]
  ./pushit.sh --patch [message]
  ./pushit.sh --minor [message]
  ./pushit.sh --major [message]
  ./pushit.sh --release <version> [message]
  ./pushit.sh --tracked [message]

Options:
  --patch            Increment x.y.z by 0.0.1 (default).
  --minor            Increment x.y.z by 0.1.0.
  --major            Increment x.y.z by 1.0.0.
  --release <ver>    Set an explicit semantic version.
  --tracked          Stage tracked files only (git add -u).
  -h, --help         Show this help.

The script will:
  1. Ensure the git remote is correct.
  2. Auto-bump the VERSION file (or use --release to set explicitly).
  3. Commit and push main.
  4. Create and push a matching git tag.
EOF
}

normalize_git_url() {
  local url="$1"
  url="${url%.git}"
  if [[ "$url" =~ ^git@github\.com:(.*)$ ]]; then
    url="https://github.com/${BASH_REMATCH[1]}"
  fi
  echo "$url"
}

ensure_remote_url() {
  local remote_name="$1"
  local expected_url="$2"
  local current_url=""

  if git remote get-url "$remote_name" >/dev/null 2>&1; then
    current_url="$(git remote get-url "$remote_name")"
    if [[ "$(normalize_git_url "$current_url")" != "$(normalize_git_url "$expected_url")" ]]; then
      echo "Updating remote $remote_name -> $expected_url"
      git remote set-url "$remote_name" "$expected_url"
    fi
  else
    echo "Adding remote $remote_name -> $expected_url"
    git remote add "$remote_name" "$expected_url"
  fi
}

is_valid_semver() {
  [[ "$1" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]
}

get_current_version() {
  local v=""
  if [[ -f VERSION ]]; then
    v="$(tr -d '[:space:]' < VERSION)"
  fi
  if ! is_valid_semver "$v"; then
    echo "ERROR: Could not determine a valid current version from VERSION file (expected x.y.z, got: '${v:-empty}')." >&2
    exit 1
  fi
  echo "$v"
}

bump_version() {
  local version="$1"
  local kind="$2"
  local major minor patch
  IFS='.' read -r major minor patch <<< "$version"

  case "$kind" in
    patch) patch=$((patch + 1)) ;;
    minor) minor=$((minor + 1)); patch=0 ;;
    major) major=$((major + 1)); minor=0; patch=0 ;;
    *) echo "ERROR: Unknown bump kind: $kind" >&2; exit 1 ;;
  esac

  echo "${major}.${minor}.${patch}"
}

msg_parts=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --patch)   BUMP_KIND="patch"; shift ;;
    --minor)   BUMP_KIND="minor"; shift ;;
    --major)   BUMP_KIND="major"; shift ;;
    --release)
      RELEASE_VERSION="${2:-}"
      if [[ -z "$RELEASE_VERSION" ]]; then
        echo "ERROR: --release requires a version." >&2; exit 1
      fi
      if ! is_valid_semver "$RELEASE_VERSION"; then
        echo "ERROR: --release must be in x.y.z format." >&2; exit 1
      fi
      shift 2
      ;;
    --tracked) STAGE_MODE="tracked"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; msg_parts+=("$@"); break ;;
    *) msg_parts+=("$1"); shift ;;
  esac
done

message="${msg_parts[*]:-$DEFAULT_UPDATE_MESSAGE}"

if [[ ! -d .git ]]; then
  echo "Initializing git repository..."
  git init
  git branch -M "$SOURCE_BRANCH"
fi

ensure_remote_url "$SOURCE_REMOTE" "$REPO_URL"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "$SOURCE_BRANCH" ]]; then
  echo "ERROR: pushit.sh must be run from the $SOURCE_BRANCH branch." >&2
  echo "Current branch: $CURRENT_BRANCH" >&2
  exit 1
fi

current_version="$(get_current_version)"

if [[ -n "$RELEASE_VERSION" ]]; then
  next_version="$RELEASE_VERSION"
  echo "Using explicit version: $next_version"
else
  next_version="$(bump_version "$current_version" "$BUMP_KIND")"
  echo "Bumping version: $current_version -> $next_version"
fi

printf '%s\n' "$next_version" > VERSION

if [[ "$STAGE_MODE" == "tracked" ]]; then
  git add -u
else
  git add -A
fi

if ! git diff --cached --quiet; then
  git commit -m "$message"
else
  echo "No staged changes to commit."
fi

if git ls-remote --exit-code --heads "$SOURCE_REMOTE" "$SOURCE_BRANCH" >/dev/null 2>&1; then
  git pull --rebase "$SOURCE_REMOTE" "$SOURCE_BRANCH"
fi

echo "Pushing to $SOURCE_REMOTE/$SOURCE_BRANCH..."
git push -u "$SOURCE_REMOTE" HEAD:"$SOURCE_BRANCH"

release_tag="v${next_version}"
if git rev-parse -q --verify "refs/tags/${release_tag}" >/dev/null 2>&1; then
  echo "Tag ${release_tag} already exists; skipping."
else
  git tag -a "$release_tag" -m "Release ${next_version}"
fi

echo "Pushing tag ${release_tag}..."
git push "$SOURCE_REMOTE" "$release_tag"

echo
echo "Done."
echo "https://github.com/hydrospheric0/PyLifer/tree/$SOURCE_BRANCH"
