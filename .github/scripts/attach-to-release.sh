#!/usr/bin/env bash
# Attaches build outputs to the GitHub release of a tag:
#   .github/scripts/attach-to-release.sh <tag> <file>...
#
# Used by the installer workflows (release-windows.yml, release-macos.yml, release-linux.yml).
# They never create the release: release.yml does, once, when the tag is pushed (several jobs
# creating it at the same moment made duplicate releases). So this waits for it - normally it is
# there long before a build finishes. Needs GH_TOKEN (and GH_REPO, or a checkout of the repository).
set -euo pipefail

tag="$1"; shift
[[ $# -gt 0 ]] || { echo "usage: $0 <tag> <file>..." >&2; exit 2; }

for attempt in $(seq 1 60); do   # up to 10 minutes
    count="$(gh api "repos/{owner}/{repo}/releases?per_page=100" \
        --jq "[.[] | select(.tag_name == \"$tag\")] | length")"
    if [[ "$count" -ge 1 ]]; then
        break
    fi
    if [[ "$attempt" -eq 60 ]]; then
        echo "::error::No GitHub release for $tag after 10 minutes - did the Release workflow (release.yml) run?" >&2
        exit 1
    fi
    echo "Waiting for the release of $tag (created by release.yml)..."
    sleep 10
done
if [[ "$count" -gt 1 ]]; then
    echo "::warning::$count releases share the tag $tag - delete the empty duplicate on GitHub's Releases page"
fi

gh release upload "$tag" "$@" --clobber
echo "Attached to $tag: $*"
