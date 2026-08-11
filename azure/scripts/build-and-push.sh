#!/usr/bin/env bash
# Build the app image on this machine and push it to ACR.
#   ./build-and-push.sh <acr-name> [tag] [repo-root]
# Built locally / on the runner, NOT with `az acr build`: ACR Tasks is
# disabled on this subscription (TasksOperationsNotAllowed — known from the
# edgesense deployment, see docs/INFRA.md §7).
set -euo pipefail

ACR="${1:?usage: build-and-push.sh <acr-name> [tag] [repo-root]}"
TAG="${2:-latest}"
ROOT="${3:-$(cd "$(dirname "$0")/../.." && pwd)}"
SERVER="${ACR}.azurecr.io"

az acr login --name "$ACR"
docker build -t "$SERVER/ai-gym-coach:$TAG" -t "$SERVER/ai-gym-coach:latest" "$ROOT"
docker push --all-tags "$SERVER/ai-gym-coach"
echo "pushed $SERVER/ai-gym-coach:$TAG"
