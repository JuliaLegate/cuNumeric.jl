#!/usr/bin/env bash
set -e

# shellcheck disable=SC1091
source /etc/profile.d/cunumeric.sh

if [[ $# -eq 0 ]]; then
    set -- /bin/bash
elif [[ $1 == -* ]]; then
    set -- /bin/bash "$@"
fi

exec "$@"
