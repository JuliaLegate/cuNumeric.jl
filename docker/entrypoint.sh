#!/usr/bin/env bash
set -e

source /etc/.env

if [[ $# -eq 0 ]]; then
    set -- /bin/bash
elif [[ $1 == -* ]]; then
    set -- /bin/bash "$@"
fi

exec "$@"
