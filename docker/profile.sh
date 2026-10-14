# shellcheck shell=sh
# Keep interactive login shells consistent with the container entrypoint.
export PATH="/opt/conda/bin:/usr/local/julia/bin:/usr/local/bin:${PATH}"

if [ -r /etc/.env ]; then
    # shellcheck disable=SC1091
    . /etc/.env
fi
