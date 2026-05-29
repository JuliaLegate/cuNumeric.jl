#!/usr/bin/env python3
"""Version consistency check for pull requests targeting main."""

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(subprocess.run(
    ["git", "rev-parse", "--show-toplevel"],
    capture_output=True, text=True, check=True,
).stdout.strip())

# ── repo-specific config ─────────────────────────────────────────────────────
# To reuse this script for another package (e.g. Legate.jl), change only this block.

PACKAGE_NAME = "cuNumeric"

WRAPPER_VERSION_FILE   = "lib/cunumeric_jl_wrapper/VERSION"
WRAPPER_JLL_COMPAT_KEY = "cunumeric_jl_wrapper_jll"
WRAPPER_SRC_PREFIXES   = (
    "lib/cunumeric_jl_wrapper/src/",
    "lib/cunumeric_jl_wrapper/include/",
    "lib/cunumeric_jl_wrapper/CMakeLists.txt",
)

SUBPKG_TOML_PATH    = "lib/CNPreferences/Project.toml"
SUBPKG_COMPAT_KEY   = "CNPreferences"
SUBPKG_SRC_PREFIXES = (
    "lib/CNPreferences/src/",
)

# ─────────────────────────────────────────────────────────────────────────────


def parse_version(v: str) -> tuple:
    return tuple(int(x) for x in v.strip().split("."))


def version_gt(a: str, b: str) -> bool:
    return parse_version(a) > parse_version(b)


def git_show(ref: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        capture_output=True, text=True, check=True,
        cwd=REPO_ROOT,
    )
    return result.stdout


def changed_files(base_ref: str) -> list:
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        capture_output=True, text=True, check=True,
        cwd=REPO_ROOT,
    )
    return result.stdout.splitlines()


def parse_top_level_version(toml_text: str) -> str:
    for line in toml_text.splitlines():
        stripped = line.strip()
        if re.match(r"^\[", stripped):
            break
        m = re.match(r'^version\s*=\s*"([^"]+)"', stripped)
        if m:
            return m.group(1)
    raise ValueError("Could not find top-level version in TOML")


def parse_compat_section(toml_text: str) -> dict:
    in_compat = False
    result = {}
    for line in toml_text.splitlines():
        stripped = line.strip()
        if re.match(r"^\[", stripped):
            in_compat = stripped == "[compat]"
            continue
        if in_compat and stripped and not stripped.startswith("#"):
            m = re.match(r'^([\w_-]+)\s*=\s*"([^"]+)"', stripped)
            if m:
                result[m.group(1)] = m.group(2)
    return result


def check_package_version(base_ref: str, pr_toml: str, errors: list):
    main_toml = git_show(base_ref, "Project.toml")
    main_ver = parse_top_level_version(main_toml)
    pr_ver = parse_top_level_version(pr_toml)

    print(f"\t[{PACKAGE_NAME}]\tmain={main_ver}\tpr={pr_ver}")
    if not version_gt(pr_ver, main_ver):
        errors.append(
            f"{PACKAGE_NAME} version must be greater than main.\n"
            f"\tmain: {main_ver}  →  pr: {pr_ver}\n"
            f"\tBump the version in Project.toml (patch, minor, or major)."
        )
    else:
        print(f"\t\tOK ({main_ver} → {pr_ver})")


def check_wrapper_version(base_ref: str, changed: list, errors: list):
    src_changed = [f for f in changed if any(f.startswith(p) for p in WRAPPER_SRC_PREFIXES)]
    if not src_changed:
        print("\t[wrapper]\tno source changes — skipping")
        return

    print(f"\t[wrapper]\t{len(src_changed)} source file(s) changed:")
    for f in src_changed:
        print(f"\t\t{f}")

    pr_ver = (REPO_ROOT / WRAPPER_VERSION_FILE).read_text().strip()
    try:
        main_ver = git_show(base_ref, WRAPPER_VERSION_FILE).strip()
    except subprocess.CalledProcessError:
        main_ver = "0.0.0"

    print(f"\t\tVERSION\tmain={main_ver}\tpr={pr_ver}")
    if not version_gt(pr_ver, main_ver):
        errors.append(
            f"{WRAPPER_VERSION_FILE} must be incremented when wrapper source changes.\n"
            f"\tmain: {main_ver}  →  pr: {pr_ver}\n"
            f"\tBump {WRAPPER_VERSION_FILE}."
        )
    else:
        print(f"\t\tOK VERSION ({main_ver} → {pr_ver})")


def check_wrapper_compat_sync(pr_toml: str, errors: list):
    wrapper_ver = (REPO_ROOT / WRAPPER_VERSION_FILE).read_text().strip()
    compat_ver = parse_compat_section(pr_toml).get(WRAPPER_JLL_COMPAT_KEY)

    lhs = WRAPPER_VERSION_FILE
    rhs = f"Project.toml [compat] {WRAPPER_JLL_COMPAT_KEY}"
    w = max(len(lhs), len(rhs))
    print(f"\t[wrapper compat sync]")
    print(f"\t\t{lhs:<{w}} = {wrapper_ver}")
    print(f"\t\t{rhs:<{w}} = {compat_ver}")
    if compat_ver is None:
        errors.append(
            f"{WRAPPER_JLL_COMPAT_KEY} not found in Project.toml [compat]."
        )
    elif compat_ver != wrapper_ver:
        errors.append(
            f"Project.toml [compat] {WRAPPER_JLL_COMPAT_KEY}={compat_ver} does not match {WRAPPER_VERSION_FILE}={wrapper_ver}.\n"
            f"\tSet {WRAPPER_JLL_COMPAT_KEY} = \"{wrapper_ver}\" in Project.toml [compat]."
        )
    else:
        print(f"\t\tOK")


def check_subpkg_version(base_ref: str, pr_toml: str, changed: list, errors: list):
    src_changed = [f for f in changed if any(f.startswith(p) for p in SUBPKG_SRC_PREFIXES)]
    if not src_changed:
        print(f"\t[{SUBPKG_COMPAT_KEY}]\tno source changes — skipping")
        return

    print(f"\t[{SUBPKG_COMPAT_KEY}]\t{len(src_changed)} source file(s) changed:")
    for f in src_changed:
        print(f"\t\t{f}")

    pr_subpkg_toml = (REPO_ROOT / SUBPKG_TOML_PATH).read_text()
    try:
        main_subpkg_toml = git_show(base_ref, SUBPKG_TOML_PATH)
    except subprocess.CalledProcessError:
        main_subpkg_toml = 'version = "0.0.0"'

    pr_ver = parse_top_level_version(pr_subpkg_toml)
    main_ver = parse_top_level_version(main_subpkg_toml)

    print(f"\t\tversion\tmain={main_ver}\tpr={pr_ver}")
    if not version_gt(pr_ver, main_ver):
        errors.append(
            f"{SUBPKG_TOML_PATH} version must be incremented when {SUBPKG_COMPAT_KEY} source changes.\n"
            f"\tmain: {main_ver}  →  pr: {pr_ver}\n"
            f"\tBump the version in {SUBPKG_TOML_PATH}."
        )
    else:
        print(f"\t\tOK version ({main_ver} → {pr_ver})")

    compat_ver = parse_compat_section(pr_toml).get(SUBPKG_COMPAT_KEY)
    if compat_ver is None:
        errors.append(
            f"{SUBPKG_COMPAT_KEY} not found in Project.toml [compat]."
        )
    elif compat_ver != pr_ver:
        errors.append(
            f"Project.toml [compat] {SUBPKG_COMPAT_KEY}={compat_ver} does not match {SUBPKG_TOML_PATH} version={pr_ver}.\n"
            f"\tSet {SUBPKG_COMPAT_KEY} = \"{pr_ver}\" in Project.toml [compat]."
        )
    else:
        print(f"\t\tOK compat {SUBPKG_COMPAT_KEY}={compat_ver}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-ref", default="origin/main",
                        help="Git ref to compare against (default: origin/main)")
    args = parser.parse_args()

    base_ref = args.base_ref
    errors = []

    pr_toml = (REPO_ROOT / "Project.toml").read_text()
    changed = changed_files(base_ref)

    print("Version check")
    print("─" * 60)
    check_package_version(base_ref, pr_toml, errors)
    check_wrapper_version(base_ref, changed, errors)
    check_wrapper_compat_sync(pr_toml, errors)
    check_subpkg_version(base_ref, pr_toml, changed, errors)
    print("─" * 60)

    if errors:
        print(f"\nFAILED — {len(errors)} error(s):\n")
        for i, e in enumerate(errors, 1):
            print(f"\t{i}. {e}\n")
        sys.exit(1)
    else:
        print("\nPASSED")


if __name__ == "__main__":
    main()
