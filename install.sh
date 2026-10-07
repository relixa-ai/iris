#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' 2>/dev/null; then
            PY="$candidate"
            break
        fi
    done
    PY="${PY:-python3}"
fi
SPARK=0
for arg in "$@"; do
    case "$arg" in
        --spark) SPARK=1 ;;
        *) printf 'unknown option: %s (the one option is --spark)\n' "$arg" >&2; exit 2 ;;
    esac
done

fail() {
    printf '\nERROR: %s\n' "$1" >&2
    printf '       %s\n' "$2" >&2
    exit 1
}

echo '[1/6] checking Python'
command -v "$PY" >/dev/null 2>&1 || fail \
    "no '$PY' on PATH." \
    "Install Python 3.10, 3.11, or 3.12 from https://www.python.org/downloads/, or point this script at one: PYTHON=/path/to/python3.11 bash install.sh"
PY_VER="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
"$PY" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)' || fail \
    "Python $PY_VER is not supported." \
    "iris needs Python >= 3.10 and < 3.13 (numba and numpy publish no wheels for 3.13 yet). Install one of those and re-run: PYTHON=/path/to/python3.12 bash install.sh"
echo "      OK: Python $PY_VER at $(command -v "$PY")"

echo '[2/6] checking the build tools'
if [ "$(uname)" = Darwin ] && [ "$("$PY" -c 'import platform; print(platform.machine())')" = x86_64 ]; then
    xcode-select -p >/dev/null 2>&1 || fail \
        "the Xcode Command Line Tools are not installed." \
        "On a Mac with an Intel processor, one dependency, py_stringmatching, is built from source. Install the tools with: xcode-select --install"
    echo "      OK: the Xcode Command Line Tools"
else
    echo "      skipped: every dependency installs from a prebuilt package here"
fi

echo '[3/6] checking Java'
if [ "$SPARK" = 0 ]; then
    echo '      skipped: the local engine needs no Java (bash install.sh --spark adds Spark)'
else
JAVA_BIN="java"
if ! command -v java >/dev/null 2>&1; then
    if [ -n "${JAVA_HOME:-}" ] && [ -x "$JAVA_HOME/bin/java" ]; then
        JAVA_BIN="$JAVA_HOME/bin/java"
    else
        fail "no Java runtime found." \
             "Spark needs one. Install a JDK 17 build from https://adoptium.net/temurin/releases/?version=17, or on Debian or Ubuntu: apt-get install openjdk-17-jdk-headless"
    fi
fi
JAVA_RAW="$("$JAVA_BIN" -version 2>&1 | head -1 | sed 's/.*version "\([^"]*\)".*/\1/')"
JAVA_MAJOR="$(printf '%s' "$JAVA_RAW" | awk -F'[."]' '{ if ($1 == 1) print $2; else print $1 }')"
case "$JAVA_MAJOR" in
    8|11|17) ;;
    *) fail "Java $JAVA_RAW is not supported by Spark 3.5." \
            "Spark 3.5.x supports Java 8, 11, and 17 only. On Java 21+ the JVM blocks the memory access Arrow needs, Spark silently stops using Arrow, and the failure surfaces much later as an unrelated type error. Install JDK 17 from https://adoptium.net/temurin/releases/?version=17 and set JAVA_HOME to it." ;;
esac
echo "      OK: Java $JAVA_RAW"
fi

echo '[4/6] creating .venv'
"$PY" -c 'import ensurepip, venv' 2>/dev/null || fail \
    "Python $PY_VER cannot create a virtual environment: its venv module is not installed." \
    "On Debian or Ubuntu: sudo apt-get install python3-venv (or python$PY_VER-venv)"
"$PY" -m venv .venv
echo "      OK: .venv/ created with Python $PY_VER"

echo '[5/6] installing iris and its dependencies with pip, from PyPI (a few minutes)'
.venv/bin/python -m pip install --upgrade pip setuptools wheel >/dev/null
if [ "$SPARK" = 1 ]; then TARGET='.[spark]'; else TARGET='.'; fi
.venv/bin/python -m pip install -e "$TARGET" -c constraints.txt >/dev/null
echo "      OK: iris $(.venv/bin/python -c 'import iris; print(iris.__version__)') installed"

echo '[6/6] smoke test'
.venv/bin/iris smoke
if [ "$SPARK" = 1 ]; then
    .venv/bin/iris smoke --engine spark
fi

cat <<'EOF'

  ┌────────────────────────────────────────────────────────┐
  │  iris is installed.                                    │
  │                                                        │
  │  Turn the environment on:                              │
  │      source .venv/bin/activate                         │
  │                                                        │
  │  Run the first example:                                │
  │      python examples/run_passive.py                    │
  │                                                        │
  │  The tutorial is doc/tutorial.md.                      │
  └────────────────────────────────────────────────────────┘
EOF
