#!/usr/bin/env sh
# Quarkon installer for Linux and macOS.
#
# Installs Quarkon into ~/.local/share/quarkon, links the `quarkon` command into
# ~/.local/bin, adds a desktop entry on Linux, and starts the graphical wizard.
#
#   sh install.sh            # install and launch
#   sh install.sh --no-launch
#
# No administrator rights are required. Nothing is written outside the user's
# home directory.

set -eu

APP_DIR="${HOME}/.local/share/quarkon"
BIN_DIR="${HOME}/.local/bin"
VENV_DIR="${APP_DIR}/.venv"
SOURCE_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
LAUNCH=1

for arg in "$@"; do
    case "$arg" in
        --no-launch) LAUNCH=0 ;;
        --help|-h) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

step() { printf '\033[36m==>\033[0m %s\n' "$1"; }
ok()   { printf '    %s\n' "$1"; }
fail() {
    printf '\033[31mProblem:\033[0m %s\n' "$1" >&2
    printf 'Fix the problem, then run this script again. For a diagnosis run: quarkon doctor\n' >&2
    exit 1
}

step "Checking Python"
PYTHON=""
for candidate in python3 python3.13 python3.12 python3.11 python3.10 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            PYTHON="$candidate"
            break
        fi
    fi
done
[ -n "$PYTHON" ] || fail "Python 3.10 or newer is required. Install it with your package manager (for example: sudo apt install python3.12 python3.12-venv)."
ok "$($PYTHON --version 2>&1)"

step "Preparing folders"
mkdir -p "$APP_DIR" "$BIN_DIR"
ok "$APP_DIR"

if [ ! -x "$VENV_DIR/bin/python" ]; then
    step "Creating a private environment (this takes a minute)"
    "$PYTHON" -m venv "$VENV_DIR" || fail "Could not create the virtual environment (on Debian/Ubuntu install python3-venv)."
fi

step "Installing Quarkon"
"$VENV_DIR/bin/python" -m pip install --upgrade pip --quiet
if [ -f "$SOURCE_DIR/pyproject.toml" ]; then
    "$VENV_DIR/bin/python" -m pip install --quiet "$SOURCE_DIR"
else
    "$VENV_DIR/bin/python" -m pip install --quiet quarkon
fi || fail "Installation failed. Check the internet connection or the company proxy."
ok "Installed"

step "Linking the command"
ln -sf "$VENV_DIR/bin/quarkon" "$BIN_DIR/quarkon"
ok "$BIN_DIR/quarkon"

case "$(uname -s)" in
    Linux)
        step "Creating the desktop entry"
        APPS_DIR="${HOME}/.local/share/applications"
        mkdir -p "$APPS_DIR"
        cat > "${APPS_DIR}/quarkon.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Quarkon
Comment=Turn a reasoning model into a local model your company can run itself
Exec=${BIN_DIR}/quarkon launcher
Terminal=false
Categories=Development;Utility;
EOF
        ok "${APPS_DIR}/quarkon.desktop"
        ;;
esac

case ":${PATH}:" in
    *":${BIN_DIR}:"*) ;;
    *)
        printf '\nNote: add %s to your PATH to use the "quarkon" command:\n' "$BIN_DIR"
        printf '  echo '"'"'export PATH="$HOME/.local/bin:$PATH"'"'"' >> ~/.profile\n'
        ;;
esac

if [ "$LAUNCH" -eq 1 ]; then
    step "Starting Quarkon"
    "$VENV_DIR/bin/quarkon" launcher
else
    printf '\nInstallation complete. Start with: %s/quarkon launcher\n' "$BIN_DIR"
fi
