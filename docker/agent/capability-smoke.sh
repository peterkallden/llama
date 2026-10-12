#!/usr/bin/env bash
set -euo pipefail

# This is an image contract check, not a release-version check.  Keep it
# executable during the image build so a missing external tool fails early.
for command_name in \
    git cmake ninja gcc g++ clang clangd gdb pandoc python3 tesseract mutool gs nginx openssl; do
    command -v "$command_name" >/dev/null
done

git --version
cmake --version | head -n 1
ninja --version
gcc --version | head -n 1
g++ --version | head -n 1
clang --version | head -n 1
clangd --version | head -n 1
gdb --version | head -n 1
pandoc --version | head -n 1
python3 --version
tesseract --version | head -n 1
mutool -v 2>&1 | head -n 1
gs --version | head -n 1
nginx -v 2>&1
openssl version
test -s /opt/llama-agent-web/dist/index.html

# Exercise the two small local development paths that do not need project
# sources or a model: compile a C++ translation unit and convert Markdown.
smoke_dir="$(mktemp -d)"
trap 'rm -rf "$smoke_dir"' EXIT
cat > "$smoke_dir/smoke.cpp" <<'EOF'
int main() { return 0; }
EOF
g++ -std=c++17 -fsyntax-only "$smoke_dir/smoke.cpp"
printf '# llama-agent image\n' > "$smoke_dir/input.md"
pandoc "$smoke_dir/input.md" -f markdown -t plain -o "$smoke_dir/output.txt"
grep -Fq 'llama-agent image' "$smoke_dir/output.txt"
