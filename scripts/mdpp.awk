# Supports:
#   !INCLUDE_IF_EXISTS "relative/path.md"
# Path is resolved relative to: base (directory of the input .md)

BEGIN { if (base == "") base = "." }

match($0, /^[[:space:]]*!INCLUDE_IF_EXISTS[[:space:]]+"([^"]+)"[[:space:]]*$/, m) {
  path = base "/" m[1]

  # If file exists, print it; otherwise print nothing (silently)
  while ((getline line < path) > 0) print line
  close(path)
  next
}

{ print }
