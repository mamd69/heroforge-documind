#!/bin/bash
# PostToolUse Auto-Formatter Hook
# Runs after Claude uses Write or Edit tools

# Read the JSON input from stdin (Claude provides tool context)
INPUT=$(cat)

# Extract the file path from the tool input
# The structure varies by tool, so we check for common patterns
FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // .tool_input.path // empty' 2>/dev/null)

if [ -z "$FILE" ] || [ "$FILE" = "null" ]; then
    # No file path found, exit silently
    exit 0
fi

# Only format if file exists
if [ ! -f "$FILE" ]; then
    exit 0
fi

# Determine file type and format accordingly
case "$FILE" in
    *.py)
        if command -v black &> /dev/null; then
            black "$FILE" --quiet 2>/dev/null
        fi
        ;;
    *.js|*.jsx|*.ts|*.tsx)
        if command -v prettier &> /dev/null; then
            prettier --write "$FILE" > /dev/null 2>&1
        fi
        ;;
    *.md)
        # Markdown: Remove trailing whitespace
        sed -i 's/[[:space:]]*$//' "$FILE" 2>/dev/null
        ;;
esac

exit 0