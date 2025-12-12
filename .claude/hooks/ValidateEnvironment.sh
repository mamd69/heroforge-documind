#!/bin/bash
# Environment Validation Script
# Called by SessionStart hook to validate project environment

echo "🔍 Environment Validator Running..."
echo ""

# Check 1: Verify we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ ERROR: Not in project root directory"
    exit 2  # Exit code 2 = blocking error shown to Claude
fi
echo "✅ In project root directory"

# Check 2: Verify required directories exist
required_dirs=(".claude" "docs" "src")
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "⚠️  WARNING: $dir directory missing - creating..."
        mkdir -p "$dir"
    else
        echo "✅ $dir directory exists"
    fi
done

# Check 3: Verify .env file exists (if .env.example exists)
if [ -f ".env.example" ] && [ ! -f ".env" ]; then
    echo "⚠️  WARNING: .env.example exists but .env missing"
    echo "   Run: cp .env.example .env"
fi

# Check 4: Verify git status
if command -v git &> /dev/null; then
    if [ -n "$(git status --porcelain)" ]; then
        echo "⚠️  WARNING: Uncommitted changes detected"
        echo "   Files changed: $(git status --short | wc -l)"
    else
        echo "✅ Git working tree clean"
    fi
fi

# Check 5: Log the validation
timestamp=$(date +"%Y-%m-%d %H:%M:%S")
echo "$timestamp - Environment validated" >> .claude/hooks/session.log

echo ""
echo "✅ Environment validation complete - ready to proceed"