#!/bin/bash
# DocuMind Pipeline Orchestrator - Demonstration Script
# This script demonstrates all capabilities of the orchestrator

set -e  # Exit on error

echo "================================================================"
echo "   DocuMind Pipeline Orchestrator - Feature Demonstration"
echo "================================================================"
echo ""

# Change to the pipeline directory
cd "$(dirname "$0")"

echo "📂 Current directory: $(pwd)"
echo ""

# Demo 1: Single File Processing
echo "═══════════════════════════════════════════════════════════════"
echo "Demo 1: Single File Processing"
echo "═══════════════════════════════════════════════════════════════"
python orchestrate.py ../../../demo-docs/sample1.md
echo ""

# Demo 2: Multiple File Processing
echo "═══════════════════════════════════════════════════════════════"
echo "Demo 2: Multiple File Processing"
echo "═══════════════════════════════════════════════════════════════"
python orchestrate.py ../../../demo-docs/sample1.md ../../../demo-docs/sample2.txt ../../../demo-docs/sample3.md
echo ""

# Demo 3: Directory Processing
echo "═══════════════════════════════════════════════════════════════"
echo "Demo 3: Directory Processing (Recursive)"
echo "═══════════════════════════════════════════════════════════════"
python orchestrate.py -d ../../../demo-docs/
echo ""

# Demo 4: High Parallelism
echo "═══════════════════════════════════════════════════════════════"
echo "Demo 4: High Parallelism (20 concurrent documents)"
echo "═══════════════════════════════════════════════════════════════"
python orchestrate.py -d ../../../demo-docs/ --max-parallel 20
echo ""

# Demo 5: Quiet Mode with JSON Output
echo "═══════════════════════════════════════════════════════════════"
echo "Demo 5: Quiet Mode with JSON Output"
echo "═══════════════════════════════════════════════════════════════"
python orchestrate.py -d ../../../demo-docs/ --quiet --json-output demo-report.json
echo "✅ Report saved to demo-report.json"
echo "📊 First 10 lines of JSON report:"
head -10 demo-report.json
echo ""

# Demo 6: Error Handling (with missing file)
echo "═══════════════════════════════════════════════════════════════"
echo "Demo 6: Error Handling (continue on error)"
echo "═══════════════════════════════════════════════════════════════"
python orchestrate.py ../../../demo-docs/sample1.md missing-file.pdf ../../../demo-docs/sample2.txt || true
echo ""

# Demo 7: Help Information
echo "═══════════════════════════════════════════════════════════════"
echo "Demo 7: Help Information"
echo "═══════════════════════════════════════════════════════════════"
python orchestrate.py --help
echo ""

# Demo 8: Run Test Suite
echo "═══════════════════════════════════════════════════════════════"
echo "Demo 8: Automated Test Suite"
echo "═══════════════════════════════════════════════════════════════"
python test_orchestrator.py
echo ""

echo "================================================================"
echo "   All Demonstrations Complete!"
echo "================================================================"
echo ""
echo "📊 Key Features Demonstrated:"
echo "  ✅ Single file processing"
echo "  ✅ Multiple file processing"
echo "  ✅ Directory processing (recursive)"
echo "  ✅ Configurable parallelism"
echo "  ✅ Quiet mode"
echo "  ✅ JSON report generation"
echo "  ✅ Error handling (continue-on-error)"
echo "  ✅ Comprehensive help documentation"
echo "  ✅ Automated testing"
echo ""
echo "📁 Generated Files:"
echo "  - demo-report.json (JSON processing report)"
echo "  - test-report.json (from earlier tests)"
echo ""
echo "🎯 Next Steps:"
echo "  1. Implement real agent classes (extract, chunk, embed, write)"
echo "  2. Replace mock agents with production implementations"
echo "  3. Add environment configuration for API keys"
echo "  4. Deploy to production environment"
echo ""
