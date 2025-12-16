# HeroForge.AI Course: AI-Powered Software Development 
## Lesson 3 Workshop: Skills, Subagents, Hooks & DocuMind Foundation

**Estimated Time:** 45-60 minutes\
**Difficulty:** Intermediate\
**Prerequisites:** Completed Sessions 1-2 (Agentic Engineering fundamentals, Claude Code CLI)

---

## Pre-Workshop Setup (REQUIRED)

### Complete the DocuMind Setup Guide First!

Before starting this workshop, you MUST complete the **S3-Documind-setup-guide.md** which walks you through:

1. **Fork the student repository:** https://github.com/mamd69/heroforge-documind
2. **Launch GitHub Codespace** from your fork
3. **Install Claude Code** (`npm install -g @anthropic-ai/claude-code`)
4. **Install Dialogue Reporter** (`npx dialogue-reporter install`)
5. **Initialize Claude Code** (run `/init` inside Claude Code session)
6. **Configure API keys** (Anthropic, OpenAI)
7. **Verify your environment**

**All workshop exercises assume you are working in your forked `heroforge-documind` Codespace.**

If you haven't completed the setup guide, stop now and complete it first. The workshop won't work properly without the correct environment.

---

### Quick Verification

In your **heroforge-documind Codespace**, run these commands to verify you're ready:

```bash
# Verify you're in the right repo
pwd
# Expected: /workspaces/heroforge-documind

# Verify Claude Code
claude --version

# Verify Dialogue Reporter hooks
ls .claude/hooks/
# Should show dialogue reporter files

# Verify Claude Code was initialized
ls .claude/
# Should show CLAUDE.md and other config files

# Or verify CLAUDE.md exists
cat CLAUDE.md 2>/dev/null && echo "✅ Claude Code initialized" || echo "❌ Run /init inside Claude Code"

# Verify API key is set
echo $ANTHROPIC_API_KEY
# Should show sk-ant-xxxxx
```

**All checks pass?** You're ready to start the workshop!

---

## Workshop Objectives

By completing this workshop, you will:
- [x] Understand the difference between Personal and Project Skills
- [x] Create a custom Claude Skill for document processing
- [x] Build a Subagent with specialized knowledge for code review
- [x] Implement SessionStart and PostToolUse hooks for automation
- [x] Start building the DocuMind foundation (file upload & basic processing)
- [x] Apply the skill/subagent/hook architecture to a real application

---

## Git Workflow Basics (5 minutes)

### Concept: Version Control for AI Development

When building AI systems like DocuMind, version control is essential for:
- **Tracking experiments:** "Which prompt worked better?"
- **Collaboration:** Multiple developers working on different features
- **Rollback capability:** Undo changes that broke the system
- **Code review:** Get feedback before merging changes

### Exercise 0.1: Set Up Your Development Branch

**Task:** Create a feature branch for document processing work.

**Step 1: Create a GitHub Issue (2 mins)**

Go to your GitHub repository and create an issue:
```
Title: Implement document processing pipeline
Labels: feature, session-3
Description:
- Create multi-agent document processing architecture
- Implement ClaudeFlow swarm coordination
- Add semantic chunking and embedding generation
- Test pipeline with sample documents
```

Note the issue number (e.g., #12)

**Step 2: Create Feature Branch (1 min)**

```bash
# Check current branch
git branch
# Should show: * main

# Create feature branch from issue
git checkout -b issue-12-document-processing

# Verify you're on the new branch
git branch
# Should show: * issue-12-document-processing
```

**Branch Naming Convention:**
- Format: `issue-<number>-<short-description>`
- Example: `issue-12-document-processing`
- Use hyphens, lowercase, keep under 50 characters

**Step 3: Configure Git (if first time) (2 mins)**

```bash
# Set your name and email (if not already configured)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list | grep user
```

### When to Commit

Throughout this workshop, commit your work at these milestones:
1. **After Exercise 1.1:** Architecture design complete
2. **After Exercise 2.2:** ClaudeFlow agents spawned
3. **After Exercise 3.1:** Pipeline coordinator implemented
4. **After Module 4:** Challenge complete

### Commit Message Guidelines

Use this format:
```
<type>: <short summary (50 chars or less)>

<optional detailed description>

Closes #<issue-number>
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Adding tests
- `refactor:` Code restructuring

**Examples:**
```bash
# Good commit message
git commit -m "feat: implement ClaudeFlow multi-agent coordinator

- Add PipelineCoordinator class with async document processing
- Implement parallel execution with asyncio.gather
- Add error recovery with exponential backoff
- Track processing metrics per document

Closes #12"

# Bad commit messages (avoid these)
git commit -m "updates"
git commit -m "fixed stuff"
git commit -m "wip"
```

**Quick Commit Flow:**

```bash
# 1. Check what changed
git status

# 2. Add files (be selective!)
git add src/documind/pipeline_coordinator.py

# 3. Commit with clear message
git commit -m "feat: add pipeline coordinator for document processing

Implements parallel document processing with ClaudeFlow agents.
Includes error handling and performance metrics.

Closes #12"

# 4. View your commit history
git log --oneline -5
```

**Pro Tip:** Commit small, logical units of work. Each commit should represent one complete idea.

---

## Planning Your Work with GitHub Issues (5 minutes)

### Concept: Issue-Driven Development

Professional developers don't just start coding. They plan work using **GitHub Issues**:

**Benefits:**
- 📋 **Clear requirements** - Document what to build before building it
- 💬 **Discussion space** - Get feedback from teammates before coding
- 📊 **Progress tracking** - See what's done, in progress, and todo
- 🔗 **Traceability** - Link commits and PRs to original requirements
- 📚 **Documentation** - History of decisions and rationale

**When to create an issue:**
- ✅ Before starting a new feature
- ✅ When you find a bug
- ✅ For improvements or optimizations
- ✅ For questions or discussion

### Issue-Driven Workflow

```
1. Create Issue → Document what to build (#12)
2. Discuss → Get feedback from team/instructor
3. Break Down → Add checkboxes for sub-tasks
4. Create Branch → git checkout -b issue-12-feature
5. Implement → Write code
6. Commit → Reference issue: "Closes #12"
7. Create PR → Links to issue automatically
8. Review → Discuss in PR
9. Merge → Issue auto-closes
10. Celebrate → Feature complete! 🎉
```

### Best Practices

**DO:**
- ✅ Create issue BEFORE starting work
- ✅ Write clear, specific requirements
- ✅ Add acceptance criteria (how to know it's done)
- ✅ Reference issues in commits: `git commit -m "feat: add feature (Closes #12)"`
- ✅ Close issues when work is merged

**DON'T:**
- ❌ Start coding without an issue
- ❌ Create vague issues: "Improve the thing"
- ❌ Forget to close issues (leads to zombie issues)
- ❌ Put code in issues (use gists/branches instead)

Now you're ready to implement the feature tracked by this issue! 🚀

---

## Module 1: Understanding Skills (15 minutes)

### Concept Review

**What is a Claude Skill?**

A Claude Skill is a reusable, shareable instruction set that gives Claude specialized capabilities. Think of it as a "knowledge module" or "expert persona" that can be invoked on-demand.

**Types of Skills:**
1. **Personal Skills** (`~/.claude/skills/`) - Available across all your projects
2. **Project Skills** (`.claude/skills/`) - Specific to one repository/project

**Skills vs. Regular Prompts:**
- **Regular Prompt**: One-off instruction ("Write a Python function...")
- **Skill**: Reusable expert system ("You are a Python testing expert. When asked to test code, you follow these rules...")

**Skill Structure:**
```markdown
# Skill Name

Brief description

## Usage
When to invoke this skill

## Rules
1. Rule one
2. Rule two

## Example
Example invocation
```

---

### Exercise 1.1: Create Your First Skill - Document Parser Expert

**Task:** Create a skill that makes Claude an expert at parsing and extracting information from documents.

**Instructions:**

**Step 1: Create the Skill Directory (3 mins)**

In your Codespace terminal:
```bash
# Create project skills directory with skill folder
mkdir -p .claude/skills/document-parser

# Create the skill definition file
touch .claude/skills/document-parser/skill.md
```

**Step 2: Write the Skill Definition (7 mins)**

Open `.claude/skills/document-parser/skill.md` and add:

```markdown
# Document Parser Expert

You are an expert at parsing, analyzing, and extracting structured information from documents.

## Usage
Invoke this skill when working with document processing, text extraction, or content analysis.

## Rules
1. Always identify the document type first (PDF, Markdown, text, etc.)
2. Extract key metadata (title, author, date, sections)
3. Create structured summaries with clear hierarchies
4. Identify and extract key entities (names, dates, locations, concepts)
5. Preserve important formatting and structure
6. Flag any ambiguous or unclear content
7. Output in clean, parseable JSON or Markdown format

## Output Format
For each document analyzed:
- **Metadata**: Type, size, creation date
- **Structure**: Sections, headings, hierarchy
- **Key Entities**: People, places, dates, concepts
- **Summary**: 2-3 sentence overview
- **Action Items**: Extracted tasks or next steps (if any)

## Example
Input: "Analyze this meeting notes document"
Output:
```json
{
  "metadata": {
    "type": "Meeting Notes",
    "date": "2025-11-24",
    "participants": ["Alice", "Bob"]
  },
  "structure": {
    "sections": ["Agenda", "Discussion", "Action Items"]
  },
  "key_entities": {
    "people": ["Alice", "Bob"],
    "topics": ["Q4 Planning", "Budget Review"]
  },
  "summary": "Team meeting discussing Q4 planning and budget allocation.",
  "action_items": [
    "Alice: Submit budget proposal by Friday",
    "Bob: Schedule follow-up meeting"
  ]
}
```
```

**Step 3: Test the Skill (5 mins)**

In your Claude Code terminal (`dsp`), type:
```
Use @document-parser to analyze this text: "Project Update - November 24, 2025. Team leads discussed the DocuMind implementation timeline. Sarah will complete the database schema by Monday. John will write integration tests. Next meeting: December 1st."
```

**Expected Outcome:**
Claude should respond using the structured JSON format defined in your skill, extracting participants, dates, and action items.

---

**🔄 Git Checkpoint**

Commit your progress:

```bash
git add .claude/skills/document-parser/skill.md
git commit -m "feat: create document parser skill

- Add document parser expert skill with structured output format
- Define metadata extraction and entity recognition rules
- Include JSON output template for consistent analysis

Relates to #12"
```

---

### Quiz 1:

**Question 1:** What is the difference between Personal and Project skills?

   a) Personal skills are stored in your home directory and available everywhere; Project skills are in `.claude/skills/` and specific to one repository\
   b) Personal skills are free; Project skills cost money\
   c) Personal skills use Python; Project skills use JavaScript\
   d) There is no difference

**Question 2:** When should you create a skill instead of just writing a prompt?\
   a) When you need the same expert behavior across multiple conversations or projects\
   b) When you want to write less than 10 words\
   c) Skills are always better than prompts in every situation\
   d) Only when working with Python code

**Question 3:** What goes in the "Rules" section of a skill definition?\
   a) The specific behaviors and constraints Claude should follow when using this skill\
   b) Legal terms and conditions\
   c) Database connection strings\
   d) Your API keys

**Answers:**
1. **a)** Personal skills (~/.claude/skills/) are globally available; Project skills (.claude/skills/) are repository-specific
2. **a)** Skills are best for reusable expert behaviors you'll need repeatedly
3. **a)** Rules define the specific behaviors and constraints for the skill

---

## Module 2: Building Subagents (15 minutes)

### Concept Review

**What is a Subagent?**

A Subagent is a specialized AI agent that Claude can spawn to handle specific tasks. While Skills give Claude expertise, Subagents allow Claude to *delegate* work to separate, focused agents.

**Skills vs. Subagents:**
| Feature | Skill | Subagent |
|---------|-------|----------|
| **What it is** | Knowledge module | Separate agent instance |
| **How it works** | Claude uses the skill's rules | Claude spawns independent agent |
| **Best for** | Adding expertise | Delegating complex subtasks |
| **Example** | "Document parser expert" | "Code reviewer that checks security" |

**Subagent Structure:**
```yaml
name: subagent-name
role: What this agent does
expertise: Specific knowledge domain
tools: Available tools/capabilities
constraints: Limitations and rules
```

---

### Exercise 2.1: Create a Code Review Subagent

**Task:** Build a subagent that specializes in security-focused code review.

**Instructions:**

**Step 1: Create Subagent Directory (2 mins)**

```bash
mkdir -p .claude/agents
touch .claude/agents/security-reviewer.md
```

**Step 2: Define the Subagent (8 mins)**

Open `.claude/agents/security-reviewer.md` and add:

```markdown
# Security Review Subagent

## Role
Expert security-focused code reviewer specializing in identifying vulnerabilities, security anti-patterns, and compliance issues.

## Expertise
- OWASP Top 10 vulnerabilities
- Secure coding practices (Python, JavaScript, SQL)
- Authentication & authorization patterns
- Data validation and sanitization
- Secret management
- SQL injection prevention
- XSS prevention
- CSRF protection

## Review Checklist
When reviewing code, ALWAYS check:

### Input Validation
- [ ] All user inputs validated before use
- [ ] Type checking enforced
- [ ] Length limits applied
- [ ] Special characters sanitized

### Authentication & Authorization
- [ ] Authentication required for sensitive operations
- [ ] Authorization checks present
- [ ] Session management secure
- [ ] Password handling follows best practices

### Data Protection
- [ ] No hardcoded secrets or API keys
- [ ] Sensitive data encrypted at rest
- [ ] HTTPS enforced for data in transit
- [ ] Database queries parameterized (no string concatenation)

### Error Handling
- [ ] No sensitive information in error messages
- [ ] Proper exception handling
- [ ] Logging doesn't expose secrets

### Dependencies
- [ ] No known vulnerable dependencies
- [ ] Dependencies from trusted sources

## Output Format
Provide security review in this structure:

### Summary
- Overall security rating: HIGH RISK / MEDIUM RISK / LOW RISK / SECURE
- Critical issues: [count]
- Warnings: [count]

### Critical Issues 🚨
[List any immediate security vulnerabilities that must be fixed]

### Warnings ⚠️
[List security concerns that should be addressed]

### Recommendations ✅
[List best practice improvements]

### Approved Items ✓
[List security controls that are correctly implemented]
```

**Step 3: Invoke the Subagent (5 mins)**

In Claude Code terminal:
```
@security-reviewer Review this code for security issues:

def upload_file(filename, content):
    path = f"/uploads/{filename}"
    with open(path, 'w') as f:
        f.write(content)
    return path
```

**Expected Outcome:**
The subagent should identify:
- Path traversal vulnerability (filename not sanitized)
- No file type validation
- No size limits
- Arbitrary file write capability

---

### Exercise 2.2: Create a Documentation Subagent

**Challenge (Optional Advanced Task):**

Create a subagent called `.claude/agents/doc-writer.md` that:
- Generates comprehensive documentation for code
- Follows JSDoc or Python docstring standards
- Creates README files with usage examples
- Identifies undocumented functions

Test it on your DocuMind code from the demo!

Pro-Tip: First create a GitHub issue. Then ask Claude Code to implement the issue. Watch with amazement!

---

**🔄 Git Checkpoint**

Commit your progress:

```bash
git add .claude/agents/security-reviewer.md
git commit -m "feat: add security review subagent

- Create specialized security reviewer with OWASP checklist
- Implement vulnerability detection for SQL injection and path traversal
- Add structured output format with risk ratings

Relates to #12"
```

---

### Quiz 2:

**Question 1:** What is the primary difference between a Skill and a Subagent?\
   a) Skills provide expertise to Claude; Subagents are separate agents Claude delegates work to\
   b) Skills are written in YAML; Subagents are written in JSON\
   c) Skills are faster than Subagents\
   d) Subagents can only review code

**Question 2:** When should you use a Subagent instead of a Skill?\
   a) When the task is complex enough to warrant a separate, focused agent with its own context\
   b) When you want to save money\
   c) Only on Tuesdays\
   d) Whenever you write Python code

**Question 3:** In the security reviewer subagent, what does the checklist accomplish?\
   a) It ensures the subagent systematically checks for specific security issues every time\
   b) It makes the subagent slower\
   c) It's just for decoration\
   d) It replaces the need for actual code review

**Answers:**
1. **a)** Skills = expertise for Claude; Subagents = separate specialized agents
2. **a)** Use subagents for complex, focused tasks that benefit from isolated context
3. **a)** Checklists ensure systematic, repeatable security analysis

---

## Module 3: Implementing Hooks (15 minutes)

### Concept Review

**What are Hooks?**

Hooks are automation scripts that run automatically at specific points in Claude Code's workflow. They are configured in `.claude/settings.json` and execute shell commands or scripts.

**Hook Event Types:**
| Event | When it Runs |
|-------|--------------|
| `PreToolUse` | Before Claude uses a tool (can block) |
| `PostToolUse` | After Claude uses a tool |
| `UserPromptSubmit` | When user submits a prompt |
| `SessionStart` | When Claude session begins |
| `SessionEnd` | When Claude session ends |
| `Stop` | When Claude finishes responding |

**Why Use Hooks?**
- Automate repetitive tasks (formatting, linting, testing)
- Enforce quality standards automatically
- Create audit trails
- Coordinate with other systems (git, databases, APIs)

---

### Exercise 3.1: Extend the Existing SessionStart Hook

**Task:** Extend the existing SessionStart hook (installed by Dialogue Reporter) to add environment validation.

> **Note:** The Dialogue Reporter you installed earlier already created `.claude/hooks/SessionStart.sh`. Instead of replacing it, we'll create a validation script that the existing hook can call, or we'll learn how to extend hooks.

**Instructions:**

**Step 1: Review Existing Hooks (2 mins)**

First, let's see what hooks already exist from Dialogue Reporter.  Ask Claude Code:
```bash
What do the current hooks do?
```

**Step 2: Create a Validation Script (5 mins)**

Instead of overwriting the existing SessionStart.sh, we'll create a separate validation script (in terminal):

```bash
touch .claude/hooks/ValidateEnvironment.sh
chmod +x .claude/hooks/ValidateEnvironment.sh
```

Open `.claude/hooks/ValidateEnvironment.sh` and add:

```bash
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
```

**Step 3: Extend the Existing SessionStart Hook (3 mins)**

Now we need to modify the existing `SessionStart.sh` to also call our validation script. Open `.claude/hooks/SessionStart.sh` and add this line at the end (before any final exit):

```bash
# Add this line to call our validation script
.claude/hooks/ValidateEnvironment.sh
```

**Alternative: Add a Second SessionStart Hook Entry**

If you don't want to modify the Dialogue Reporter hook, you can add a second entry in `.claude/settings.json`. The hooks array can have multiple entries that all run.

> **Important for Codespace users:** Your `.claude/settings.json` already has hooks from Dialogue Reporter. You need to ADD to the existing file, not replace it. Open the file and add the ValidateEnvironment entry to the SessionStart array.

Here's what your complete settings.json should look like after adding the validation hook:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/SessionStart.sh"
          }
        ]
      },
      {
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/ValidateEnvironment.sh"
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/UserPromptSubmit.sh"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/Stop.sh"
          }
        ]
      }
    ]
  }
}
```


**Important notes about hook configuration:**
- Hooks are registered in `.claude/settings.json`, NOT a separate config file
- Hook commands run from the project root, so `.claude/hooks/` paths work directly
- Exit code `0` = success, exit code `2` = blocking error shown to Claude
- The `type` can be `"command"` (shell) or `"prompt"` (LLM evaluation)
- Multiple hook entries for the same event all run in sequence
- **Don't delete existing Dialogue Reporter hooks** (UserPromptSubmit, Stop) - they capture your conversations

---

**🔄 Git Checkpoint**

Commit your progress:

```bash
git add .claude/hooks/ValidateEnvironment.sh .claude/settings.json
git commit -m "feat: add environment validation hook

- Add environment validation checks (directories, dependencies)
- Check git status and working tree cleanliness
- Log validation for audit trail
- Extends existing Dialogue Reporter SessionStart hook

Relates to #12"
```

---

**Step 4: Test the Hook (5 mins)**

```bash
# Test the validation script manually first
./.claude/hooks/ValidateEnvironment.sh

# Expected output:
# 🔍 Environment Validator Running...
# ✅ In project root directory
# ✅ .claude directory exists
# ✅ docs directory exists
# ✅ src directory exists
# ✅ Git working tree clean
# ✅ Environment validation complete - ready to proceed
```

Then start Claude Code to see both hooks run automatically:
```bash
dsp
```

---

### Exercise 3.2: Create a PostToolUse Hook for Auto-Formatting

**Task:** Create a hook that automatically formats code after Claude writes or edits a file.

**Instructions:**

**Step 1: Create the Hook Script (5 mins)**

```bash
touch .claude/hooks/FormatOnSave.sh
chmod +x .claude/hooks/FormatOnSave.sh
```

Open `.claude/hooks/FormatOnSave.sh` and add:

```bash
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
```

**Step 2: Update settings.json with the New Hook**

Update `.claude/settings.json` to add the PostToolUse hook. This combines with the existing Dialogue Reporter hooks:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "matcher": "startup",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/SessionStart.sh"
          }
        ]
      },
      {
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/ValidateEnvironment.sh"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/FormatOnSave.sh"
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/UserPromptSubmit.sh"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": ".claude/hooks/Stop.sh"
          }
        ]
      }
    ]
  }
}
```

**Key configuration details:**
- `"matcher": "Write|Edit"` - Only runs for Write and Edit tools (regex pattern)
- The hook receives JSON via stdin with tool context
- Use `jq` to parse the JSON and extract the file path
- Existing Dialogue Reporter hooks (SessionStart.sh, UserPromptSubmit.sh, Stop.sh) are preserved

**Step 3: Test the Hook**

```bash
# Test that the hook works by starting Claude and editing a file
dsp
# Then ask Claude to create a Python file - the hook will auto-format it
create a test python file so we can test the '/workspaces/documind-mark/.claude/hooks/FormatOnSave.sh'.  verify for me that it works 
```

---

**🔄 Git Checkpoint**

Commit your hooks:

```bash
git add .claude/hooks/FormatOnSave.sh .claude/settings.json
git commit -m "feat: add auto-formatting PostToolUse hook

- Format Python files with black
- Format JS/TS files with prettier
- Clean trailing whitespace in Markdown
- Preserves existing Dialogue Reporter hooks

Relates to #12"
```

---

### Quiz 3:

**Question 1:** Where are Claude Code hooks configured?\
   a) In `.claude/settings.json` under the `"hooks"` key\
   b) In a separate `.claude/hooks/config.json` file\
   c) In `package.json`\
   d) Hooks cannot be configured

**Question 2:** What exit code should a hook return to block Claude with an error message?\
   a) Exit code 2\
   b) Exit code 0\
   c) Exit code 1\
   d) Exit code -1

**Question 3:** What does the `"matcher": "Write|Edit"` pattern do in a PostToolUse hook?\
   a) It filters the hook to only run when Claude uses Write or Edit tools\
   b) It matches all tools\
   c) It writes and edits files\
   d) It's a syntax error

**Answers:**
1. **a)** Hooks are configured in `.claude/settings.json`
2. **a)** Exit code 2 = blocking error shown to Claude
3. **a)** The matcher is a regex pattern that filters which tools trigger the hook

---

## Module 4: Challenge Project - Mini DocuMind (15 minutes)

### Challenge Overview

Build a simplified version of DocuMind that demonstrates Skills, Subagents, and Hooks working together.

**Your Mission:**
Create a document processing pipeline that:
1. Uses a Skill to analyze uploaded documents
2. Uses a Subagent to validate document security
3. Uses Hooks to automate the workflow

---

### Challenge Requirements

**Feature:** Document Upload & Analysis Pipeline

**What to Build:**

1. **Document Upload Function** (Python)
   - Accept file path as input
   - Read file contents
   - Return document metadata

2. **Use the Document Parser Skill**
   - Invoke your document-parser skill from Module 1
   - Extract structured information from the document

3. **Security Review via Subagent**
   - Use your security-reviewer subagent to check for:
     - Path traversal risks
     - File type validation
     - Content security issues

4. **Automated Hooks**
   - SessionStart hook validates environment
   - PostToolUse hook auto-formats files

---

### Starter Code

Create `src/documind/upload_handler.py`:

```python
"""
DocuMind Document Upload Handler
Demonstrates Skills, Subagents, and Hooks integration
"""
import os
import json
from datetime import datetime
from pathlib import Path

def validate_file_path(file_path):
    """
    Validates that file path is safe and exists.

    TODO: Add security validation
    - Check for path traversal attempts
    - Validate file extension
    - Check file size limits
    """
    # Your code here
    pass

def read_document(file_path):
    """
    Reads document contents safely.

    TODO: Implement safe file reading
    - Open file in read mode
    - Handle different encodings
    - Return contents as string
    """
    # Your code here
    pass

def extract_metadata(file_path, contents):
    """
    Extracts metadata from document.

    TODO: Extract:
    - File name
    - File size
    - Created date
    - Modified date
    - Number of lines/words
    """
    # Your code here
    pass

def analyze_document(file_path):
    """
    Main function: orchestrates document analysis.

    This function should:
    1. Validate the file path (security)
    2. Read the document
    3. Extract metadata
    4. Return structured analysis

    TODO: Implement full pipeline
    """
    # Your code here
    pass

# Test the function
if __name__ == "__main__":
    # Create a sample document for testing
    test_doc = "test_document.txt"
    with open(test_doc, 'w') as f:
        f.write("Sample document for DocuMind testing.\nThis is line 2.")

    # Analyze it
    result = analyze_document(test_doc)
    print(json.dumps(result, indent=2))
```

---

### Your Task

**Step 1: Implement the Functions (10 mins)**

Using Claude Code with your skills and subagents:

1. Open `dsp` (Claude Code CLI)
2. Invoke your document-parser skill:
   ```
   @document-parser Complete the implementation of src/documind/upload_handler.py.

   Requirements:
   - validate_file_path: Check for "../" path traversal, validate extensions (.txt, .md, .pdf)
   - read_document: Safely read file contents, handle UTF-8 encoding
   - extract_metadata: Get file size, dates, word/line count
   - analyze_document: Orchestrate the full pipeline, return JSON
   ```

3. After implementation, invoke your security subagent:
   ```
   @security-reviewer Review src/documind/upload_handler.py for security vulnerabilities.
   ```
   HINT: Just tell Claude to "do it"

4. Fix any security issues identified

**Step 2: Test with Hooks (5 mins)**

```bash
# Run validation hook manually to validate environment
./.claude/hooks/ValidateEnvironment.sh

# Test the implementation
python src/documind/upload_handler.py

# Expected output: JSON with metadata
{
  "file": "test_document.txt",
  "size": 67,
  "lines": 2,
  "words": 9,
  "created": "2025-11-24T10:30:00",
  "safe": true
}
```

---

### Success Criteria

Your implementation is complete when:

- [ ] `validate_file_path` prevents path traversal attacks
- [ ] `validate_file_path` only allows .txt, .md, .pdf files
- [ ] `read_document` safely reads file contents
- [ ] `extract_metadata` returns file stats (size, dates, word count)
- [ ] `analyze_document` orchestrates the full pipeline
- [ ] Security subagent finds NO critical issues
- [ ] SessionStart hook passes all checks
- [ ] Test run produces valid JSON output

---

## Answer Key

### Exercise 1.1 Solution

The skill file should be created at `.claude/skills/document-parser/skill.md` with the content provided in the exercise.

To test:
```
@document-parser Analyze this text: "Project Update - November 24, 2025..."
```

Expected output:
```json
{
  "metadata": {
    "type": "Project Update",
    "date": "2025-11-24"
  },
  "key_entities": {
    "people": ["Sarah", "John"],
    "dates": ["Monday", "December 1st"]
  },
  "summary": "DocuMind implementation timeline discussion with assigned tasks.",
  "action_items": [
    "Sarah: Complete database schema by Monday",
    "John: Write integration tests",
    "Team: Next meeting December 1st"
  ]
}
```

---

### Exercise 2.1 Solution

The subagent should be at `.claude/agents/security-reviewer.md`.

For the upload_file test code, the review should identify:

**Security Review:**

### Summary
- Overall security rating: **HIGH RISK** 🚨
- Critical issues: 2
- Warnings: 1

### Critical Issues 🚨
1. **Path Traversal Vulnerability**: The filename is directly concatenated without sanitization. An attacker can use "../" to write files anywhere on the system.
   ```python
   # Attack: filename = "../../etc/passwd"
   # Results in: /uploads/../../etc/passwd = /etc/passwd
   ```

2. **Arbitrary File Write**: No restrictions on file type, size, or content. Attacker can upload malicious files.

### Warnings ⚠️
1. No authentication check - anyone can upload files

### Recommendations ✅
```python
import os
from pathlib import Path

ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.md'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
UPLOAD_DIR = Path("/uploads").resolve()

def upload_file(filename, content):
    # Sanitize filename
    safe_filename = Path(filename).name
    file_path = (UPLOAD_DIR / safe_filename).resolve()

    # Verify path is within upload directory
    if not str(file_path).startswith(str(UPLOAD_DIR)):
        raise ValueError("Invalid file path")

    # Check extension
    if file_path.suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("File type not allowed")

    # Check size
    if len(content) > MAX_FILE_SIZE:
        raise ValueError("File too large")

    with open(file_path, 'w') as f:
        f.write(content)

    return str(file_path)
```

---

### Module 4 Challenge Solution

Complete implementation of `src/documind/upload_handler.py`:

```python
"""
DocuMind Document Upload Handler
Demonstrates Skills, Subagents, and Hooks integration
"""
import os
import json
from datetime import datetime
from pathlib import Path

# Security configuration
ALLOWED_EXTENSIONS = {'.txt', '.md', '.pdf'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
BASE_DIR = Path(__file__).parent.parent.parent.resolve()

def validate_file_path(file_path):
    """
    Validates that file path is safe and exists.

    Security checks:
    - Prevents path traversal
    - Validates file extension
    - Checks file size limits
    """
    try:
        # Convert to Path object and resolve
        path = Path(file_path).resolve()

        # Check if file exists
        if not path.exists():
            return False, "File does not exist"

        # Check for path traversal
        # Ensure resolved path is within allowed directory
        if not str(path).startswith(str(BASE_DIR)):
            return False, "Path traversal attempt detected"

        # Validate file extension
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            return False, f"File type not allowed. Allowed: {ALLOWED_EXTENSIONS}"

        # Check file size
        file_size = path.stat().st_size
        if file_size > MAX_FILE_SIZE:
            return False, f"File too large. Max: {MAX_FILE_SIZE} bytes"

        # Check if it's a file (not directory)
        if not path.is_file():
            return False, "Path is not a file"

        return True, "File is valid"

    except Exception as e:
        return False, f"Validation error: {str(e)}"

def read_document(file_path):
    """
    Reads document contents safely.

    Handles:
    - Different encodings (UTF-8, fallback to latin-1)
    - Read errors
    - Returns contents as string
    """
    try:
        # Try UTF-8 first
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Failed to read file: {str(e)}")
    except Exception as e:
        raise Exception(f"Error reading file: {str(e)}")

def extract_metadata(file_path, contents):
    """
    Extracts metadata from document.

    Returns:
    - File name
    - File size (bytes)
    - Created date
    - Modified date
    - Number of lines
    - Number of words
    """
    path = Path(file_path)
    stats = path.stat()

    # Count lines and words
    lines = contents.split('\n')
    line_count = len(lines)
    word_count = sum(len(line.split()) for line in lines)

    metadata = {
        "file_name": path.name,
        "file_path": str(path),
        "file_size_bytes": stats.st_size,
        "file_extension": path.suffix,
        "created_timestamp": datetime.fromtimestamp(stats.st_ctime).isoformat(),
        "modified_timestamp": datetime.fromtimestamp(stats.st_mtime).isoformat(),
        "line_count": line_count,
        "word_count": word_count,
        "character_count": len(contents)
    }

    return metadata

def analyze_document(file_path):
    """
    Main function: orchestrates document analysis.

    Pipeline:
    1. Validate the file path (security)
    2. Read the document
    3. Extract metadata
    4. Return structured analysis
    """
    result = {
        "status": "unknown",
        "file": file_path,
        "timestamp": datetime.now().isoformat(),
        "metadata": None,
        "preview": None,
        "error": None
    }

    try:
        # Step 1: Validate file path
        is_valid, message = validate_file_path(file_path)
        if not is_valid:
            result["status"] = "error"
            result["error"] = f"Validation failed: {message}"
            return result

        # Step 2: Read document
        contents = read_document(file_path)

        # Step 3: Extract metadata
        metadata = extract_metadata(file_path, contents)

        # Step 4: Create preview (first 200 characters)
        preview = contents[:200]
        if len(contents) > 200:
            preview += "..."

        # Success
        result["status"] = "success"
        result["metadata"] = metadata
        result["preview"] = preview

        return result

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result

# Test the function
if __name__ == "__main__":
    # Create a sample document for testing
    test_doc = "test_document.txt"
    with open(test_doc, 'w') as f:
        f.write("Sample document for DocuMind testing.\n")
        f.write("This is line 2 with some additional content.\n")
        f.write("Line 3 continues the document with more information.")

    print("=" * 60)
    print("DocuMind Document Analysis Test")
    print("=" * 60)
    print()

    # Analyze it
    result = analyze_document(test_doc)
    print(json.dumps(result, indent=2))

    print()
    print("=" * 60)
    print("Testing Security Validation")
    print("=" * 60)
    print()

    # Test path traversal attempt
    malicious_path = "../../../etc/passwd"
    malicious_result = analyze_document(malicious_path)
    print("Path traversal test:")
    print(json.dumps(malicious_result, indent=2))

    print()

    # Test invalid extension
    invalid_ext = "test.exe"
    with open(invalid_ext, 'w') as f:
        f.write("test")
    invalid_result = analyze_document(invalid_ext)
    print("Invalid extension test:")
    print(json.dumps(invalid_result, indent=2))

    # Cleanup
    os.remove(test_doc)
    if os.path.exists(invalid_ext):
        os.remove(invalid_ext)
```

**Expected Output:**
```json
{
  "status": "success",
  "file": "test_document.txt",
  "timestamp": "2025-11-24T10:30:00.123456",
  "metadata": {
    "file_name": "test_document.txt",
    "file_path": "/workspaces/heroforge-documind/test_document.txt",
    "file_size_bytes": 156,
    "file_extension": ".txt",
    "created_timestamp": "2025-11-24T10:29:55.000000",
    "modified_timestamp": "2025-11-24T10:29:55.000000",
    "line_count": 3,
    "word_count": 18,
    "character_count": 156
  },
  "preview": "Sample document for DocuMind testing.\nThis is line 2 with some additional content.\nLine 3 continues the document with more information."
}
```

**Security Test Results:**
```json
{
  "status": "error",
  "file": "../../../etc/passwd",
  "error": "Validation failed: Path traversal attempt detected"
}
```

---

**🔄 Git Checkpoint**

Commit your progress:

```bash
git add src/documind/upload_handler.py tests/
git commit -m "feat: complete document upload handler with security validation

- Implement secure file path validation with path traversal protection
- Add multi-encoding document reader (UTF-8, latin-1 fallback)
- Extract comprehensive metadata (size, dates, word/line counts)
- Create document analysis pipeline with error handling
- Add security tests for path traversal and invalid extensions

Closes #12"
```

---

## Additional Challenges (Optional)

For students who finish early:

### Challenge 1: Advanced Skill - Code Explainer
Create a skill at `.claude/skills/code-explainer/skill.md` that:
- Analyzes code and explains it line-by-line
- Identifies design patterns used
- Suggests improvements
- Generates beginner-friendly documentation

Pro-Tip: 

Test it on the upload_handler.py solution!

### Challenge 2: Testing Subagent
Create `.claude/agents/test-writer.md` that:
- Generates pytest unit tests
- Ensures 80%+ code coverage
- Creates fixtures and mocks
- Tests both success and failure cases

Use it to create `tests/test_upload_handler.py`!

### Challenge 3: Git Commit Hook
Create a PostToolUse hook for git operations that:
- Runs tests before allowing commit
- Auto-generates commit message from changes
- Checks for sensitive data in commits
- Updates CHANGELOG.md automatically

### Challenge 4: Create a Skill Using the Skill Creator
Use the **Claude Skill Creator** skill to build your own custom skill!

**Setup:**
1. Download the skill-creator skill from: https://github.com/anthropics/skills/tree/main/skills/skill-creator
2. Drag the file into your `.claude/skills/` directory (just like Mark demonstrated in the lesson)

**Task:**
Use the skill-creator to build one of these skills:
- **API Documentation Expert** - Generates OpenAPI/Swagger docs from code
- **Performance Analyzer** - Reviews code for optimization opportunities
- **Accessibility Checker** - Validates UI code for WCAG compliance
- **Database Schema Designer** - Creates normalized database schemas from requirements

**How to use:**
```
@skill-creator Create a skill for [your chosen domain]
```

The skill-creator will guide you through:
- Defining the skill's purpose and expertise
- Creating appropriate rules and constraints
- Building example inputs/outputs
- Structuring the skill.md file properly

Test your new skill on real code from the DocuMind project!

---

## Troubleshooting

### Common Issue 1: Skill Not Loading
**Problem:** `@document-parser` returns "Skill not found"

**Solution:**
1. Check file exists: `ls -la .claude/skills/document-parser/`
2. Verify path is `.claude/skills/document-parser/skill.md`
3. Ensure file has content (not empty)
4. Restart Claude Code session

---

### Common Issue 2: Hook Permission Denied
**Problem:** `bash: ./.claude/hooks/ValidateEnvironment.sh: Permission denied`

**Solution:**
```bash
chmod +x .claude/hooks/ValidateEnvironment.sh
chmod +x .claude/hooks/FormatOnSave.sh
```

---

### Common Issue 3: Subagent Not Activating
**Problem:** Subagent doesn't provide specialized analysis

**Solution:**
1. Ensure file is at `.claude/agents/[name].md`
2. Include clear "Role" and "Expertise" sections
3. Invoke with `@[name]` (e.g., `@security-reviewer`)
4. Check for YAML frontmatter syntax errors

---

### Common Issue 4: Path Traversal Test Failing
**Problem:** Security test should block "../" but doesn't

**Solution:**
Ensure you're using `Path.resolve()` and checking if the resolved path starts with the base directory:

```python
path = Path(file_path).resolve()
if not str(path).startswith(str(BASE_DIR)):
    return False, "Path traversal detected"
```

---

## 5 Ways to Get Better Results from Claude Code

Before you continue to the next session, internalize these tips:

### 1. Demand proof, not promises
Ask "show me the diff" or "run the tests" — don't accept "I've updated the file" without evidence.

### 2. Break tasks into atomic steps
Instead of "refactor auth," try: "list files touching auth → show current flow → update login function → run tests."

### 3. Use "then verify" as a suffix
"Add the env variable to .env, then cat the file to confirm." Build verification into every request.

### 4. Ask for state before and after
"Show test output, make the fix, show test output again." Clear before/after comparison reveals what changed.

### 5. Be skeptical of "Done!" without artifacts
No terminal output, diff, or file contents? Assume nothing happened. Real completions produce evidence.

> **Core principle:** Treat Claude Code like a junior dev who's eager to please but needs to show their work.

---

## Key Takeaways

By completing this workshop, you've learned:

1. **Skills** provide reusable expertise to Claude for specific domains
2. **Subagents** delegate complex subtasks to specialized agent instances
3. **Hooks** automate workflow steps before/after tasks and edits
4. **Security** must be designed into file handling from the start
5. **DocuMind** foundation demonstrates real-world application architecture

**The Triad Pattern:**
```
Skill (Expertise) + Subagent (Delegation) + Hooks (Automation) = Powerful Workflow
```

---

## Next Session Preview

In **Session 4: Database Integration with Supabase MCP**, we'll:
- Connect DocuMind to a real PostgreSQL database
- Use the Supabase MCP server for database operations
- Implement document storage with metadata
- Create a conversation history table
- Build our first API endpoint

**Preparation:**
1. Create a free Supabase account at supabase.com
2. Keep your `.env` file handy for API keys
3. Review PostgreSQL basics (optional)

See you in Session 4!

---

**Workshop Complete! 🎉**

You've built the foundation of DocuMind with Skills, Subagents, and Hooks. You're ready to integrate real database storage!
