# DocuMind - AI-Powered Knowledge Management System

Build an intelligent Q&A system for company documents using Claude, RAG, and agentic engineering.  Documind is an accompanying sample app to master the concepts you learn in the HeroForge AI-Powered Software Development Course.

## What is DocuMind?

DocuMind is an AI chatbot that answers questions from your company's documents.
Ask questions in natural language, get accurate answers with source citations.

## What You'll Build (Sessions 3-10)

| Session | Topic | What You'll Create |
|---------|-------|-------------------|
| S3 | Skills, Subagents, Hooks | Document upload pipeline |
| S4 | Database & MCP | Supabase integration |
| S5 | Multi-Agent Systems | Parallel document processing |
| S6 | RAG Implementation | Q&A interface |
| S7 | Advanced Parsing | PDF/DOCX support |
| S8 | Vector Databases | Semantic search |
| S9 | Memory & Learning | Conversation memory |
| S10 | Evaluation | Quality metrics |

## Quick Start

1. **Fork this repository** to your GitHub account
2. **Launch in Codespaces:** Click green `Code` button → `Codespaces` → `Create codespace on main`
3. **Follow setup guide:** See `docs/guides/SETUP.md`
4. **Get workshop files from your instructor**

## Prerequisites

- Basic Python & JavaScript knowledge
- GitHub account (free)
- Anthropic API key (free tier available)
- OpenAI API key (pay-as-you-go)

## Project Structure

```
heroforge-documind/
├── .claude/              # Claude Code configuration
│   ├── agents/          # AI agent definitions
│   ├── skills/          # Custom skills (you'll create these!)
│   ├── subagents/       # Specialized agents (you'll create these!)
│   └── hooks/           # Automation scripts (you'll create these!)
├── src/
│   └── documind/        # Main DocuMind application code
├── tests/               # Test files
├── docs/
│   ├── guides/          # Setup and troubleshooting guides
│   └── workshops/       # Workshop files (S3, S4, and future sessions)
├── .env.example         # Template for environment variables
└── package.json         # Node.js dependencies
```

## Technologies

- **Claude Code** - AI-powered development assistant
- **Supabase** - PostgreSQL database with pgvector for semantic search
- **OpenRouter** - Multi-model LLM access (GPT-4, Gemini, etc.)
- **RAGAS** - Evaluation framework for RAG systems

## Environment Setup

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Add your API keys to .env file

# 3. Install dependencies
npm install

# 4. Run tests
npm test

# 5. Launch Claude Code
dsp
```

## Getting Help

- **Workshop issues:** Raise your hand or ask your instructor
- **Technical bugs:** Create a GitHub Issue in your fork
- **Course questions:** Ask in the course chat/Discord

## License

MIT License - See [LICENSE](LICENSE) file
