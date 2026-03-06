<div align="center">

<h1>VEXIS-CLI-1</h1>

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Experimental-orange?style=flat-square)]()

**AI agent that automates command-line tasks**

</div>

---

## About

**VEXIS-CLI-1** is an AI command-line agent that processes natural language instructions to execute terminal operations. Supports both local Ollama models and Google Gemini cloud models.

### AI Providers
- **Ollama**: Local models with privacy-first design (Gemini 3 Flash, open-source models)
- **Google Gemini**: Cloud-based with enterprise-grade reliability (Gemini 3)

> **⚠️ Important Note**: Some Ollama models may experience compatibility issues or errors. If you encounter problems with specific models, try alternatives like `gemma3:4b`, `qwen2.5:3b`, or `deepseek-r1:7b`.

> **Note**: Experimental project. Use with curiosity!

---

## Features

- Natural language to CLI conversion
- Command execution with intelligent error handling
- File operations and workflow automation
- Enhanced Ollama error handling with user-friendly guidance
- One-liner execution: `python3 run.py "do something"`

---

## Installation

```bash
git clone https://github.com/AInohogosya-team/VEXIS-CLI-1.git
cd VEXIS-CLI-1
python3 run.py "list files"  # Dependencies handled automatically
```

### Requirements
- Python 3.9+
- Google AI API key for Gemini models
- Optional: Ollama account for cloud models (`ollama signin`)

---

## Usage

```bash
# Basic commands
python3 run.py "list files in current directory"
python3 run.py "create hello.txt with content 'Hello World'"
python3 run.py "show system information"

# Options
python3 run.py "instruction" --debug     # Verbose logging
python3 run.py "instruction" --no-prompt # Skip provider selection
```

---

## Configuration

Edit `config.yaml`:

```yaml
api:
  preferred_provider: "ollama"  # "ollama" or "google"
  local_endpoint: "http://localhost:11434"
  local_model: "gemma3:4b"  # Recommended stable model
  timeout: 120
  max_retries: 3
```

**Model Recommendations:**
- **Stable**: `gemma3:4b`, `qwen2.5:3b`, `deepseek-r1:7b`
- **Experimental**: `gemini-3-flash-preview:latest` (may have issues)
- **Cloud**: Google Gemini 2.5 Flash (most reliable)

---

## Error Handling

Comprehensive Ollama error guidance:
- **Permission Errors**: macOS Full Disk Access, Linux permissions, Windows admin
- **Model Errors**: Available models list and alternative suggestions
- **Connection Issues**: Service restart and port checking
- **Installation Problems**: Platform-specific instructions

---

## Architecture

Two-phase execution engine:
1. **Command Planning**: Natural language analysis
2. **Terminal Execution**: Implementation with error recovery

### Core Components
- `TwoPhaseEngine` - Orchestration
- `ModelRunner` - AI provider abstraction
- `CommandParser` - Natural language processing
- `TaskVerifier` - Validation and error handling

---

<div align="center">

**VEXIS-CLI-1 - Intelligent command-line automation**

</div>
