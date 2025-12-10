# Ollama Setup Guide

This guide will help you set up Ollama for AI-powered data cleaning with Smart Cleaner.

## Why Ollama?

- **100% Free** - No API costs, ever
- **100% Local** - Your data never leaves your machine
- **100% Private** - No cloud, no tracking
- **Fast** - Runs on your hardware

## Installation

### macOS

```bash
# Using Homebrew (recommended)
brew install ollama

# Or download from https://ollama.com/download
```

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Windows

Download the installer from [https://ollama.com/download](https://ollama.com/download)

## Pull a Model

After installing Ollama, pull a model:

```bash
# Recommended - best balance of quality and speed
ollama pull llama3.2

# Alternative - faster but slightly less accurate
ollama pull mistral

# For more powerful hardware
ollama pull llama3.1:8b
```

## Verify Installation

```bash
# Check Ollama is running
ollama list

# Test the model
ollama run llama3.2 "Hello, how are you?"
```

## Start Ollama Server

Ollama runs as a background service. It usually starts automatically after installation.

```bash
# If needed, start manually
ollama serve
```

The server runs on `http://localhost:11434` by default.

## Use with Smart Cleaner

### Command Line

```bash
# Uses llama3.2 by default
python universal_data_pipeline.py your_data.csv

# Use a different model
python universal_data_pipeline.py your_data.csv --model mistral
```

### Python API

```python
from smart_cleaner import AutoPreprocessor, PipelineConfig

config = PipelineConfig(
    use_ai_recommendations=True,
    ai_provider="ollama",
    ollama_model="llama3.2",  # Change model here
)

preprocessor = AutoPreprocessor(config)
cleaned_df, report = preprocessor.process(df)
```

## Environment Variables (Optional)

```bash
# Set default model
export OLLAMA_MODEL="llama3.2"

# Custom server URL (if running Ollama elsewhere)
export OLLAMA_BASE_URL="http://localhost:11434"
```

## Recommended Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| `llama3.2` | 2GB | Fast | Excellent | **Recommended** |
| `mistral` | 4GB | Faster | Good | Quick processing |
| `llama3.1:8b` | 5GB | Medium | Excellent | Complex datasets |
| `codellama` | 4GB | Fast | Good | Code-heavy data |

## Troubleshooting

### "Connection refused" error

Make sure Ollama is running:
```bash
ollama serve
```

### "Model not found" error

Pull the model first:
```bash
ollama pull llama3.2
```

### Slow performance

Try a smaller model:
```bash
ollama pull mistral
```

### Out of memory

Use a smaller model or increase system swap space.

## Hardware Requirements

| Model | Minimum RAM | Recommended RAM |
|-------|-------------|-----------------|
| `llama3.2` | 4GB | 8GB |
| `mistral` | 4GB | 8GB |
| `llama3.1:8b` | 8GB | 16GB |

## Getting Help

- Ollama documentation: https://ollama.com/docs
- Ollama Discord: https://discord.gg/ollama
- GitHub Issues: https://github.com/ollama/ollama/issues
