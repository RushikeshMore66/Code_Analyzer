# Code Analyzer

A Python-based static analysis tool that creates a Knowledge Base from your code's issues and uses **Retrieval-Augmented Generation (RAG)** to answer questions and suggest fixes.

## Features

- **Static Code Analysis**: Detects common code smells and issues:
  - **Long Functions**: Identifies functions exceeding a specified line count (default 20).
  - **Bare Excepts**: Flags `except:` blocks without specific exception types.
  - **TODO Comments**: Finds and tracks TODOs left in the code.
- **RAG-Powered Q&A**:
  - Converts detected issues into structured "Knowledge Documents".
  - Retrieves relevant issues based on your natural language queries.
  - Uses an LLM (via Groq API) to explain the problem and suggest solutions based on the specific context of your code.
- **Interactive CLI**: Query the analyzer about the issues found in your file.
- **JSON Output**: Supports JSON output for integration with other tools or extensions.

## Prerequisites

- Python 3.8+
- [Groq API Key](https://console.groq.com/) for LLM functionality.

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install groq python-dotenv
   ```
3. Create a `.env` file in the root directory and add your Groq API key:
   ```ini
   GROQ_API_KEY=your_api_key_here
   ```

## Usage

### Interactive Mode

Run the analyzer on a Python file to see issues and enter the interactive Q&A mode:

```bash
python code_analyzer.py <filename>
```

**Example:**
```bash
python code_analyzer.py test.py
```

Once the analysis is complete, you can type questions like:
- "How do I fix the bare except?"
- "Why is the function 'process_data' flagged?"

Type `exit` to quit.

### JSON Output

To get the analysis results in JSON format (useful for integrations):

```bash
python code_analyzer.py --json <filename>
```

## How it Works

1. **Parsing**: Uses Python's `ast` module to parse the source code.
2. **Analysis**: Runs defined `Rule` classes against the AST to find violations.
3. **Knowledge Building**: Converts violations into rich documents containing the problem description, location, and generic solutions.
4. **Retrieval**: Uses keyword matching to find relevant issues based on your query.
5. **Generation**: Sends the retrieved context and your question to the Llama-3 model (via Groq) to generate a specific, helpful response.
