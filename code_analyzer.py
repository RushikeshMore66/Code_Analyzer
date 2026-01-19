import ast
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from groq import Groq
import os
import json
from dotenv import load_dotenv
load_dotenv()


try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception:
    groq_client = None


def call_llm(prompt: str) -> str:
    """
    Calls Groq with an open-source LLM.
    """
    if not groq_client:
        return "LLM processing is unavailable (GROQ_API_KEY not set)."

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior software engineer. "
                    "Answer ONLY using the provided context. "
                    "If the answer is not in the context, say: "
                    "'Not enough information available from analysis.'"
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,  # LOW = less hallucination
        max_tokens=400
    )

    return response.choices[0].message.content.strip()


def build_rag_prompt(query: str, docs: List["KnowledgeDocument"]) -> str:
    context_blocks = []

    for doc in docs:
        block = f"""
Rule: {doc.rule_id}
Location: {doc.filename}:{doc.lineno}
Message: {doc.message}
Problem: {doc.problem}
Solution: {doc.solution}
"""
        context_blocks.append(block.strip())

    context_text = "\n\n".join(context_blocks)

    return f"""
CONTEXT:
{context_text}

QUESTION:
{query}
"""



@dataclass
class Issue:
    rule_id: str
    filename: str
    lineno: int
    message: str

@dataclass
class KnowledgeDocument:
    id: str
    rule_id: str
    filename: str
    lineno: int
    message: str
    symbol: Optional[str]
    problem: str
    solution: str
    code_context: Optional[str]

class Rule(ABC):
    @abstractmethod
    def check(self, tree: ast.AST, filename: str) -> List[Issue]:
        pass

    @property
    @abstractmethod
    def rule_id(self) -> str:
        pass

def extract_symbol_and_context(tree: ast.AST, lineno: int) -> Tuple[Optional[str], Optional[str]]:
    """
    find the function and the class that contains the line number
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", node.lineno)
            if start <= lineno <= end:
                return node.name, f"{node.name} (lines {start}-{end})"
    return None, None
    
class KnowledgeBuilder:
    def __init__(self, knowledge_base: Dict[str, Dict[str, str]]):
        self.knowledge_base = knowledge_base

    def build(self, issue: Issue, tree: ast.AST) -> KnowledgeDocument:
        kb = self.knowledge_base.get(issue.rule_id, {})
        symbol, context = extract_symbol_and_context(tree, issue.lineno)
        
        return KnowledgeDocument(
            id=f"{issue.rule_id}::{issue.filename}::{issue.lineno}",
            rule_id=issue.rule_id,
            filename=issue.filename,
            lineno=issue.lineno,
            message=issue.message,
            symbol=symbol,
            problem=kb.get("problem", "Unknown problem"),
            solution=kb.get("solution", "No solution available"),
            code_context=context
        )

class KeywordRetriever:
    def __init__(self, documents: List[KnowledgeDocument]):
        self.documents = documents

    def retrieve(self, query: str, top_k: int = 3) -> List[KnowledgeDocument]:
        query_tokens = set(query.lower().split())
        scored_docs = []

        for doc in self.documents:
            text = " ".join([
                doc.rule_id,
                doc.message,
                doc.problem,
                doc.solution,
                doc.symbol or "",
                doc.filename
            ]).lower()

            score = sum(1 for token in query_tokens if token in text)

            if score > 0:
                scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
       
    
class LongFunctionRule(Rule):
    def __init__(self, max_lines: int = 20):
        self.max_lines = max_lines

    @property
    def rule_id(self) -> str:
        return "LONG_FUNCTION"

    def check(self, tree: ast.AST, filename: str) -> List[Issue]:
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                length = (getattr(node, "end_lineno", node.lineno) - node.lineno) + 1
                if length > self.max_lines:
                    issues.append(Issue(
                        self.rule_id, filename, node.lineno,
                        f"Function '{node.name}' is too long ({length} lines)"
                    ))
        return issues

class BareExceptRule(Rule):
    @property
    def rule_id(self) -> str:
        return "BARE_EXCEPT"

    def check(self, tree: ast.AST, filename: str) -> List[Issue]:
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append(Issue(
                    self.rule_id, filename, node.lineno,
                    "Exception handler is missing a specific type"
                ))
        return issues

class TodoCommentRule(Rule):
    @property
    def rule_id(self) -> str:
        return "TODO_COMMENT"

    def check(self, tree: ast.AST, filename: str) -> List[Issue]:
        issues = []
        try:
            with open(filename, "r", encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    if "TODO" in line:
                        issues.append(Issue(self.rule_id, filename, i, "TODO comment found"))
        except Exception:
            pass
        return issues

KNOWLEDGE_BASE: Dict[str, Dict[str, str]] = {
    "LONG_FUNCTION": {
        "problem": "Functions that are too long are hard to read and maintain.",
        "solution": "Refactor the function by breaking it into smaller, more modular functions."
    },
    "BARE_EXCEPT": {
        "problem": "Catching all exceptions can hide bugs and system errors.",
        "solution": "Specify the exception you want to catch (e.g., ValueError, KeyError)."
    },
    "TODO_COMMENT": {
        "problem": "Unresolved TODOs contribute to technical debt.",
        "solution": "Implement the task or track it in a project management tool."
    }
}

class CodeAnalyzer:
    def __init__(self):
        self.rules: List[Rule] = []

    def add_rule(self, rule: Rule):
        self.rules.append(rule)

    def analyze(self, filename: str) -> List[Issue]:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            return []

        all_issues = []
        for rule in self.rules:
            all_issues.extend(rule.check(tree, filename))
        return sorted(all_issues, key=lambda x: x.lineno)

def main():
    if len(sys.argv) < 2:
        print("Usage: python code_analyzer.py <filename>")
        return

    filename = sys.argv[1]
    analyzer = CodeAnalyzer()
    analyzer.add_rule(LongFunctionRule())
    analyzer.add_rule(BareExceptRule())
    analyzer.add_rule(TodoCommentRule())

    try:
        with open(filename, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception as e:
        print(f"Could not parse file for context extraction: {e}")
        tree = ast.parse("") # Empty tree as fallback

    issues = analyzer.analyze(filename)
    builder = KnowledgeBuilder(KNOWLEDGE_BASE)
    documents = [builder.build(issue, tree) for issue in issues]

    print(f"\nAnalysis for: {filename}")
    print("=" * 40)
    
    if not issues:
        print("No issues found!")
    else:
        for doc in documents:
            print(f"[{doc.rule_id}] {doc.filename}:{doc.lineno}")
            print(f"   Message: {doc.message}")
            if doc.symbol:
                print(f"   Context: {doc.symbol}")
            print(f"   Problem: {doc.problem}")
            print(f"   Solution: {doc.solution}")
            print("-" * 20)
    
    retriever = KeywordRetriever(documents)

    print("\nAsk questions about the code (type 'exit' to quit)")
    while True:
        query = input("\n> ").strip()
        if query.lower() == "exit":
            break

        results = retriever.retrieve(query)

        if not results:
            print("No relevant information found.")
            continue

        print("\nRetrieved Context:")
        print("=" * 30)
        for doc in results:
            print(f"- [{doc.rule_id}] {doc.filename}:{doc.lineno}")
            print(f"  Message: {doc.message}")
            print(f"  Problem: {doc.problem}")
            print(f"  Solution: {doc.solution}")
            print()



        prompt = build_rag_prompt(query, results)
        answer = call_llm(prompt)

        print("\nAnswer:")
        print("=" * 30)
        print(answer)

def analyze_file_as_json(filename: str):
    analyzer = CodeAnalyzer()
    analyzer.add_rule(LongFunctionRule())
    analyzer.add_rule(BareExceptRule())
    analyzer.add_rule(TodoCommentRule())

    with open(filename, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    issues = analyzer.analyze(filename)
    builder = KnowledgeBuilder(KNOWLEDGE_BASE)
    documents = [builder.build(issue, tree) for issue in issues]

    return [
        {
            "rule": doc.rule_id,
            "line": doc.lineno,
            "message": doc.message,
            "problem": doc.problem,
            "solution": doc.solution,
        }
        for doc in documents
    ]

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--json":
        result = analyze_file_as_json(sys.argv[2])
        print(json.dumps(result))
    else:
        main()
