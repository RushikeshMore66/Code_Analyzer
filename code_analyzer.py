import ast
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class Issue:
    rule_id: str
    filename: str
    lineno: int
    message: str

class Rule(ABC):
    @abstractmethod
    def check(self, tree: ast.AST, filename: str) -> List[Issue]:
        pass

    @property
    @abstractmethod
    def rule_id(self) -> str:
        pass

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

class Explainer:
    def explain(self, issue: Issue) -> str:
        kb = KNOWLEDGE_BASE.get(issue.rule_id, {})
        if not kb:
            return "No additional information available."
        return f"   - Problem: {kb['problem']}\n   - Solution: {kb.get('solution', 'No solution found.')}"

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

    issues = analyzer.analyze(filename)
    explainer = Explainer()

    print(f"\nAnalysis for: {filename}")
    print("=" * 40)
    
    if not issues:
        print("No issues found!")
    else:
        for issue in issues:
            print(f"\n[{issue.rule_id}] {issue.filename}:{issue.lineno} - {issue.message}")
            print(explainer.explain(issue))

if __name__ == "__main__":
    main()
