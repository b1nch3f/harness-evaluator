import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "test_bench"


def test_pinned_imports():
    import google.adk  # noqa: F401
    from google.adk.models.lite_llm import LiteLlm  # noqa: F401
    from google.adk.skills import load_skills_from_dir  # noqa: F401
    from google.adk.tools.skill_toolset import SkillToolset  # noqa: F401


def test_runtime_code_never_imports_tau2():
    """CLAUDE.md rule 2: tau2 code may run only in tests."""
    offenders = []
    for path in SRC.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            offenders += [f"{path.name}: {n}" for n in names if n.split(".")[0] == "tau2"]
    assert offenders == []
