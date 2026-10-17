def test_pinned_imports():
    import google.adk  # noqa: F401
    import tau2  # noqa: F401
    from google.adk.skills import load_skills_from_dir  # noqa: F401
    from google.adk.tools.skill_toolset import SkillToolset  # noqa: F401
    from tau2.gym.gym_agent import AgentGymEnv  # noqa: F401
