from test_bench.model import ModelSettings, completion_kwargs, endpoint_label

S = ModelSettings(
    model="openai/openai/gpt-oss-120b",
    api_base="https://api.example.com/v2/abc/openai/v1",
    api_key="secret-key",
    temperature=0.0,
    reasoning_effort="high",
)


def test_reasoning_effort_goes_in_extra_body():
    kw = completion_kwargs(S)
    assert kw["extra_body"] == {"reasoning_effort": "high"}
    assert "reasoning_effort" not in kw


def test_public_settings_have_no_key():
    pub = S.public()
    assert "api_key" not in pub
    assert "secret-key" not in str(pub)


def test_endpoint_label_drops_credentials():
    assert endpoint_label("https://user:pw@host.example/v1?key=x") == "host.example/v1"


def test_optional_sampling_settings_reach_litellm(monkeypatch):
    from test_bench.model import REQUEST_TIMEOUT_S, agent_settings, completion_kwargs

    for k, v in {"X_MODEL": "openai/m", "X_API_BASE": "https://h.example/v1", "X_API_KEY": "k",
                 "X_TOP_P": "0.95", "X_MAX_TOKENS": "16384", "X_TIMEOUT": "300"}.items():
        monkeypatch.setenv(k, v)
    s = agent_settings("X")
    kw = completion_kwargs(s)
    assert (kw["top_p"], kw["max_tokens"], kw["timeout"]) == (0.95, 16384, 300)
    assert s.public()["top_p"] == 0.95 and "api_key" not in s.public()
    monkeypatch.delenv("X_TOP_P"), monkeypatch.delenv("X_MAX_TOKENS"), monkeypatch.delenv("X_TIMEOUT")
    kw = completion_kwargs(agent_settings("X"))
    assert "top_p" not in kw and "max_tokens" not in kw and kw["timeout"] == REQUEST_TIMEOUT_S
