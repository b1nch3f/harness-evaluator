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
