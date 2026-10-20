"""End-to-end: `strands-decider ask` downloads a published checkpoint and answers.

Marked `integration` because it hits the Hugging Face Hub and runs a real forward
pass on ~2B weights, so it is slow and bandwidth-heavy. The default pytest run
excludes it (see pyproject [tool.pytest.ini_options]); opt in with
`pytest -m integration`.

The model id, revision and HF cache location are env-overridable so a developer
can point the test at a smaller/newer checkpoint or at a pre-warmed cache.
"""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from strands_decider import cli
from strands_decider.schema import SystemOneResponse

DEFAULT_MODEL = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"


@pytest.mark.integration
def test_ask_downloads_hub_checkpoint_and_answers(tmp_path, monkeypatch):
    model = os.environ.get("STRANDS_DECIDER_HF_MODEL", DEFAULT_MODEL)
    # Isolate the download so a fresh run tests the download path end-to-end.
    # Point STRANDS_DECIDER_HF_CACHE at a shared cache to reuse weights between runs.
    monkeypatch.setenv("HF_HOME", os.environ.get("STRANDS_DECIDER_HF_CACHE", str(tmp_path / "hf")))
    # ask_cmd flips this and the next test in the session would otherwise see it set.
    monkeypatch.setattr(cli, "_exit_fast", False)

    result = CliRunner(mix_stderr=False).invoke(
        cli.app,
        [
            "ask",
            model,
            "--state",
            "The sky is blue on a clear day.",
            "--noul",
            "Is the statement true?",
            "--device",
            "cpu",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stderr or result.output
    assert cli._exit_fast, "ask should request the fast exit once it has printed its answer"

    response = SystemOneResponse.model_validate_json(result.stdout.strip())
    [(name, answer)] = response.answers.items()
    assert name == "noul_0"
    assert answer.type == "noul"
    assert 0.0 <= answer.noul <= 1.0
