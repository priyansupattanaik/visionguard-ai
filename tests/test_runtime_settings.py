from visionguard.runtime.settings import PipelineSettings


def test_pipeline_settings_bound_invalid_runtime_values(monkeypatch):
    monkeypatch.delenv("VISION_GUARD_OUT_DIR", raising=False)
    monkeypatch.setenv("MIN_EVIDENCE_CONFIDENCE", "not-a-number")
    monkeypatch.setenv("MAX_EXHAUSTIVE_VERIFICATION_FRAMES", "0")

    settings = PipelineSettings.from_env("custom-output")

    assert settings.out_dir == "custom-output"
    assert settings.minimum_evidence_confidence == 0.25
    assert settings.max_exhaustive_verification_frames == 1


def test_pipeline_settings_honour_explicit_output_override(monkeypatch):
    monkeypatch.setenv("VISION_GUARD_OUT_DIR", "runtime-output")

    assert PipelineSettings.from_env("constructor-output").out_dir == "runtime-output"
