import wave
from pathlib import Path

import pytest

from mmm.config.config_manager import ConfigManager
from mmm.config.defaults import DEFAULT_CONFIG
from mmm.core.file_processor import FileProcessor
from mmm.sanitization.metadata_cleaner import MetadataCleaner


def _write_minimal_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(b"\x00\x00" * 8)


def test_wav_metadata_cleaning_does_not_mutate_original(monkeypatch, tmp_path):
    input_file = tmp_path / "source.wav"
    output_file = tmp_path / "clean.wav"
    _write_minimal_wav(input_file)
    original_bytes = input_file.read_bytes()

    class FakeWave:
        def __init__(self, path):
            self.path = Path(path)
            self.tags = {"INAM": "title"}

        def save(self):
            self.path.write_bytes(b"mutagen wrote here")

    monkeypatch.setattr("mmm.sanitization.metadata_cleaner.WAVE", FakeWave)

    result = MetadataCleaner().clean_file(input_file, output_file)

    assert result["success"] is True
    assert "mutagen_clear" in result["methods_used"]
    assert input_file.read_bytes() == original_bytes
    assert output_file.exists()


def test_empty_yaml_config_file_loads_as_defaults(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("", encoding="utf-8")

    manager = ConfigManager(config_file=config_file)

    assert manager.get("version") == DEFAULT_CONFIG["version"]
    assert manager.get("paranoia_level") == DEFAULT_CONFIG["paranoia_level"]


def test_non_mapping_yaml_config_file_is_rejected(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="YAML mapping"):
        ConfigManager(config_file=config_file)


def test_output_path_rejects_sibling_directory_prefix_bypass(tmp_path):
    processor = object.__new__(FileProcessor)
    processor.config = {
        "batch_processing.naming_pattern": "../cleaned_evil/{name}{ext}",
    }

    with pytest.raises(ValueError, match="outside output directory"):
        processor._generate_output_path(tmp_path / "track.wav", tmp_path / "cleaned")


def test_output_path_allows_files_inside_output_directory(tmp_path):
    processor = object.__new__(FileProcessor)
    processor.config = {
        "batch_processing.naming_pattern": "{name}_clean{ext}",
    }

    output_path = processor._generate_output_path(
        tmp_path / "track.wav", tmp_path / "cleaned"
    )

    assert output_path == (tmp_path / "cleaned" / "track_clean.wav").resolve()
