"""Optional runtime wiring is verified with fake modules and synthetic audio."""

import math
import sys
import wave
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest


def wav_file(tmp_path, seconds=2):
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setparams((1, 2, 4, 0, "NONE", "not compressed"))
        stream.writeframes(b"\0\0" * (4 * seconds))
    return path


class Array(list):
    def all(self):
        return all(self)

    def mean(self, axis):
        assert axis == 1
        return Array(sum(row) / len(row) for row in self)

    def __gt__(self, value):
        return Array(x > value for x in self)


def fake_numpy(monkeypatch):
    def finite(data):
        flat = [x for row in data for x in row] if data and isinstance(data[0], list) else data
        return Array(math.isfinite(x) for x in flat)

    monkeypatch.setitem(
        sys.modules,
        "numpy",
        NS(isfinite=finite, any=any, abs=lambda data: Array(abs(x) for x in data)),
    )


def test_pcm_window_is_complete_mono_and_uses_selected_samples(tmp_path, monkeypatch):
    from shono.api.audio import window_wav_bytes

    path = wav_file(tmp_path)
    fake_numpy(monkeypatch)
    observed = {}

    class Stream:
        samplerate, frames = 4, 8

        def __init__(self, filename):
            assert filename == str(path)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def seek(self, first):
            observed["first"] = first

        def read(self, *, frames, dtype, always_2d):
            observed["read"] = (frames, dtype, always_2d)
            return Array([[0.6, 0.2]] * frames)

    def write(buffer, data, sr, *, format, subtype):
        observed["write"] = (list(data), sr, format, subtype)
        buffer.write(b"fake PCM")

    monkeypatch.setitem(sys.modules, "soundfile", NS(SoundFile=Stream, write=write))
    assert window_wav_bytes(str(path), 0.5, 1) == (b"fake PCM", 4)
    assert observed["first"] == 2
    assert observed["read"] == (4, "float64", True)
    assert observed["write"] == ([0.4] * 4, 4, "WAV", "PCM_16")


@pytest.mark.parametrize("sample", [float("nan"), float("inf"), 1.5])
def test_pcm_invalid_samples_reject_before_encoding(tmp_path, monkeypatch, sample):
    from shono.api.audio import window_wav_bytes

    path = wav_file(tmp_path)
    fake_numpy(monkeypatch)

    class Stream:
        samplerate, frames = 4, 8

        def __init__(self, *_):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def seek(self, *_):
            pass

        def read(self, **kwargs):
            return Array([[sample]] * kwargs["frames"])

    def fail(*args, **kwargs):
        pytest.fail("invalid samples must not be encoded")

    monkeypatch.setitem(sys.modules, "soundfile", NS(SoundFile=Stream, write=fail))
    with pytest.raises(ValueError):
        window_wav_bytes(str(path))


def test_deepgram_7_media_call_keyword_contract(monkeypatch):
    from shono.api.deepgram import DeepgramTranscriber

    observed = {}

    def transcribe_file(*, request, model, language, smart_format, request_options):
        observed["request"] = (request, model, language, smart_format, request_options)
        return NS(results=NS(channels=[NS(alternatives=[NS(transcript=" কথা ")])]))

    def client(*, api_key):
        observed["key"] = api_key
        return NS(listen=NS(v1=NS(media=NS(transcribe_file=transcribe_file))))

    monkeypatch.setitem(sys.modules, "deepgram", NS(DeepgramClient=client))
    monkeypatch.setattr("shono.api.deepgram.window_wav_bytes", lambda *args: (b"PCM", 16000))
    adapter = DeepgramTranscriber("fake-key")
    assert adapter.transcribe("synthetic.wav") == "কথা"
    assert observed == {
        "key": "fake-key",
        "request": (b"PCM", "nova-3", "bn", True, {"max_retries": 0}),
    }


def test_google_chirp_short_request_and_retries_disabled(monkeypatch):
    from shono.api.google_chirp import GoogleChirpTranscriber

    observed = {}

    def encode(*args, **kwargs):
        observed["limit"] = kwargs["max_duration_s"]
        return b"PCM", 16000

    def recognize(*, request, retry):
        observed["request"], observed["retry"] = request, retry
        return NS(results=[NS(alternatives=[NS(transcript="বাংলা")]), NS(alternatives=[])])

    cloud_speech = NS(
        RecognitionConfig=lambda **kw: NS(**kw),
        AutoDetectDecodingConfig=lambda: "auto",
        RecognizeRequest=lambda **kw: NS(**kw),
    )
    monkeypatch.setitem(sys.modules, "google.cloud.speech_v2.types", NS(cloud_speech=cloud_speech))
    monkeypatch.setattr("shono.api.google_chirp.window_wav_bytes", encode)
    adapter = GoogleChirpTranscriber("fake-project")
    monkeypatch.setattr(
        adapter, "_client_and_recognizer", lambda: (NS(recognize=recognize), "recognizer")
    )
    assert adapter.transcribe("synthetic.wav", 0, 59) == "বাংলা"
    assert observed["limit"] == 60
    assert observed["retry"] is None
    assert observed["request"].config.model == "chirp_2"
    assert observed["request"].content == b"PCM"


def test_google_window_rejects_before_sdk_loading(monkeypatch):
    from shono.api.google_chirp import GoogleChirpTranscriber

    def reject(*args, **kwargs):
        assert kwargs["max_duration_s"] == 60
        raise ValueError("window shorter than 60 s")

    monkeypatch.setattr("shono.api.google_chirp.window_wav_bytes", reject)
    adapter = GoogleChirpTranscriber("fake-project")
    monkeypatch.setattr(
        adapter, "_client_and_recognizer", lambda: pytest.fail("SDK must remain unloaded")
    )
    with pytest.raises(ValueError, match="60"):
        adapter.transcribe("synthetic.wav", 0, 60)


def test_pyannote_4_output_token_device_and_cache(tmp_path, monkeypatch):
    from shono.diarize import PyannoteDiarizer

    path = wav_file(tmp_path)
    observed = {"loads": 0}

    class Pipeline:
        def to(self, device):
            observed["device"] = device

        def __call__(self, filename):
            assert filename == str(path)
            return NS(speaker_diarization=[(NS(start=0, end=1), "A")])

        @classmethod
        def from_pretrained(cls, model, *, token):
            observed["loads"] += 1
            observed["token"], observed["model"] = token, model
            return cls()

    monkeypatch.setitem(sys.modules, "torch", NS(device=lambda name: f"device:{name}"))
    monkeypatch.setitem(sys.modules, "pyannote.audio", NS(Pipeline=Pipeline))
    adapter = PyannoteDiarizer(hf_token="fake-token", device="cpu")
    assert adapter.diarize(str(path))[0].speaker == "A"
    assert adapter.diarize(str(path))[0].duration_s == 1
    assert observed["loads"] == 1
    assert observed["token"] == "fake-token"
    assert observed["device"] == "device:cpu"


def test_missing_pyannote_audio_never_loads_weights(tmp_path, monkeypatch):
    from shono.diarize import PyannoteDiarizer

    adapter = PyannoteDiarizer()
    monkeypatch.setattr(adapter, "_load", lambda: pytest.fail("weights must remain unloaded"))
    with pytest.raises(ValueError, match="missing audio"):
        adapter.diarize(str(tmp_path / "missing.wav"))


def test_hf_whisper_fake_generation_and_observed_settings(tmp_path, monkeypatch):
    from shono.transcribe import ShortFormWhisperTranscriber

    path = wav_file(tmp_path)
    fake_numpy(monkeypatch)
    observed = {}

    class Features:
        def to(self, device):
            observed["features_device"] = device
            return self

    class Processor:
        @classmethod
        def from_pretrained(cls, name):
            observed["processor"] = name
            return cls()

        def __call__(self, audio, *, sampling_rate, return_tensors):
            observed["features"] = (len(audio), sampling_rate, return_tensors)
            return NS(input_features=Features())

        def batch_decode(self, ids, *, skip_special_tokens):
            assert ids == [1] and skip_special_tokens
            return [" বাংলা "]

    class Model:
        config = NS(_commit_hash="fake-revision")
        generation_config = NS(to_dict=lambda: {"num_beams": 5, "do_sample": False})

        @classmethod
        def from_pretrained(cls, name):
            observed["model"] = name
            return cls()

        def to(self, device):
            observed["device"] = device

        def eval(self):
            observed["eval"] = True

        def generate(self, features, *, language, task):
            observed["generate"] = (language, task)
            return [1]

    monkeypatch.setitem(
        sys.modules, "torch", NS(no_grad=nullcontext, cuda=NS(is_available=lambda: False))
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        NS(WhisperProcessor=Processor, WhisperForConditionalGeneration=Model),
    )
    monkeypatch.setitem(
        sys.modules, "librosa", NS(load=lambda *args, **kw: (Array([0] * 32000), 16000))
    )
    adapter = ShortFormWhisperTranscriber("local-checkpoint")
    assert adapter.transcribe(str(path)) == "বাংলা"
    assert observed["features"] == (32000, 16000, "pt")
    assert observed["generate"] == ("bn", "transcribe")
    assert observed["features_device"] == "cpu"
    assert adapter.observed_runtime_context()["generation_config"]["num_beams"] == 5
    assert adapter.observed_runtime_context()["model_revision"] == "fake-revision"


def test_faster_whisper_does_not_invent_missing_word_timestamps(tmp_path, monkeypatch):
    from shono.transcribe.pipeline import FasterWhisperTranscriber

    path = wav_file(tmp_path)
    fake_numpy(monkeypatch)
    monkeypatch.setitem(
        sys.modules, "librosa", NS(load=lambda *a, **kw: (Array([0] * 16000), 16000))
    )
    segment = NS(text="কথা", words=None, avg_logprob=-0.2, compression_ratio=1, no_speech_prob=0)
    adapter = FasterWhisperTranscriber("local-checkpoint", device="cpu", compute_type="int8")
    monkeypatch.setattr(
        adapter, "_load", lambda: NS(transcribe=lambda *a, **kw: (iter([segment]), None))
    )
    result = adapter.transcribe_chunk(str(path), 0, 1)
    assert result.text == "কথা"
    assert result.words == ()


def test_silero_model_load_is_cached_and_validates_input_first(tmp_path, monkeypatch):
    from shono.transcribe.pipeline import SileroVAD

    path = wav_file(tmp_path)
    loads = []

    def load(*args, **kwargs):
        loads.append((args, kwargs))
        return "model", (
            lambda *a, **kw: [{"start": 0, "end": 16000}],
            None,
            lambda *a, **kw: "audio",
            None,
        )

    monkeypatch.setitem(sys.modules, "torch", NS(hub=NS(load=load)))
    vad = SileroVAD()
    with pytest.raises(ValueError):
        vad.detect(str(tmp_path / "missing.wav"))
    assert loads == []
    assert vad.detect(str(path))[0].duration_s == 1
    vad.detect(str(path))
    assert len(loads) == 1
