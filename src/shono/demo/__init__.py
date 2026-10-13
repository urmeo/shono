"""Transcription demo and model-card rendering; Gradio loads when the app starts."""

from importlib import import_module

from shono.demo.model_card import render_model_card
from shono.demo.pipeline import format_transcript, transcribe_recording

__all__ = [
    "build_app",
    "format_transcript",
    "make_transcribe_fn",
    "render_model_card",
    "transcribe_recording",
]


def __getattr__(name: str):
    if name in {"build_app", "make_transcribe_fn"}:
        return getattr(import_module("shono.demo.app"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
