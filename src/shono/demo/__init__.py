"""The public demo: transcription orchestration, formatting, and the model card.

The orchestration, formatter, and model-card generator are pure and imported
eagerly; the Gradio app keeps its ``gradio`` import lazy, so this package loads
without gradio or a GPU.
"""

from shono.demo.app import build_app, make_transcribe_fn
from shono.demo.model_card import render_model_card
from shono.demo.pipeline import format_transcript, transcribe_recording

__all__ = [
    "build_app",
    "format_transcript",
    "make_transcribe_fn",
    "render_model_card",
    "transcribe_recording",
]
