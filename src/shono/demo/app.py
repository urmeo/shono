"""The public demo (Hugging Face Space): upload Bengali audio → speaker transcript.

A thin Gradio wrapper over :func:`shono.demo.pipeline.transcribe_recording`, wiring
the real Silero VAD + faster-whisper + pyannote stack. gradio and the model are
gated (installed on the Space, not in the scoring env), so both the ``gradio``
import and the pipeline construction are lazy — importing this module needs
neither. Launching the app is what pulls them in.
"""

from __future__ import annotations


def make_transcribe_fn(model_dir: str, *, hf_token: str | None = None, language: str = "bn"):
    """Build the callable the UI invokes: a recording path → a formatted transcript."""
    from shono.demo.pipeline import format_transcript, transcribe_recording
    from shono.diarize import PyannoteDiarizer
    from shono.transcribe import FasterWhisperTranscriber, LongFormTranscriber, SileroVAD

    pipeline = LongFormTranscriber(
        SileroVAD(), FasterWhisperTranscriber(model_dir, language=language)
    )
    diarizer = PyannoteDiarizer(hf_token=hf_token)

    def run(audio_path: str) -> str:
        import soundfile as sf

        if not audio_path:
            return ""
        duration = sf.info(audio_path).duration
        transcript = transcribe_recording(audio_path, duration, pipeline, diarizer)
        return format_transcript(transcript)

    return run


def build_app(model_dir: str, *, hf_token: str | None = None, language: str = "bn"):
    """Build the Gradio ``Blocks`` app (gradio required)."""
    import gradio as gr

    run = make_transcribe_fn(model_dir, hf_token=hf_token, language=language)
    with gr.Blocks(title="Shono — Bengali ASR that survives the real world") as app:
        gr.Markdown(
            "# শোনো · Shono\n"
            "Upload Bengali audio — a lecture, a podcast, an interview — and get a "
            "speaker-attributed transcript. Long-form, diarized, code-switch aware."
        )
        audio = gr.Audio(type="filepath", label="Bengali audio")
        button = gr.Button("Transcribe", variant="primary")
        output = gr.Textbox(label="Speaker-attributed transcript", lines=20)
        button.click(run, inputs=audio, outputs=output)
    return app


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(prog="python -m shono.demo.app")
    parser.add_argument("--model", required=True, help="path to the CTranslate2 model directory")
    parser.add_argument("--language", default="bn")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    app = build_app(args.model, hf_token=os.environ.get("HF_TOKEN"), language=args.language)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
