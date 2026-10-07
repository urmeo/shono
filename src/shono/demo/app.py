"""Build a lazy local Gradio transcription app."""

from __future__ import annotations


def make_transcribe_fn(
    model_dir: str,
    *,
    hf_token: str | None = None,
    language: str = "bn",
    device: str = "cpu",
    compute_type: str = "int8",
):
    """Build the callable the UI invokes: a recording path → a formatted transcript."""
    from shono.demo.pipeline import format_transcript, transcribe_recording
    from shono.diarize import PyannoteDiarizer
    from shono.transcribe import FasterWhisperTranscriber, LongFormTranscriber, SileroVAD
    from shono.transcribe.convert import validate_ct2_checkpoint

    model_dir = str(validate_ct2_checkpoint(model_dir))

    pipeline = LongFormTranscriber(
        SileroVAD(),
        FasterWhisperTranscriber(
            model_dir, language=language, device=device, compute_type=compute_type
        ),
    )
    diarizer = PyannoteDiarizer(hf_token=hf_token, device=device)

    def run(audio_path: str) -> str:
        if not audio_path:
            return ""
        from shono.data.audio_paths import validate_audio_window

        duration = validate_audio_window(audio_path)
        transcript = transcribe_recording(audio_path, duration, pipeline, diarizer)
        return format_transcript(transcript)

    return run


def build_app(
    model_dir: str,
    *,
    hf_token: str | None = None,
    language: str = "bn",
    device: str = "cpu",
    compute_type: str = "int8",
):
    """Build the Gradio ``Blocks`` app (gradio required)."""
    import gradio as gr

    run = make_transcribe_fn(
        model_dir, hf_token=hf_token, language=language, device=device, compute_type=compute_type
    )
    with gr.Blocks(title="Shono | Bengali transcription") as app:
        gr.Markdown(
            "# শোনো · Shono\n"
            "Upload Bengali audio for transcription. Speaker labels describe the dominant "
            "speaker per segment; missing word timings are marked. Models load on first use."
        )
        audio = gr.Audio(type="filepath", label="Bengali audio")
        button = gr.Button("Transcribe", variant="primary")
        output = gr.Textbox(label="Speaker-attributed transcript", lines=20)
        button.click(run, inputs=audio, outputs=output)
    return app


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(prog="python -m shono.demo.app", allow_abbrev=False)
    parser.add_argument("--model", required=True, help="path to the CTranslate2 model directory")
    parser.add_argument("--language", default="bn")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args(argv)
    if not Path(args.model).is_dir():
        parser.error("--model must be an existing CTranslate2 model directory")
    try:
        from shono.transcribe.convert import validate_ct2_checkpoint

        validate_ct2_checkpoint(args.model)
        app = build_app(
            args.model,
            hf_token=os.environ.get("HF_TOKEN"),
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
        )
        app.launch(share=args.share)
        return 0
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"error starting demo: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
