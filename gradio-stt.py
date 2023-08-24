from transformers import pipeline

model_id = "sanchit-gandhi/whisper-small-dv"  # update with your model id
pipe = pipeline("automatic-speech-recognition", model=model_id)

def transcribe_speech(filepath):
    output = pipe(
        filepath,
        max_new_tokens=256,
        generate_kwargs={
            "task": "transcribe",
            "language": "sinhalese",
        },  # update with the language you've fine-tuned on
        chunk_length_s=30,
        batch_size=8,
    )
    return output["text"]

import gradio as gr

demo = gr.Blocks()

mic_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs=gr.outputs.Textbox(),
)

file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(source="upload", type="filepath"),
    outputs=gr.outputs.Textbox(),
)

if __name__ == '__main__':
    with demo:
        gr.TabbedInterface(
            [mic_transcribe, file_transcribe],
            ["Transcribe Microphone", "Transcribe Audio File"],
        )

    demo.launch(debug=True)