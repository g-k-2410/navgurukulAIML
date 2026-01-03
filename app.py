import gradio as gr
from pipeline.ocr import extract_screen_text
from pipeline.stt import transcribe_audio
from pipeline.interview_engine import generate_question
from pipeline.evaluation import evaluate_response

context_store = ""

def process_step(screen_image, audio):
    global context_store

    if screen_image is not None:
        context_store += extract_screen_text(screen_image)

    if audio is not None:
        response_text = transcribe_audio(audio)
        question = generate_question(context_store, response_text)
        score = evaluate_response(context_store, response_text)

        return question, score

    return "", ""

with gr.Blocks() as demo:
    gr.Markdown("# AI-Driven Automated Project Interviewer")

    with gr.Row():
        screen = gr.Image(label="Upload Screen / Slide / Code Screenshot")
        audio = gr.Audio(type="filepath", label="Explain Your Project")

    question = gr.Textbox(label="AI Interviewer Question")
    score = gr.JSON(label="Evaluation & Feedback")

    btn = gr.Button("Submit")
    btn.click(process_step, inputs=[screen, audio], outputs=[question, score])

demo.launch()
