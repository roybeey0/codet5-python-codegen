import os
os.environ["GRADIO_DEFAULT_LOCALE"] = "en"
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LANGUAGE"] = "en"

import gradio as gr
from inference import load_model, generate_code, DEMO_DOCSTRINGS

model, tokenizer, device = load_model()

def generate(docstring, num_candidates, num_beams, temperature, top_p, max_output_tokens):
    if not docstring.strip():
        return "⚠️ Please enter a docstring."

    results = generate_code(
        docstring=docstring,
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_target_length=max_output_tokens,
        num_beams=max(num_beams, num_candidates),
        num_return_sequences=num_candidates,
        temperature=temperature,
        top_p=top_p,
    )

    output = ""
    for i, code in enumerate(results, 1):
        output += f"# -- Candidate {i} --\n"
        output += code.strip() + "\n\n"

    return output.strip()

CSS = """
#title { text-align: center; font-size: 2rem; font-weight: 700; margin-bottom: 4px; }
#subtitle { text-align: center; color: #6b7280; margin-bottom: 16px; }
"""

DESCRIPTION = """
<div id='title'>🧠 CodeT5 Python Code Generator</div>
<div id='subtitle'>Fine-tuned on CodeSearchNet Python · Salesforce/codet5-base · HuggingFace Transformers</div>
"""

EXAMPLES = [[d, 2, 5, 0.8, 0.95, 256] for d in DEMO_DOCSTRINGS[:6]]

with gr.Blocks(css=CSS) as demo:
    gr.HTML(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=2):
            docstring_input = gr.Textbox(
                label="📝 Natural Language Docstring",
                placeholder="e.g. Calculate the factorial of n recursively.",
                lines=4,
            )

            with gr.Accordion("⚙️ Generation Settings", open=False):
                num_candidates = gr.Slider(1, 5, value=2, step=1, label="Number of Candidates")
                num_beams = gr.Slider(1, 10, value=5, step=1, label="Beam Search Width")
                temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p (Nucleus Sampling)")
                max_tokens = gr.Slider(64, 512, value=256, step=32, label="Max Output Tokens")

            generate_btn = gr.Button("🚀 Generate Code", variant="primary", size="lg")

        with gr.Column(scale=3):
            code_output = gr.Code(
                label="💻 Generated Python Code",
                language="python",
                lines=20,
            )

    generate_btn.click(
        fn=generate,
        inputs=[docstring_input, num_candidates, num_beams, temperature, top_p, max_tokens],
        outputs=code_output,
    )
    docstring_input.submit(
        fn=generate,
        inputs=[docstring_input, num_candidates, num_beams, temperature, top_p, max_tokens],
        outputs=code_output,
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[docstring_input, num_candidates, num_beams, temperature, top_p, max_tokens],
        outputs=code_output,
        fn=generate,
        cache_examples=False,
        label="💡 Click an example to try it",
    )

    gr.Markdown("""
    ---
    **How it works:**
    1. Enter a natural-language description of a Python function.
    2. The fine-tuned CodeT5 transformer decodes the docstring into source code using beam search.
    3. Multiple candidates are ranked by beam score (best first).

    **Model:** `Salesforce/codet5-base` fine-tuned on CodeSearchNet Python  
    **Framework:** PyTorch + HuggingFace Transformers
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)