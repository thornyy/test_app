
import os
import gradio as gr
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Load API key
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing from your .env file.")

# Create AI model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    api_key=api_key
)

# Grammar checker function
def check_grammar(text):
    if not text.strip():
        return "Please enter some text.", ""

    messages = [
        SystemMessage(
            content=(
                "You are an AI Grammar Checker. "
                "Correct spelling, grammar, punctuation, and sentence structure. "
                "Preserve the original meaning and tone. "
                "Explain the important corrections in simple language. "
                "Return your answer using exactly these two sections:\n"
                "CORRECTED TEXT:\n"
                "EXPLANATION:"
            )
        ),
        HumanMessage(content=f"Check this text:\n\n{text}")
    ]

    response = llm.invoke(messages)
    result = response.content

    # Split corrected text and explanation
    if "EXPLANATION:" in result:
        parts = result.split("EXPLANATION:", 1)
        corrected = parts[0].replace("CORRECTED TEXT:", "").strip()
        explanation = parts[1].strip()
    else:
        corrected = result
        explanation = "No explanation was provided."

    return corrected, explanation

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# ✍️ AI Grammar Checker")
    gr.Markdown(
        "Correct spelling, grammar, punctuation, and sentence structure."
    )

    original_text = gr.Textbox(
        label="Original Text",
        placeholder="Enter your sentences here...",
        lines=7
    )

    check_button = gr.Button("Check Grammar", variant="primary")

    corrected_text = gr.Textbox(
        label="Corrected Text",
        lines=7
    )

    explanation = gr.Textbox(
        label="Explanation",
        lines=5
    )

    check_button.click(
        fn=check_grammar,
        inputs=original_text,
        outputs=[corrected_text, explanation]
    )

demo.launch()
