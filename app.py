# AI-POWERED SEMANTIC DUPLICATE DETECTOR
# ============================================================

import os
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Validated threshold from the project experiments
FINAL_THRESHOLD = 0.40

# Store session comparisons
session_scores = []


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Sentence Transformer model...")

model = SentenceTransformer(MODEL_NAME)

print("Model loaded successfully.")


# ============================================================
# SEMANTIC SIMILARITY FUNCTION
# ============================================================

def calculate_similarity(question_a, question_b):

    if not question_a.strip() or not question_b.strip():
        return None

    embeddings = model.encode(
        [question_a, question_b],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    # Convert to percentage
    similarity_percentage = float(similarity * 100)

    return similarity_percentage


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_similarity(question_a, question_b):

    if not question_a.strip() or not question_b.strip():

        return (
            """
            <div class="result-card error">
                <div class="result-title">⚠️ Input Required</div>
                <div>Please enter both questions before analyzing.</div>
            </div>
            """,
            0,
            f"{FINAL_THRESHOLD * 100:.0f}%",
            "Please provide both questions.",
            None,
            None
        )

    similarity_percentage = calculate_similarity(
        question_a,
        question_b
    )

    # Save score for session visualization
    session_scores.append(similarity_percentage)

    # Convert threshold to percentage
    threshold_percentage = FINAL_THRESHOLD * 100

    # Classification
    if similarity_percentage >= threshold_percentage:

        prediction = "Semantic duplicate"

        explanation = (
            f"Similarity came out to {similarity_percentage:.2f}%, "
            f"which is at or above the {threshold_percentage:.0f}% "
            f"decision threshold — so the pair is classified as "
            f"a semantic duplicate."
        )

        result_html = f"""
        <div class="result-card duplicate">
            <div class="result-title">
                🟢 Semantic duplicate
            </div>

            <div class="result-subtitle">
                These questions carry highly similar meaning.
            </div>
        </div>
        """

    else:

        prediction = "Not a semantic duplicate"

        explanation = (
            f"Similarity came out to {similarity_percentage:.2f}%, "
            f"which is below the {threshold_percentage:.0f}% "
            f"decision threshold — so the pair is classified "
            f"as non-duplicate."
        )

        result_html = f"""
        <div class="result-card nonduplicate">
            <div class="result-title">
                🔵 Not a semantic duplicate
            </div>

            <div class="result-subtitle">
                These questions do not have sufficiently similar meaning.
            </div>
        </div>
        """

    # ========================================================
    # SESSION GRAPH
    # ========================================================

    fig, ax = plt.subplots(figsize=(8, 3.5))

    x_values = range(1, len(session_scores) + 1)

    ax.bar(
        x_values,
        session_scores
    )

    ax.axhline(
        y=threshold_percentage,
        linestyle="--",
        label=f"Decision threshold = {threshold_percentage:.0f}%"
    )

    ax.set_title(
        "Similarity across this session's comparisons"
    )

    ax.set_xlabel("Comparison")

    ax.set_ylabel("Similarity %")

    ax.set_ylim(0, 100)

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    # ========================================================
    # GENERATE REPORT
    # ========================================================

    report_text = f"""
AI-POWERED SEMANTIC DUPLICATE DETECTOR
========================================

Question A:
{question_a}

Question B:
{question_b}

----------------------------------------
RESULT
----------------------------------------

Prediction:
{prediction}

Similarity Score:
{similarity_percentage:.2f}%

Decision Threshold:
{threshold_percentage:.0f}%

Explanation:
{explanation}

----------------------------------------
MODEL
----------------------------------------

Sentence Transformer:
{MODEL_NAME}

Similarity Metric:
Cosine Similarity

Validated Decision Threshold:
{FINAL_THRESHOLD:.2f}
"""

    report_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".txt",
        prefix="semantic_duplicate_report_"
    )

    report_file.write(
        report_text.encode("utf-8")
    )

    report_file.close()

    return (
        result_html,
        round(similarity_percentage, 2),
        f"{threshold_percentage:.0f}%",
        explanation,
        fig,
        report_file.name
    )


# ============================================================
# CLEAR FUNCTION
# ============================================================

def clear_all():

    session_scores.clear()

    return (
        "",
        "",
        "",
        0,
        f"{FINAL_THRESHOLD * 100:.0f}%",
        "",
        None,
        None
    )


# ============================================================
# CUSTOM CSS
# ============================================================

custom_css = """

body {
    background: #090d09 !important;
}

.gradio-container {
    max-width: 1250px !important;
    margin: auto !important;
    background: #090d09 !important;
    color: white !important;
}

.hero {
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 30px;

    background:
        radial-gradient(
            circle at 20% 20%,
            rgba(0,255,100,0.10),
            transparent 30%
        ),
        #0d120d;

    border: 1px solid #1b351f;
}

.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
}

.hero p {
    color: #9ba99d;
    font-size: 16px;
}

.section-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 15px;
}

.result-card {
    padding: 22px;
    border-radius: 14px;
    margin-bottom: 15px;
}

.duplicate {
    background: rgba(0, 255, 130, 0.08);
    border: 1px solid #1d7850;
}

.nonduplicate {
    background: rgba(60, 150, 255, 0.08);
    border: 1px solid #28629a;
}

.error {
    background: rgba(255, 70, 70, 0.08);
    border: 1px solid #9a3030;
}

.result-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 7px;
}

.result-subtitle {
    color: #a7b2aa;
}

.metric-box {
    padding: 18px;
    border-radius: 12px;
    background: #101610;
    border: 1px solid #263226;
}

footer {
    display: none !important;
}

"""


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="AI Semantic Duplicate Detector",
    css=custom_css,
    theme=gr.themes.Base()
) as demo:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="hero">

            <div style="color:#22c55e;
                        font-size:14px;
                        margin-bottom:8px;">
                ● EMBEDDING SIMILARITY
            </div>

            <h1>
                Semantic Duplicate Detector
            </h1>

            <p>
                Encode two questions into vector space and
                measure how close their meaning sits using
                cosine similarity against a validated
                decision threshold.
            </p>

        </div>
        """
    )

    # --------------------------------------------------------
    # INPUT SECTION
    # --------------------------------------------------------

    gr.Markdown(
        "## 🔍 Compare two questions"
    )

    with gr.Row():

        with gr.Column():

            question_a = gr.Textbox(
                label="Question A",
                placeholder="Enter the first question...",
                lines=5,
                value="How do I reset my password?"
            )

        with gr.Column():

            question_b = gr.Textbox(
                label="Question B",
                placeholder="Enter the second question...",
                lines=5,
                value="I forgot my password. How can I reset it?"
            )

    # --------------------------------------------------------
    # BUTTONS
    # --------------------------------------------------------

    with gr.Row():

        analyze_button = gr.Button(
            "🔍 Analyze similarity",
            variant="primary"
        )

        clear_button = gr.Button(
            "🗑️ Clear"
        )

        download_button = gr.DownloadButton(
            "⬇️ Download report",
            visible=True
        )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    gr.Markdown(
        "## 📊 Result"
    )

    result_output = gr.HTML()

    with gr.Row():

        similarity_output = gr.Number(
            label="🧠 Similarity Score",
            precision=2
        )

        threshold_output = gr.Textbox(
            label="⚙️ Decision Threshold"
        )

    explanation_output = gr.Textbox(
        label="💡 Explanation",
        lines=3
    )

    # --------------------------------------------------------
    # SESSION SUMMARY
    # --------------------------------------------------------

    gr.Markdown(
        "## 📈 Session Summary"
    )

    session_plot = gr.Plot(
        label="Similarity across comparisons"
    )

    # --------------------------------------------------------
    # EXAMPLES
    # --------------------------------------------------------

    gr.Markdown(
        "## 🧪 Try example pairs"
    )

    gr.Examples(
        examples=[
            [
                "How do I reset my password?",
                "I forgot my password. How can I reset it?"
            ],
            [
                "How can I learn Python?",
                "What is the best way to start learning Python?"
            ],
            [
                "How do I contact someone on Quora?",
                "How can I message someone on Quora?"
            ],
            [
                "What is the capital of India?",
                "How do I make chocolate cake?"
            ],
            [
                "How can I improve my programming skills?",
                "What are the best ways to become better at coding?"
            ]
        ],
        inputs=[
            question_a,
            question_b
        ]
    )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    gr.Markdown(
        """
        <div style="text-align:center;
                    color:#6d776f;
                    margin-top:30px;
                    padding:20px;">

        Sentence Transformers · Cosine Similarity ·
        Semantic Duplicate Detection

        </div>
        """
    )

    # ========================================================
    # EVENTS
    # ========================================================

    analyze_button.click(
        fn=analyze_similarity,

        inputs=[
            question_a,
            question_b
        ],

        outputs=[
            result_output,
            similarity_output,
            threshold_output,
            explanation_output,
            session_plot,
            download_button
        ]
    )

    clear_button.click(
        fn=clear_all,

        inputs=[],

        outputs=[
            question_a,
            question_b,
            result_output,
            similarity_output,
            threshold_output,
            explanation_output,
            session_plot,
            download_button
        ]
    )


# ============================================================
# RENDER / PRODUCTION LAUNCH
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
