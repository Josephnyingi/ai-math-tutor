"""
HuggingFace Spaces entry point.
Spaces expects app.py with a Gradio `demo` block that auto-launches.
"""
from demo import build_ui

demo, theme = build_ui()

if __name__ == "__main__":
    demo.launch(theme=theme)
