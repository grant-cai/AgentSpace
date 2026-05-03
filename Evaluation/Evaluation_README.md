# Evaluation Module for Graph RAG

This directory contains tools and documentation for evaluating the performance of Graph RAG agents.

## Files

1.  **`TEST_SET_GUIDELINES.md`**: Detailed instructions on what makes a high-quality test set for Graph RAG. It covers question complexity (Local, Multi-hop, Global) and JSON formatting standards.
2.  **`Question_Generator.ipynb`**: A Jupyter Notebook that automates the generation of these test questions.
    *   It reads the guidelines from `TEST_SET_GUIDELINES.md`.
    *   It uses the Gemini API to analyze raw text provided by the user.
    *   It generates a JSON file (`test_questions.json`) containing diverse questions tailored for Graph RAG evaluation.

## How to use

1.  Open `Question_Generator.ipynb` in your Jupyter environment.
2.  Follow the steps in the notebook:
    *   Set up your Google API Key.
    *   Provide the source information (text) you want to test against.
    *   Run the generation cell to produce the `test_questions.json` file.
3.  Use the generated JSON file to benchmark your Graph RAG agent's retrieval and reasoning capabilities.
