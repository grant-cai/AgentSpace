# Guidelines for Creating Graph RAG Test Question Sets

This document outlines the principles and methodology for developing a high-quality test set to evaluate the performance of a Graph RAG (Retrieval-Augmented Generation) agent.

## 1. Core Principles

**A. Focus on Substantive Content:** The primary goal is to test comprehension and reasoning about the *content* within the provided documents. Questions **must not** be about metadata, such as file names, document titles, or other trivial details. The evaluation should focus on the actual information, concepts, and relationships described in the text.

## 2. Characteristics of an Ideal Question Set

An ideal test set for Graph RAG should be diverse, challenging, and specifically designed to test the unique capabilities of graph structures (traversing relationships and aggregating community information).

### A. Diversity in Complexity
1.  **Direct Retrieval (Level 1):** Questions about specific concepts, definitions, or core arguments.
    *   *Example:* "How does Grant define the concept of 'minimalist tutoring'?"
2.  **Relationship Discovery (Level 2):** Questions requiring the identification of a connection between two specific themes or ideas.
    *   *Example:* "How does Grant's tutoring philosophy at the Sweetland Center influence his approach to providing feedback on creative writing?"
3.  **Multi-hop Reasoning (Level 3):** Questions that require traversing multiple conceptual links (e.g., connecting a pedagogical theory to a specific practice and its intended outcome).
    *   *Example:* "How do Grant's views on linguistic justice inform his specific strategies for supporting ESL students in navigating standard academic English?"
4.  **Global/Community Aggregation (Level 4):** Questions that require summarizing high-level themes, shifts in perspective, or overarching philosophies across the entire body of content.
    *   *Example:* "What are the recurring themes in Grant's evolution as a tutor, specifically regarding the balance between student autonomy and instructor authority?"

### B. Diversity in Question Types
*   **Fact-based:** Specific data points.
*   **Thematic/Conceptual:** High-level ideas and abstractions.
*   **Comparative:** Comparing two entities based on their relational context.
*   **Counterfactual/Negative:** Asking about things NOT in the graph to test "I don't know" performance.
*   **Content-focused:** Questions should primarily engage with the substantive information, themes, and concepts presented in the text, avoiding queries about peripheral or trivial details.

## 3. How to Develop the Test Set from Information

To develop a test set from a raw document:

1.  **Identify Key Entities:** List the primary people, organizations, concepts, and events.
2.  **Map Relationships:** Identify how these entities interact.
3.  **Formulate Pathways:** Trace a path from one entity to another through a third entity to create a multi-hop question.
4.  **Identify Clusters:** Group related entities (e.g., "Education", "Tutoring Style") and create questions that require summarizing the entire group.

## 4. Creating the Test Set (JSON Format)

The output should be a JSON array of objects. Each object must follow this schema:

```json
[
  {
    "id": "Q1",
    "query": "The question text",
    "complexity": "local | multi-hop | global",
    "target_entities": ["Entity1", "Entity2"],
    "reasoning_path": "Entity1 -> Relationship -> Entity2",
    "reference": "A brief summary of the correct answer"
  }
]
```

## 5. Instructions for the Generator Notebook

1.  **Input:** Provide the raw text or the extracted triple information.
2.  **Processing:** The notebook will use an LLM (e.g., Gemini) to analyze the text and generate questions based on the levels described above.
3.  **Validation:** Ensure each generated question has a clear path in the information provided.
4.  **Export:** Save the results as `test_questions.json`.
