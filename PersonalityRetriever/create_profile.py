"""
create_profile.py

Reads an interview transcript and uses Gemini to generate
personality_summary.json for the Personality RAG system.

The JSON profile is a behavioral quick-reference — it captures the person's
identity, personality, general approach, and how they handle common situations.
It does not need to be exhaustive; the FAISS retrieval layer handles edge cases
and deep specifics. The schema is inferred from the transcript, so it works for
any role or person.

Usage:
    python create_profile.py
    python create_profile.py --transcript interview_transcript.md --output personality_summary.json
"""

import argparse
import json
import os


def load_transcript(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_extraction_prompt(transcript: str) -> str:
    return f"""You are building a behavioral profile JSON that will be used as a quick-reference for an AI agent impersonating this person.

Your goal: capture the essential personality, approach, and behavioral patterns of the subject — enough that someone reading the JSON can understand *how this person thinks, communicates, and handles situations* without reading the full transcript.

This JSON is NOT meant to be exhaustive. A separate retrieval system handles edge cases and deep specifics. Focus on what is broadly true and reusable across many interactions.

Return ONLY valid JSON — no markdown fences, no commentary.

What to include (adapt field names to fit the person's role):
- Basic identity: who they are, their role, background at a glance
- Personality & tone: how they come across, communication style, humor, energy
- Core values & priorities: what matters most to them
- General approach / philosophy: how they think about and tackle their work
- Common situations: how they typically handle difficult, common, or recurring scenarios relevant to their role
- Things they always do or never do: concrete behavioral rules or constraints
- Any signature phrases, frameworks, or go-to methods they mention

Keep values concrete and specific — prefer "uses Socratic questioning instead of giving direct answers" over "is a good listener."
Let the person's role and answers determine the JSON structure. Do not use a one-size-fits-all schema.
If something is not mentioned in the transcript, omit it rather than guessing.

TRANSCRIPT:
{transcript}
"""


def generate_profile(transcript: str, api_key: str) -> dict:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        google_api_key=api_key,
    )

    prompt = build_extraction_prompt(transcript)
    response = llm.invoke([HumanMessage(content=prompt)])
    raw = response.content.strip()

    # Strip markdown code fences if the model added them anyway
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(
        description="Generate personality_summary.json from an interview transcript."
    )
    parser.add_argument(
        "--transcript",
        default="interview_transcript.md",
        help="Path to the interview transcript (default: interview_transcript.md)",
    )
    parser.add_argument(
        "--output",
        default="personality_summary.json",
        help="Output path for the JSON profile (default: personality_summary.json)",
    )
    parser.add_argument(
        "--api-key-file",
        default="api_key.txt",
        help="File containing the Google API key (default: api_key.txt)",
    )
    args = parser.parse_args()

    # Resolve API key
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        if os.path.exists(args.api_key_file):
            with open(args.api_key_file, "r") as f:
                api_key = f.read().strip()
            os.environ["GOOGLE_API_KEY"] = api_key
        else:
            raise RuntimeError(
                "Google API key not found. Set GOOGLE_API_KEY or provide --api-key-file."
            )

    print(f"Loading transcript from: {args.transcript}")
    transcript = load_transcript(args.transcript)

    print("Sending transcript to Gemini for profile extraction...")
    profile = generate_profile(transcript, api_key)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"Profile saved to: {args.output}")


if __name__ == "__main__":
    main()
