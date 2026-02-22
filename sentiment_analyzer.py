"""
sentiment_analyzer.py
----------------------
Task 5: Sentiment analysis for emotionally-aware chatbot responses.
Uses VADER (Valence Aware Dictionary Sentiment Reasoner) — offline, fast,
specifically tuned for conversational and social text.
"""

from dataclasses import dataclass
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


@dataclass
class SentimentResult:
    label:      str    # "positive" | "neutral" | "negative"
    compound:   float  # -1.0 to +1.0
    positive:   float  # 0.0 to 1.0
    neutral:    float
    negative:   float
    emoji:      str
    tone_label: str    # human-readable tone

    @property
    def intensity(self) -> str:
        """Returns intensity: mild / moderate / strong."""
        abs_c = abs(self.compound)
        if abs_c >= 0.7:
            return "strong"
        elif abs_c >= 0.35:
            return "moderate"
        else:
            return "mild"

    @property
    def system_instruction(self) -> str:
        """Returns a system-prompt snippet to inject into the LLM."""
        if self.label == "negative":
            if self.intensity == "strong":
                return (
                    "The user seems very frustrated or upset. "
                    "Be extra empathetic, acknowledge their feelings first, "
                    "apologise if appropriate, then help solve the problem calmly."
                )
            else:
                return (
                    "The user sounds a little unhappy or concerned. "
                    "Respond with empathy and reassurance before providing your answer."
                )
        elif self.label == "positive":
            if self.intensity == "strong":
                return (
                    "The user is very happy and enthusiastic! "
                    "Match their energy with a warm, upbeat tone while staying helpful."
                )
            else:
                return (
                    "The user seems satisfied or in a good mood. "
                    "Maintain a friendly, positive tone."
                )
        else:
            return "The user's tone is neutral. Be clear, professional, and helpful."


_LABEL_META = {
    "positive": ("positive", "😊", "Positive"),
    "negative": ("negative", "😟", "Negative"),
    "neutral":  ("neutral",  "😐", "Neutral"),
}


def analyze_sentiment(text: str) -> SentimentResult:
    """
    Analyse sentiment of a text string using VADER.
    Returns a SentimentResult with label, scores, emoji, and LLM instructions.
    """
    scores = _analyzer.polarity_scores(text)
    c      = scores["compound"]

    if c >= 0.05:
        key = "positive"
    elif c <= -0.05:
        key = "negative"
    else:
        key = "neutral"

    label, emoji, tone_label = _LABEL_META[key]
    return SentimentResult(
        label      = label,
        compound   = round(c, 4),
        positive   = round(scores["pos"], 4),
        neutral    = round(scores["neu"], 4),
        negative   = round(scores["neg"], 4),
        emoji      = emoji,
        tone_label = tone_label,
    )
