import pytest
from sentiment_analyzer import analyze_sentiment

def test_positive_sentiment():
    text = "I love this research assistant, it's amazing!"
    result = analyze_sentiment(text)
    assert result.label == "positive"
    assert result.compound > 0.5
    assert "happy" in result.system_instruction or "satisfied" in result.system_instruction

def test_negative_sentiment():
    text = "I am very frustrated with the slow results."
    result = analyze_sentiment(text)
    assert result.label == "negative"
    assert result.compound < -0.4
    assert "empathetic" in result.system_instruction or "frustrated" in result.system_instruction

def test_neutral_sentiment():
    text = "The paper was published in 2023."
    result = analyze_sentiment(text)
    assert result.label == "neutral"
    assert -0.05 < result.compound < 0.05
    assert "professional" in result.system_instruction
