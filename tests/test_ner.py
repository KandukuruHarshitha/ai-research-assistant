import pytest
from medical_ner import extract_medical_entities

def test_extract_symptoms():
    text = "I have a fever and a bad cough."
    entities = extract_medical_entities(text)
    assert "fever" in entities.symptoms
    assert "cough" in entities.symptoms

def test_extract_diseases():
    text = "Patient has a history of diabetes and asthma."
    entities = extract_medical_entities(text)
    assert "diabetes" in entities.diseases
    assert "asthma" in entities.diseases

def test_extract_medications():
    text = "Taking aspirin and metformin for treatment."
    entities = extract_medical_entities(text)
    assert "aspirin" in entities.medications
    assert "metformin" in entities.medications

def test_extract_body_parts():
    text = "Pain in the knee and shoulder."
    entities = extract_medical_entities(text)
    assert "knee" in entities.body_parts
    assert "shoulder" in entities.body_parts

def test_no_entities():
    text = "The weather is nice today."
    entities = extract_medical_entities(text)
    assert not entities.has_entities()
