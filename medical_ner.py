"""
medical_ner.py
--------------
Task 3: Basic medical entity recognition.
Detects symptoms, diseases, treatments, medications, and body parts
from user queries using rule-based matching — no extra models needed.
"""

import re
from dataclasses import dataclass, field

# -------------------- ENTITY DICTIONARIES --------------------

SYMPTOMS = {
    "fever", "cough", "headache", "fatigue", "nausea", "vomiting",
    "diarrhea", "chest pain", "shortness of breath", "dizziness",
    "rash", "swelling", "pain", "bleeding", "numbness", "tingling",
    "weakness", "weight loss", "weight gain", "insomnia", "anxiety",
    "depression", "confusion", "seizure", "fainting", "palpitations",
    "sweating", "chills", "loss of appetite", "sore throat", "runny nose",
    "congestion", "back pain", "joint pain", "muscle ache", "itching",
    "bruising", "jaundice", "bloating", "constipation", "cramps",
    "night sweats", "hair loss", "high temperature", "low temperature",
    "abdominal pain", "chest tightness", "wheezing", "sneezing",
    "blurred vision", "memory loss", "mood swings", "irritability",
}

DISEASES = {
    "diabetes", "cancer", "hypertension", "asthma", "arthritis",
    "alzheimer", "parkinson", "stroke", "heart disease", "pneumonia",
    "tuberculosis", "hiv", "aids", "hepatitis", "anemia", "eczema",
    "psoriasis", "lupus", "fibromyalgia", "multiple sclerosis",
    "epilepsy", "migraine", "hypothyroidism", "hyperthyroidism",
    "crohn's disease", "irritable bowel syndrome", "ibs", "gerd",
    "chronic kidney disease", "ckd", "copd", "covid", "influenza",
    "malaria", "dengue", "typhoid", "cholera", "meningitis",
    "sepsis", "cellulitis", "sinusitis", "bronchitis", "appendicitis",
    "pancreatitis", "gastritis", "colitis", "dermatitis", "glaucoma",
    "cataracts", "scoliosis", "osteoporosis", "gout", "leukemia",
    "lymphoma", "melanoma", "breast cancer", "lung cancer",
    "prostate cancer", "colon cancer", "thyroid cancer",
    "type 1 diabetes", "type 2 diabetes", "gestational diabetes",
    "rheumatoid arthritis", "osteoarthritis", "heart failure",
}

TREATMENTS = {
    "surgery", "chemotherapy", "radiation", "physiotherapy",
    "dialysis", "transplant", "vaccination", "immunotherapy",
    "psychotherapy", "cognitive behavioral therapy", "cbt",
    "insulin therapy", "oxygen therapy", "blood transfusion",
    "stem cell therapy", "laser therapy", "acupuncture",
    "rehabilitation", "palliative care", "hormone therapy",
    "antibiotic therapy", "antiviral therapy", "biopsy",
    "endoscopy", "colonoscopy", "mri", "ct scan", "x-ray",
    "ultrasound", "ecg", "eeg", "blood test", "urine test",
}

MEDICATIONS = {
    "aspirin", "ibuprofen", "paracetamol", "acetaminophen",
    "metformin", "insulin", "lisinopril", "atorvastatin",
    "amoxicillin", "azithromycin", "prednisone", "metoprolol",
    "omeprazole", "levothyroxine", "amlodipine", "simvastatin",
    "ciprofloxacin", "doxycycline", "warfarin", "clopidogrel",
    "sertraline", "fluoxetine", "alprazolam", "lorazepam",
    "cetirizine", "loratadine", "montelukast", "albuterol",
    "salbutamol", "methotrexate", "hydroxychloroquine",
    "gabapentin", "pregabalin", "tramadol", "morphine",
    "codeine", "naproxen", "diclofenac", "pantoprazole",
}

BODY_PARTS = {
    "heart", "lung", "liver", "kidney", "brain", "stomach",
    "intestine", "colon", "pancreas", "thyroid", "spleen",
    "bladder", "uterus", "prostate", "breast", "skin", "bone",
    "muscle", "joint", "nerve", "eye", "ear", "nose", "throat",
    "spine", "hip", "knee", "shoulder", "elbow", "wrist", "ankle",
    "gallbladder", "appendix", "esophagus", "trachea", "artery",
    "vein", "lymph node", "adrenal gland", "pituitary gland",
}

# -------------------- DATA CLASS --------------------

@dataclass
class MedicalEntities:
    symptoms:    list = field(default_factory=list)
    diseases:    list = field(default_factory=list)
    treatments:  list = field(default_factory=list)
    medications: list = field(default_factory=list)
    body_parts:  list = field(default_factory=list)

    def has_entities(self) -> bool:
        return any([
            self.symptoms, self.diseases, self.treatments,
            self.medications, self.body_parts
        ])

    def to_dict(self) -> dict:
        return {
            "Symptoms":    self.symptoms,
            "Diseases":    self.diseases,
            "Treatments":  self.treatments,
            "Medications": self.medications,
            "Body Parts":  self.body_parts,
        }


# -------------------- EXTRACTOR --------------------

def extract_medical_entities(text: str) -> MedicalEntities:
    """
    Extract medical entities from plain text using rule-based matching.
    Case-insensitive. Supports both single-word and multi-word terms.
    """
    text_lower = text.lower()
    entities = MedicalEntities()

    def find_matches(vocab: set) -> list:
        found = []
        for term in vocab:
            if " " in term:
                if term in text_lower:
                    found.append(term)
            else:
                if re.search(rf"\b{re.escape(term)}\b", text_lower):
                    found.append(term)
        return sorted(set(found))

    entities.symptoms    = find_matches(SYMPTOMS)
    entities.diseases    = find_matches(DISEASES)
    entities.treatments  = find_matches(TREATMENTS)
    entities.medications = find_matches(MEDICATIONS)
    entities.body_parts  = find_matches(BODY_PARTS)

    return entities