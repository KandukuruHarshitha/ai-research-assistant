"""
language_handler.py
--------------------
Task 6: Multilingual support for the AI Research Assistant.

Features:
- Auto-detect language of user input (langdetect)
- Translate to/from English using deep-translator (free, no API key)
- Cultural context injection for appropriate responses
- Supports: English, Spanish, French, Hindi, German, Arabic, Telugu, Japanese
"""

from dataclasses import dataclass, field
from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator


# ──────────────────────────────────────────────────────────────
# LANGUAGE REGISTRY
# ──────────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "en": {
        "name": "English",
        "native_name": "English",
        "flag": "🇬🇧",
        "cultural_note": "Respond in clear, professional English.",
        "rtl": False,
    },
    "es": {
        "name": "Spanish",
        "native_name": "Español",
        "flag": "🇪🇸",
        "cultural_note": (
            "Respond in Spanish. Use a warm, friendly tone typical of Spanish-speaking cultures. "
            "Address the user respectfully."
        ),
        "rtl": False,
    },
    "fr": {
        "name": "French",
        "native_name": "Français",
        "flag": "🇫🇷",
        "cultural_note": (
            "Respond in French. Use a polite, formal tone. "
            "French users appreciate precision and elegance in communication."
        ),
        "rtl": False,
    },
    "hi": {
        "name": "Hindi",
        "native_name": "हिन्दी",
        "flag": "🇮🇳",
        "cultural_note": (
            "Respond in Hindi (Devanagari script). Be respectful and warm. "
            "Indian users appreciate polite and helpful communication."
        ),
        "rtl": False,
    },
    "de": {
        "name": "German",
        "native_name": "Deutsch",
        "flag": "🇩🇪",
        "cultural_note": (
            "Respond in German. Be precise, thorough, and formal. "
            "German users value accuracy and detailed explanations."
        ),
        "rtl": False,
    },
    "ar": {
        "name": "Arabic",
        "native_name": "العربية",
        "flag": "🇸🇦",
        "cultural_note": (
            "Respond in Arabic (Modern Standard Arabic). "
            "Be respectful and culturally sensitive. "
            "Use formal Arabic appropriate for professional communication."
        ),
        "rtl": True,
    },
    "te": {
        "name": "Telugu",
        "native_name": "తెలుగు",
        "flag": "🇮🇳",
        "cultural_note": (
            "Respond in Telugu script. Be warm, respectful, and helpful. "
            "Use natural Telugu as spoken in Andhra Pradesh and Telangana."
        ),
        "rtl": False,
    },
    "ja": {
        "name": "Japanese",
        "native_name": "日本語",
        "flag": "🇯🇵",
        "cultural_note": (
            "Respond in Japanese (Hiragana/Kanji). Use polite keigo (敬語) language. "
            "Japanese users appreciate respectful, structured, and detailed answers."
        ),
        "rtl": False,
    },
}

# langdetect code → our registry key mapping
_LANGDETECT_MAP = {
    "en": "en", "es": "es", "fr": "fr", "hi": "hi",
    "de": "de", "ar": "ar", "te": "te", "ja": "ja",
    "zh-cn": "en",  # fallback unsupported → English
    "pt": "es",     # Portuguese → nearest listed
    "it": "fr",     # Italian fallback
}

# ──────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ──────────────────────────────────────────────────────────────
@dataclass
class LanguageResult:
    code:          str          # e.g. "hi"
    name:          str          # e.g. "Hindi"
    native_name:   str          # e.g. "हिन्दी"
    flag:          str          # e.g. "🇮🇳"
    cultural_note: str
    is_english:    bool
    rtl:           bool
    confidence:    str = "detected"

    @property
    def display(self) -> str:
        return f"{self.flag} {self.native_name}"

    @property
    def system_instruction(self) -> str:
        return self.cultural_note


# ──────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────
def detect_language(text: str) -> LanguageResult:
    """
    Detect language of input text.
    Falls back to English if detection fails or language unsupported.
    """
    try:
        raw_code = detect(text)
    except LangDetectException:
        raw_code = "en"

    code = _LANGDETECT_MAP.get(raw_code, "en")
    meta = SUPPORTED_LANGUAGES.get(code, SUPPORTED_LANGUAGES["en"])

    return LanguageResult(
        code          = code,
        name          = meta["name"],
        native_name   = meta["native_name"],
        flag          = meta["flag"],
        cultural_note = meta["cultural_note"],
        is_english    = code == "en",
        rtl           = meta["rtl"],
    )


def translate_to_english(text: str, source_lang: str) -> str:
    """Translate text from source_lang to English."""
    if source_lang == "en":
        return text
    try:
        return GoogleTranslator(source=source_lang, target="en").translate(text)
    except Exception:
        return text  # return original on failure


def translate_from_english(text: str, target_lang: str) -> str:
    """Translate text from English to target_lang."""
    if target_lang == "en":
        return text
    try:
        # deep-translator has a 5000 char limit per request — chunk if needed
        if len(text) <= 4500:
            return GoogleTranslator(source="en", target=target_lang).translate(text)
        # For long texts, translate in chunks
        chunks = _chunk_text(text, 4000)
        translated = [
            GoogleTranslator(source="en", target=target_lang).translate(c)
            for c in chunks
        ]
        return "\n".join(translated)
    except Exception:
        return text  # return English on failure


def _chunk_text(text: str, size: int) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    sentences = text.replace(".\n", ". ").split(". ")
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) < size:
            current += s + ". "
        else:
            chunks.append(current.strip())
            current = s + ". "
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text]


def get_language_options() -> dict[str, str]:
    """Return {display_name: code} for UI selectbox."""
    return {
        f"{v['flag']} {v['native_name']} ({v['name']})": k
        for k, v in SUPPORTED_LANGUAGES.items()
    }
