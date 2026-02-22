from PIL import Image
import io


def gemini_chat(history: list, image: Image.Image | None = None) -> str:
    """
    Sends full conversation history to Gemini.
    Supports optional image in the latest user message.
    Only called when an image is uploaded (uses GOOGLE_API_KEY).
    """
    from google import genai
    from google.genai import types

    # Lazy init — only runs when image chat is actually used
    client = genai.Client()

    contents = []

    # Convert chat history into Gemini format (Turn-based)
    for i, msg in enumerate(history):
        role = "user" if msg["role"] == "user" else "model"
        parts = [types.Part.from_text(text=msg["content"])]

        # If this is the last user message and an image is provided, convert and add it
        if i == len(history) - 1 and msg["role"] == "user" and image is not None:
            # Convert PIL Image → bytes (Gemini needs raw bytes, not PIL objects)
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            image_bytes = buf.getvalue()
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))

        contents.append(types.Content(role=role, parts=parts))

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents
    )

    return response.text
