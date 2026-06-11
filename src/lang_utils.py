from googletrans import Translator
from langdetect import LangDetectException, detect
import time

translator = Translator()

# Mapping of language codes to display names
LANGUAGE_MAP = {
    'en': 'English',
    'hi': 'Hindi',
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'zh-cn': 'Chinese',
    'mr': 'Marathi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'kn': 'Kannada',
}

def detect_user_language(text: str, selected_lang: str = 'en') -> str:
    """Detect the language of user input with fallback mechanisms"""
    fallback = selected_lang or 'en'
    
    try:
        lang = detect(text)
        print("Detected (langdetect):", lang)
        return lang
    except:
        print("langdetect failed")

    try:
        result = translator.detect(text)
        print("Detected (googletrans):", result.lang)
        return result.lang
    except Exception as e:
        print("googletrans detect failed:", e)
        return fallback


def translate_text(text: str, target_lang: str = "en", retries: int = 3) -> str:
    """
    Translate text with retry mechanism and error handling
    
    Args:
        text: Text to translate
        target_lang: Target language code (e.g., 'hi', 'es')
        retries: Number of retries if translation fails
        
    Returns:
        Translated text or original text if translation fails
    """
    if not text or not text.strip():
        return text
    
    if target_lang == 'en':
        return text
    
    for attempt in range(retries):
        try:
            result = translator.translate(text, dest=target_lang)
            if result and result.text:
                print(f"✓ Translation successful to {target_lang} (attempt {attempt + 1})")
                return result.text
        except Exception as e:
            print(f"❌ Translation attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(0.5)  # Wait before retry
    
    print(f"⚠️ Translation failed after {retries} attempts, returning original text")
    return text


def translate_with_fallback(text: str, target_lang: str = "en") -> str:
    """Translate text with automatic fallback to English if target is unavailable"""
    return translate_text(text, target_lang)

