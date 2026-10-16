"""
VERONICA — AI-Based Voice-Controlled Desktop Assistant
Improved macOS edition:
- SpeechRecognition + pyttsx3
- TF-IDF + Logistic Regression intent classification
- Desktop automation with an allowlist
- SQLite memory
- Reminders, system information, web searches
- Optional local AI answers through Ollama
- Confirmation before sensitive power actions
"""

import datetime as dt
import json
import os
import platform
import re
import sqlite3
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path

import pyttsx3
import pyautogui
import speech_recognition as sr
import psutil
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ASSISTANT_NAME = "VERONICA"
LANGUAGE = "en-IN"
DB_PATH = Path.home() / "veronica_memory.db"
SCREENSHOT_DIR = Path.home() / "Desktop" / "VERONICA_Screenshots"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("VERONICA_OLLAMA_MODEL", "qwen3:8b")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Voice output ----------
engine = None
try:
    engine = pyttsx3.init()
    engine.setProperty("rate", 175)
    engine.setProperty("volume", 1.0)
    voices = engine.getProperty("voices")
    preferred = ("samantha", "ava", "karen", "moira", "victoria", "female")
    for keyword in preferred:
        match = next((v for v in voices if keyword in v.name.lower()), None)
        if match:
            engine.setProperty("voice", match.id)
            print(f"Voice selected: {match.name}")
            break
except Exception as exc:
    print("Text-to-speech initialization failed:", exc)

def speak(message: str) -> None:
    """Print and speak a response; console output still works if TTS fails."""
    if not message:
        return
    print(f"\n{ASSISTANT_NAME}: {message}")
    if engine:
        try:
            engine.say(message)
            engine.runAndWait()
        except Exception as exc:
            print("TTS error:", exc)

# ---------- Speech input ----------
recognizer = sr.Recognizer()
recognizer.pause_threshold = 0.75
recognizer.dynamic_energy_threshold = True

def listen() -> str:
    try:
        with sr.Microphone() as source:
            print("\n🎤 Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.45)
            audio = recognizer.listen(source, timeout=7, phrase_time_limit=12)
        try:
            text = recognizer.recognize_google(audio, language=LANGUAGE)
            text = text.lower().strip()
            print("YOU:", text)
            return text
        except sr.UnknownValueError:
            print("I couldn't understand that.")
        except sr.RequestError as exc:
            speak("Speech recognition service is unavailable. Check your internet connection.")
            print("Recognition service error:", exc)
    except sr.WaitTimeoutError:
        pass
    except Exception as exc:
        print("Microphone error:", exc)
        print("Check microphone access and that PyAudio is installed.")
    return ""

# ---------- SQLite memory ----------
def db_connection():
    return sqlite3.connect(DB_PATH)

def initialize_database():
    with db_connection() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS command_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            command TEXT NOT NULL,
            intent TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")

def remember(memory: str):
    with db_connection() as conn:
        conn.execute(
            "INSERT INTO memories(memory, created_at) VALUES (?, ?)",
            (memory, dt.datetime.now().isoformat(timespec="seconds"))
        )

def get_memories(limit=8):
    with db_connection() as conn:
        rows = conn.execute(
            "SELECT memory FROM memories ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [row[0] for row in rows]

def log_command(command: str, intent: str):
    with db_connection() as conn:
        conn.execute(
            "INSERT INTO command_log(command, intent, created_at) VALUES (?, ?, ?)",
            (command, intent, dt.datetime.now().isoformat(timespec="seconds"))
        )

# ---------- Intent classification ----------
# A small, transparent training set is included so the project runs without an API key.
TRAINING_EXAMPLES = {
    "greeting": [
        "hello", "hi veronica", "hey there", "good morning", "hello assistant",
        "how are you", "are you there"
    ],
    "time": [
        "what time is it", "tell me the current time", "time please", "current time"
    ],
    "date": [
        "what is today's date", "tell me the date", "what day is it", "today's date"
    ],
    "open_app": [
        "open chrome", "launch safari", "start terminal", "open calculator",
        "please open vscode", "launch spotify", "open finder", "open whatsapp",
        "open system settings", "open notes", "open calendar", "start google chrome"
    ],
    "open_folder": [
        "open downloads folder", "show desktop folder", "open documents",
        "open my pictures", "open music folder", "show downloads"
    ],
    "open_website": [
        "open youtube", "go to google", "open chatgpt", "open instagram",
        "open gmail", "launch youtube website"
    ],
    "google_search": [
        "search for python tutorials", "google latest technology news",
        "search the web for machine learning", "look up best coding websites",
        "search google for weather in jaipur"
    ],
    "youtube_search": [
        "search youtube for python", "find music on youtube",
        "youtube search java tutorials", "search videos about artificial intelligence"
    ],
    "screenshot": [
        "take a screenshot", "capture my screen", "save screenshot", "screenshot please"
    ],
    "scroll_up": ["scroll up", "move page upward", "go up on this page"],
    "scroll_down": ["scroll down", "move page down", "go down on this page"],
    "system_info": [
        "system information", "tell me about my computer", "cpu and memory usage",
        "check battery percentage", "show system info"
    ],
    "remember": [
        "remember that my project is called veronica", "remember my meeting is at noon",
        "please remember that i like python"
    ],
    "recall": [
        "what do you remember", "recall my memories", "tell me what i asked you to remember"
    ],
    "reminder": [
        "remind me in 5 minutes to take a break", "set a reminder in 10 seconds",
        "remind me in 2 hours to study"
    ],
    "power_action": [
        "shut down my mac", "restart my computer", "put my mac to sleep",
        "lock my screen", "turn off the laptop"
    ],
    "question": [
        "explain artificial intelligence", "what is python", "tell me about machine learning",
        "how does the internet work", "give me an idea for a college project",
        "who are you", "what can you do"
    ],
    "exit": ["exit", "quit veronica", "stop assistant", "goodbye", "bye veronica"]
}

def build_intent_model():
    texts, labels = [], []
    for label, examples in TRAINING_EXAMPLES.items():
        texts.extend(examples)
        labels.extend([label] * len(examples))
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, strip_accents="unicode")),
        ("classifier", LogisticRegression(max_iter=1200, class_weight="balanced"))
    ])
    model.fit(texts, labels)
    return model

intent_model = build_intent_model()

def classify_intent(command: str):
    """Return intent and confidence; explicit patterns below improve slot extraction."""
    probabilities = intent_model.predict_proba([command])[0]
    best_index = probabilities.argmax()
    return intent_model.classes_[best_index], float(probabilities[best_index])

# ---------- Allowlisted macOS automation ----------
APP_NAMES = {
    "chrome": "Google Chrome", "google chrome": "Google Chrome",
    "safari": "Safari", "terminal": "Terminal", "finder": "Finder",
    "calculator": "Calculator", "whatsapp": "WhatsApp", "spotify": "Spotify",
    "discord": "Discord", "vscode": "Visual Studio Code", "vs code": "Visual Studio Code",
    "visual studio code": "Visual Studio Code", "messages": "Messages", "mail": "Mail",
    "notes": "Notes", "calendar": "Calendar", "photos": "Photos", "preview": "Preview",
    "facetime": "FaceTime", "settings": "System Settings", "system settings": "System Settings"
}
FOLDERS = {
    "downloads": Path.home() / "Downloads",
    "download": Path.home() / "Downloads",
    "desktop": Path.home() / "Desktop",
    "documents": Path.home() / "Documents",
    "document": Path.home() / "Documents",
    "pictures": Path.home() / "Pictures",
    "photos": Path.home() / "Pictures",
    "music": Path.home() / "Music",
    "movies": Path.home() / "Movies"
}
WEBSITES = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "chatgpt": "https://chatgpt.com",
    "chat gpt": "https://chatgpt.com",
    "instagram": "https://www.instagram.com",
    "gmail": "https://mail.google.com",
    "github": "https://github.com",
    "linkedin": "https://www.linkedin.com"
}

def open_app(name: str) -> bool:
    key = name.lower().strip()
    app = APP_NAMES.get(key)
    if not app:
        return False
    result = subprocess.run(["open", "-a", app], capture_output=True, text=True)
    if result.returncode == 0:
        speak(f"Opening {app}.")
        return True
    speak(f"I couldn't open {app}. Make sure it is installed.")
    print(result.stderr.strip())
    return False

def open_folder(name: str) -> bool:
    path = FOLDERS.get(name.lower().strip())
    if not path or not path.exists():
        return False
    subprocess.run(["open", str(path)], check=False)
    speak(f"Opening {name}.")
    return True

def open_website(name: str) -> bool:
    url = WEBSITES.get(name.lower().strip())
    if not url:
        return False
    webbrowser.open(url)
    speak(f"Opening {name}.")
    return True

def google_search(query: str, youtube=False):
    query = query.strip()
    if not query:
        speak("Tell me what you want me to search for.")
        return
    encoded = urllib.parse.quote_plus(query)
    if youtube:
        webbrowser.open(f"https://www.youtube.com/results?search_query={encoded}")
        speak(f"Searching YouTube for {query}.")
    else:
        webbrowser.open(f"https://www.google.com/search?q={encoded}")
        speak(f"Searching Google for {query}.")

def take_screenshot():
    try:
        filename = "VERONICA_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
        path = SCREENSHOT_DIR / filename
        pyautogui.screenshot().save(path)
        speak("Screenshot saved to the VERONICA Screenshots folder on your desktop.")
        print("Saved:", path)
    except Exception as exc:
        speak("I couldn't take a screenshot. Check Screen Recording permissions in macOS settings.")
        print("Screenshot error:", exc)

def system_info():
    mac_version = platform.mac_ver()[0] or "unknown"
    cpu = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    battery = psutil.sensors_battery()
    answer = f"macOS version {mac_version}. CPU usage is {cpu} percent. Memory usage is {memory.percent} percent."
    if battery:
        answer += f" Battery is at {battery.percent} percent."
    speak(answer)

# ---------- Reminders ----------
def reminder_worker(seconds: int, message: str):
    time.sleep(seconds)
    speak(f"Reminder: {message}")

def create_reminder(command: str) -> bool:
    match = re.search(r"\b(\d+)\s*(seconds?|minutes?|hours?)\b", command)
    if not match:
        speak("Try saying: remind me in 5 minutes to take a break.")
        return False
    amount = int(match.group(1))
    unit = match.group(2)
    multiplier = 1 if unit.startswith("second") else 60 if unit.startswith("minute") else 3600
    seconds = amount * multiplier
    message = command[match.end():].strip()
    message = re.sub(r"^(to|that)\s+", "", message)
    message = re.sub(r"^to\s+", "", message)
    if not message:
        message = "check your reminder"
    threading.Thread(target=reminder_worker, args=(seconds, message), daemon=True).start()
    speak(f"Reminder set for {amount} {unit}.")
    return True

# ---------- Optional local AI (Ollama) ----------
def ask_local_ai(question: str) -> str | None:
    """Use Ollama if it is running locally; no cloud API key is required."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": (
            "You are VERONICA, a helpful desktop voice assistant. "
            "Answer clearly in a concise, spoken-friendly way. "
            "Do not claim to have performed computer actions. User question: " + question
        ),
        "stream": False
    }).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
        return data.get("response", "").strip() or None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None

# ---------- Safe power actions ----------
def handle_power_action(command: str):
    action = None
    if "lock" in command:
        action = "lock"
    elif "sleep" in command:
        action = "sleep"
    elif "restart" in command:
        action = "restart"
    elif "shutdown" in command or "shut down" in command or "turn off" in command:
        action = "shutdown"

    if not action:
        return

    descriptions = {
        "lock": "lock your screen",
        "sleep": "put your Mac to sleep",
        "restart": "restart your Mac",
        "shutdown": "shut down your Mac"
    }
    speak(f"Do you want me to {descriptions[action]}? Say yes to confirm, or no to cancel.")
    confirmation = listen()
    if confirmation not in {"yes", "yes please", "confirm", "do it", "yeah"}:
        speak("Cancelled. No system action was taken.")
        return

    try:
        if action == "lock":
            # This triggers the macOS lock-screen shortcut via System Events.
            subprocess.run([
                "osascript", "-e",
                'tell application "System Events" to keystroke "q" using {control down, command down}'
            ], check=True)
        elif action == "sleep":
            subprocess.run(["pmset", "sleepnow"], check=True)
        elif action == "restart":
            subprocess.run(["osascript", "-e", 'tell application "System Events" to restart'], check=True)
        elif action == "shutdown":
            subprocess.run(["osascript", "-e", 'tell application "System Events" to shut down'], check=True)
        speak(f"Requested {action} action.")
    except Exception as exc:
        speak(f"I couldn't {descriptions[action]}. You may need to allow Automation permissions in macOS.")
        print("Power action error:", exc)

# ---------- Command routing ----------
def extract_after(command: str, prefixes):
    for prefix in prefixes:
        if command.startswith(prefix):
            return command[len(prefix):].strip(" .")
    return ""

def process_command(command: str):
    command = command.lower().strip()
    if not command:
        return
    intent, confidence = classify_intent(command)
    print(f"Intent: {intent} | confidence: {confidence:.2f}")
    log_command(command, intent)

    # Explicit patterns are used for safe routing and to extract command details.
    if command in {"exit", "quit", "stop assistant", "goodbye", "bye", "bye veronica", "quit veronica"}:
        speak("Goodbye. VERONICA signing off.")
        return "EXIT"

    if command in {"hello", "hi", "hey", "hello veronica", "hi veronica", "hey there"} or intent == "greeting" and confidence >= 0.30:
        speak("Hello! I'm VERONICA. How can I help you?")
        return

    if "what time" in command or "current time" in command or command == "time":
        speak("The current time is " + dt.datetime.now().strftime("%I:%M %p") + ".")
        return

    if "date" in command or "what day is it" in command or "today" in command and ("date" in command or "day" in command):
        speak("Today is " + dt.datetime.now().strftime("%A, %d %B %Y") + ".")
        return

    if command.startswith("remember "):
        memory = command[len("remember "):].strip()
        if memory:
            remember(memory)
            speak("Okay, I've saved that to memory.")
        return

    if "what do you remember" in command or "recall my memories" in command or "tell me what i asked you to remember" in command:
        memories = get_memories()
        if not memories:
            speak("I don't have any saved memories yet.")
        else:
            speak("Here are your most recent memories.")
            for memory in memories:
                speak(memory)
        return

    if "remind me" in command or intent == "reminder" and confidence >= 0.28:
        create_reminder(command)
        return

    if any(word in command for word in ("shut down", "shutdown", "restart", "put my mac to sleep", "sleep my mac", "lock my screen", "lock my mac", "turn off my mac", "turn off the laptop")):
        handle_power_action(command)
        return

    if "screenshot" in command or intent == "screenshot" and confidence >= 0.25:
        take_screenshot()
        return

    if "scroll up" in command:
        pyautogui.scroll(7)
        speak("Scrolling up.")
        return
    if "scroll down" in command:
        pyautogui.scroll(-7)
        speak("Scrolling down.")
        return

    if any(term in command for term in ("system info", "system information", "computer information", "battery percentage", "cpu usage", "memory usage")):
        system_info()
        return

    if command.startswith("search youtube"):
        google_search(extract_after(command, ["search youtube for", "search youtube"]), youtube=True)
        return
    if command.startswith("youtube search"):
        google_search(extract_after(command, ["youtube search for", "youtube search"]), youtube=True)
        return
    if command.startswith("search ") or command.startswith("google " ) or command.startswith("look up "):
        query = extract_after(command, ["search google for", "search the web for", "search for", "search", "google", "look up"])
        google_search(query)
        return

    # "Open <target>" supports only known apps, folders, and website names.
    if command.startswith(("open ", "launch ", "start ")):
        target = extract_after(command, ["open ", "launch ", "start "])
        target = re.sub(r"\s+(website|application|app|folder)$", "", target).strip()
        if open_app(target) or open_folder(target) or open_website(target):
            return
        speak(f"I don't have {target} in my allowlist yet. Add it to the app, folder, or website list in main.py.")
        return

    if command in {"what can you do", "help", "show commands"}:
        speak("I can open approved apps and websites, search Google or YouTube, tell time and date, take screenshots, scroll, report system information, save memories, set reminders, and answer questions if Ollama is running.")
        return

    # Try the local model for general questions if available.
    if intent == "question" or confidence < 0.38:
        speak("Let me think.")
        answer = ask_local_ai(command)
        if answer:
            speak(answer)
        else:
            speak("I couldn't identify a built-in command, and local AI is unavailable. Install and start Ollama to enable general question answering, or say help for commands.")
        return

    speak("I'm not sure what you meant. Try saying help to hear what I can do.")

def main():
    initialize_database()
    print("=" * 60)
    print("VERONICA — AI-BASED VOICE-CONTROLLED DESKTOP ASSISTANT")
    print("macOS | Speech recognition | NLP intent classifier | SQLite")
    print("=" * 60)
    speak("VERONICA is online. Say help to hear what I can do.")
    while True:
        try:
            command = listen()
            if command:
                result = process_command(command)
                if result == "EXIT":
                    break
        except KeyboardInterrupt:
            print("\nStopped by keyboard.")
            speak("VERONICA signing off.")
            break
        except Exception as exc:
            print("Unexpected error:", exc)
            time.sleep(0.5)

if __name__ == "__main__":
    main()
