# VERONICA — Improved Voice-Controlled Desktop Assistant (macOS)

## Features
- Speech-to-text with `SpeechRecognition` (Google recognition requires internet)
- Spoken responses with `pyttsx3`
- NLP intent classification using TF-IDF + Logistic Regression
- Allowlisted app, folder, and website opening
- Google/YouTube search, screenshot, scroll, system info
- SQLite memory and command log
- Short reminders
- Confirmation before lock/sleep/restart/shutdown
- Optional local general-question answering with Ollama

## 1. Requirements
- macOS
- Python 3.10–3.13 recommended
- Internet for Google speech recognition and web searches
- Microphone access
- For screenshots, allow Screen & System Audio Recording / Screen Recording permission to Terminal or your Python IDE in macOS Privacy & Security settings.

## 2. Install on your Mac
Open Terminal and run:

```bash
cd ~/Downloads
# If you extracted the ZIP elsewhere, cd into that extracted folder.
cd VERONICA_Improved
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install PortAudio first if PyAudio fails:
```bash
brew install portaudio
```

Then install Python dependencies:
```bash
pip install -r requirements.txt
```

If `pip install PyAudio` fails on macOS, try:
```bash
brew install portaudio
pip install --no-cache-dir PyAudio
```

## 3. Run
```bash
python main.py
```

Say `help`, `open Safari`, `search for Python tutorials`, `what time is it`,
`remember that my project is VERONICA`, `what do you remember`, or
`remind me in 1 minute to take a break`.

Exit by saying `exit` or pressing `Control+C`.

## 4. Optional local AI answers with Ollama
The assistant's intent classifier and desktop commands work without Ollama.
To answer open-ended questions locally, install Ollama from https://ollama.com,
then in a separate Terminal window run:

```bash
ollama pull qwen3:8b
ollama serve
```

If Ollama is already running, just pull the model. Then start VERONICA again.
To use another installed model:
```bash
export VERONICA_OLLAMA_MODEL="your-model-name"
python main.py
```

## 5. macOS permissions
- Microphone: System Settings → Privacy & Security → Microphone
- Automation: System Settings → Privacy & Security → Automation (if prompted)
- Screenshots: System Settings → Privacy & Security → Screen & System Audio Recording / Screen Recording
Grant permissions to the app you use to run Python (Terminal, VS Code, etc.).

## Important safety notes
- Power actions ask for a spoken confirmation. Test with `no` first.
- Only apps, folders, and websites in the allowlists can be opened.
- Do not run this assistant with administrator privileges.
- Voice recognition can mishear words; do not use it around sensitive unsaved work.
- This is an educational engineering project, not a fully autonomous agent.
# VERONICA-AI-
