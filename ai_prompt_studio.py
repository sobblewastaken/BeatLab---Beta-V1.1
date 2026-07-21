import json
import time
import random
import re
import numpy as np
import pandas as pd
import pygame
import scipy.io.wavfile as wavfile
import httpx
from ollama import Client

# Initialize audio engine
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.set_num_channels(64)

print("\n🎛️ AI DRUM STUDIO TERMINAL ACTIVE")

# --- DATASET SELECTION WORKFLOW ---
print("💿 Available Sound Profiles:")
print(" 1: Heavy Trap / Hip-Hop Kit")
print(" 2: Clean Pop / Dance Kit")
print(" 3: Acoustic Rock / Metal Kit")

try:
    kit_choice = input("👉 Choose your kit style (1, 2, or 3): ").strip()
    if kit_choice == "2":
        map_file, kit_name = "pop_kit_map.csv", "Pop Studio"
    elif kit_choice == "3":
        map_file, kit_name = "rock_kit_map.csv", "Acoustic Rock"
    else:
        map_file, kit_name = "advanced_kit_map.csv", "Premium Trap"
        
    kit_map = pd.read_csv(map_file)
except Exception as e:
    print(f"Profile error ({e}). Defaulting to original Trap Kit.")
    kit_map = pd.read_csv("advanced_kit_map.csv")
    kit_name = "Premium Trap"

def get_sample_by_cluster(cluster_id):
    pool = kit_map[kit_map["category_id"] == cluster_id]["file_path"].tolist()
    if pool:
        return pygame.mixer.Sound(random.choice(pool))
    return None

print(f"🔊 Loading your {kit_name} custom drum kits...")
sounds = {
    "kick":  get_sample_by_cluster(0),
    "808":   get_sample_by_cluster(1),
    "snare": get_sample_by_cluster(2),
    "clap":  get_sample_by_cluster(3),
    "hat":   get_sample_by_cluster(4),
    "perc":  get_sample_by_cluster(5),
    "crash": get_sample_by_cluster(6)
}

try:
    user_prompt = input("\n💬 Describe the beat/vibe you want to make:\n👉 ")
    print("")
    file_choice = input("💾 Export Format Selection:\n 1: WAV (Audio Master file)\n 2: MIDI (Note data for DAW export)\n👉 Choice (1 or 2): ").strip()
except KeyboardInterrupt:
    print("\nExiting studio.")
    exit()

system_instructions = "You are an expert music producer. Respond ONLY with a raw JSON object containing no markdown backticks. Format: {\"bpm\": 140, \"kick\": [0,1...], \"808\": [...], \"snare\": [...], \"clap\": [...], \"hat\": [...], \"perc\": [...], \"crash\": [...]} Each list must contain exactly 16 steps using 0 or 1."

ai_data = {}
try:
    print("\n📡 Handshaking with your local background Ollama engine...")
    custom_http_client = httpx.Client(timeout=3.0)
    ai_client = Client(host="http://127.0.0.1:11434", client=custom_http_client, trust_env=False)
    
    response = ai_client.chat(
        model="llama3", 
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt}
        ], 
        options={"temperature": 0.8, "num_predict": 400}
    )
    raw_content = response["message"]["content"].strip()
    
    json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    if json_match:
        ai_data = json.loads(json_match.group(0))
        print("✅ AI pattern built successfully!")
    else:
        raise ValueError("No pattern block found.")
        
except Exception as e:
    print(f"⏩ Handshake timed out or skipped ({e}). Activating internal Dynamic Rhythm Engine!")
    chosen_bpm = random.choice([85, 95, 110, 125, 140])
    ai_data = {"bpm": chosen_bpm}
    for key in sounds.keys():
        if key == "kick": ai_data[key] = [1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]
        elif key in ["snare", "clap"]: ai_data[key] = [0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]
        elif key == "hat": ai_data[key] = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
        elif key == "808" and chosen_bpm >= 130: ai_data[key] = [1,0,0,1,0,0,1,0,0,0,1,0,0,1,0,0]
        else: ai_data[key] = [1 if random.random() < 0.15 else 0 for _ in range(16)]

bpm = float(ai_data.get("bpm", 120))
order = ["kick", "808", "snare", "clap", "hat", "perc", "crash"]

if file_choice == "2":
    print(f"\n🎹 Coding a custom multitrack MIDI arrangement at {bpm} BPM...")
    output_filename = f"prompt_beat_{int(bpm)}bpm.mid"
    midi_bytes = bytearray()
    midi_bytes.extend(b'MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0')
    track_data = bytearray()
    tempo_micro = int(60000000 / bpm)
    track_data.extend(b'\x00\xff\x51\x03' + tempo_micro.to_bytes(3, 'big'))
    midi_notes = {"kick": 36, "808": 41, "snare": 38, "clap": 39, "hat": 42, "perc": 48, "crash": 49}
    ticks_per_step = 120
    for bar in range(4):
        for step in range(16):
            for key in order:
                if key in ai_data and isinstance(ai_data[key], list) and len(ai_data[key]) > 0:
                    safe_step = step % len(ai_data[key])
                    if ai_data[key][safe_step] == 1:
                        note = midi_notes[key]
                        track_data.extend(b'\x00\x99' + bytes([note, 100]))
                        track_data.extend(bytes([ticks_per_step]) + b'\x89' + bytes([note, 0]))
    track_data.extend(b'\x00\xff\x2f\x00')
    midi_bytes.extend(b'MTrk' + len(track_data).to_bytes(4, 'big') + track_data)
    with open(output_filename, "wb") as f: f.write(midi_bytes)
    print(f"🎉 SUCCESS! MIDI file created as: {output_filename}")
else:
    target_duration, sample_rate = 10.0, 44100
    step_time = 60.0 / bpm / 4.0
    step_samples = int(step_time * sample_rate)
    bar_samples = step_samples * 16
    total_output_samples = int(target_duration * sample_rate)
    output_buffer = np.zeros((total_output_samples, 2), dtype=np.float32)
    print(f"\n⚡ Mastering your 10-second master audio track at {bpm} BPM...")
    current_sample_index = 0
    while current_sample_index < total_output_samples:
        for step in range(16):
            step_start = current_sample_index + (step * step_samples)
            if step_start >= total_output_samples: break
            for key in order:
                if key in ai_data and isinstance(ai_data[key], list) and len(ai_data[key]) > 0:
                    safe_step = step % len(ai_data[key])
                    if ai_data[key][safe_step] == 1 and sounds[key]:
                        raw_array = pygame.sndarray.array(sounds[key]).astype(np.float32)
                        if np.max(np.abs(raw_array)) > 1.0: raw_array /= 32768.0
                        sound_array = np.column_stack((raw_array, raw_array)) if raw_array.ndim == 1 else raw_array
                        sound_len = len(sound_array)
                        end_pos = min(step_start + sound_len, total_output_samples)
                        available_space = end_pos - step_start
                        if available_space > 0: output_buffer[step_start:end_pos] += sound_array[:available_space]
        current_sample_index += bar_samples
    max_amplitude = np.max(np.abs(output_buffer))
    if max_amplitude > 0: output_buffer /= max_amplitude
    output_filename = f"prompt_beat_{int(bpm)}bpm.wav"
    wavfile.write(output_filename, sample_rate, (output_buffer * 32767).astype(np.int16))
    print(f"🎉 SUCCESS! Audio file created as: {output_filename}")
    try:
        pygame.mixer.Sound(output_filename).play()
        time.sleep(target_duration)
    except KeyboardInterrupt: print("Playback stopped")
print("🏁 Session complete.")
