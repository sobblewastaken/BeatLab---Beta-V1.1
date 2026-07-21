import time
import random
import numpy as np
import pandas as pd
import pygame
import scipy.io.wavfile as wavfile

# Initialize audio engine
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.set_num_channels(64) # Increased channels for dense overlap

# Load map & model
kit_map = pd.read_csv("advanced_kit_map.csv")
brain = np.load("advanced_drum_brain.npy")

# Helper to fetch a randomized sound matching the cluster ID
def get_sample_by_cluster(cluster_id):
    pool = kit_map[kit_map['category_id'] == cluster_id]['file_path'].tolist()
    if pool:
        return pygame.mixer.Sound(random.choice(pool))
    return None

print("🔊 Loading your premium custom drum kits...")
sounds = {
    "kick":  get_sample_by_cluster(0),
    "808":   get_sample_by_cluster(1),
    "snare": get_sample_by_cluster(2),
    "clap":  get_sample_by_cluster(3),
    "hat":   get_sample_by_cluster(4),
    "perc":  get_sample_by_cluster(5),
    "crash": get_sample_by_cluster(6)
}

# Fixed generation engine that unwraps multi-element matrix profiles safely
def generate_groove(model):
    grid = {k: [] for k in sounds.keys()}
    order = ["kick", "808", "snare", "clap", "hat", "perc", "crash"]
    
    for idx, key in enumerate(order):
        for step in range(16):
            prob_data = model[idx][step]
            
            # If it's an array, extract the first element safely
            if hasattr(prob_data, "__len__") or isinstance(prob_data, np.ndarray):
                prob = prob_data[0]
            else:
                prob = prob_data
                
            grid[key].append(1 if random.random() < prob else 0)
    return grid


# --- INTERACTIVE USER TERMINAL CONFIGURATION ---
print("\n🎛️ --- AI DRUM STUDIO TERMINAL ---")
try:
    bpm = float(input("👉 Enter your desired BPM (e.g. 130, 140, 160): "))
except ValueError:
    bpm = 140.0
    print("Invalid input. Defaulting to 140 BPM.")

target_duration = 10.0 # Length of file in seconds
sample_rate = 44100

# Calculate explicit rhythmic timings
step_time = 60.0 / bpm / 4.0  # Duration of one 16th note step in seconds
step_samples = int(step_time * sample_rate)
bar_samples = step_samples * 16 # Total audio samples in a full 16-step bar

# Determine total samples needed for a 10-second file
total_output_samples = int(target_duration * sample_rate)
output_buffer = np.zeros((total_output_samples, 2), dtype=np.float32)

print(f"\n⚡ Rendering a custom {target_duration} second audio track at {bpm} BPM...")

# Loop through time, generating fresh sequential AI patterns bar by bar
current_sample_index = 0
order = ["kick", "808", "snare", "clap", "hat", "perc", "crash"]

while current_sample_index < total_output_samples:
    # Re-roll the AI's dice for a fresh pattern variation every 16 steps
    active_loop = generate_groove(brain)
    
    for step in range(16):
        step_start = current_sample_index + (step * step_samples)
        
        # Break out immediately if we hit our strict 10-second limit mid-loop
        if step_start >= total_output_samples:
            break
            
        for key in order:
            if active_loop[key][step] == 1 and sounds[key]:
                raw_array = pygame.sndarray.array(sounds[key]).astype(np.float32)
                
                # Normalize values if loaded as integer array ranges
                if np.max(np.abs(raw_array)) > 1.0:
                    raw_array /= 32768.0
                
                # Force Mono files into dual-channel Stereo maps dynamically
                sound_array = np.column_stack((raw_array, raw_array)) if raw_array.ndim == 1 else raw_array
                    
                sound_len = len(sound_array)
                end_pos = min(step_start + sound_len, total_output_samples)
                available_space = end_pos - step_start
                
                if available_space > 0:
                    output_buffer[step_start:end_pos] += sound_array[:available_space]
                    
    # Move the pointer forward by one complete rhythmic bar sequence
    current_sample_index += bar_samples

# Apply standard production mastering limiters to stop digital clipping/crackle
max_amplitude = np.max(np.abs(output_buffer))
if max_amplitude > 0:
    output_buffer = output_buffer / max_amplitude

# Export master layout map back to industry standard 16-bit sound formatting
final_mix = (output_buffer * 32767).astype(np.int16)

output_filename = f"ai_trap_loop_{int(bpm)}bpm.wav"
wavfile.write(output_filename, sample_rate, final_mix)
print(f"🎉 SUCCESS! File created and saved to your workspace as: {output_filename}")

# Playback verification sequence
print(f"🔊 Playing back your fresh {target_duration}s master track...")
try:
    # Read the file back out loud directly to verify rendering authenticity
    verify_sound = pygame.mixer.Sound(output_filename)
    verify_sound.play()
    time.sleep(target_duration)
except KeyboardInterrupt:
    print("\nPlayback interrupted.")
print("🏁 Studio session complete.")
