import numpy as np

# Training Data Matrix: [Track, 16 Steps]
# Tracks Order: 0: Kick, 1: 808, 2: Snare, 3: Clap, 4: Hat, 5: Perc, 6: Crash
trap_pattern_1 = [
    [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0], # 0: Kick (On the 1 and 9)
    [1,0,0,0, 0,0,1,0, 0,0,0,0, 0,1,0,0], # 1: 808 (Sliding sub bass lines)
    [0,0,0,0, 1,0,0,0, 0,0,0,0, 1,0,0,0], # 2: Snare (Standard trap placement)
    [0,0,0,0, 0,0,0,0, 0,0,0,1, 1,0,0,0], # 3: Clap (Layered accent)
    [1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1], # 4: Hat (Rolling 16th notes)
    [0,0,1,0, 0,0,0,0, 0,1,0,0, 0,0,1,0], # 5: Perc (Syncopated filler)
    [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]  # 6: Crash (Downbeat accent)
]

def train_7track_brain(patterns):
    # 7 tracks, 16 steps, 2 binary states (0=silent, 1=play)
    matrix = np.zeros((7, 16, 2))
    
    for pattern in patterns:
        for track_idx in range(7):
            for step in range(16):
                state = pattern[track_idx][step]
                matrix[track_idx][step][state] += 1
                
    for t in range(7):
        for s in range(16):
            total = np.sum(matrix[t][s])
            if total > 0:
                matrix[t][s] /= total
            else:
                matrix[t][s] = [0.8, 0.2] # Fallback to mostly silent if untrained
    return matrix

# Run training
trained_brain = train_7track_brain([trap_pattern_1])
np.save("advanced_drum_brain.npy", trained_brain)
print("✅ 7-Track AI Rhythmic Brain saved to 'advanced_drum_brain.npy'!")
