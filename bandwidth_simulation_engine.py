import os
import h5py
import numpy as np

class MarkovBandwidthProcess:
    """
    Simulates fluctuating network bandwidth B(t) across consecutive time windows
    using a 3-State Discrete-Time Markov Chain (DTMC) + Continuous Channel Capacity.
    
    States:
      0: High Bandwidth   (e.g., 4G / 5G / Wi-Fi) -> 120 - 400 kbps
      1: Medium Bandwidth (e.g., 3G / Congested)  -> 35 - 100 kbps
      2: Low Bandwidth    (e.g., 2G / Heavy Fade) -> 5 - 30 kbps
    """
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        # Transition matrix with high diagonal inertia (network conditions are bursty/persistent)
        self.P = np.array([
            [0.80, 0.15, 0.05],  # High -> High, Med, Low
            [0.20, 0.65, 0.15],  # Med  -> High, Med, Low
            [0.10, 0.25, 0.65],  # Low  -> High, Med, Low
        ])
        self.bandwidth_ranges = {
            0: (120.0, 400.0),
            1: (35.0,  100.0),
            2: (5.0,   30.0)
        }
        self.current_state = 0

    def step(self):
        # Sample next state
        self.current_state = self.rng.choice([0, 1, 2], p=self.P[self.current_state])
        low, high = self.bandwidth_ranges[self.current_state]
        b_t = self.rng.uniform(low, high)
        return b_t, self.current_state

    def reset(self, initial_state=0):
        self.current_state = initial_state


class BandwidthCorruptor:
    """
    Applies time-windowed bandwidth-aware dropouts to aligned MOSEI audio sequences.
    """
    def __init__(self, tau_audio=32.0, delta_t=5.0, audio_fps=100.0, seed=42):
        self.tau_audio = tau_audio       # Bitrate threshold for audio in kbps
        self.delta_t = delta_t           # Window length in seconds
        self.audio_fps = audio_fps       # COVAREP is 100 Hz (10ms per frame)
        self.window_frames = int(delta_t * audio_fps)
        self.bandwidth_model = MarkovBandwidthProcess(seed=seed)
        self.rng = np.random.RandomState(seed)

    def corrupt_utterance(self, clean_audio, utterance_duration):
        """
        clean_audio: (T, D_a) numpy array of acoustic features
        utterance_duration: duration in seconds
        
        Returns:
          corrupt_audio: (T, D_a)
          mask_audio: (T,) binary array (1 = received, 0 = dropped)
          bandwidth_timeline: list of (t_start, t_end, B_k, state, dropped)
        """
        total_frames = clean_audio.shape[0]
        mask_audio = np.ones(total_frames, dtype=np.float32)
        corrupt_audio = clean_audio.copy()
        
        num_windows = max(1, int(np.ceil(utterance_duration / self.delta_t)))
        bandwidth_timeline = []

        self.bandwidth_model.reset(initial_state=0)

        for k in range(num_windows):
            w_start_time = k * self.delta_t
            w_end_time = min((k + 1) * self.delta_t, utterance_duration)
            
            f_start = int(w_start_time * self.audio_fps)
            f_end = min(int(w_end_time * self.audio_fps), total_frames)
            
            if f_start >= total_frames:
                break
                
            w_len_frames = f_end - f_start
            b_k, state_k = self.bandwidth_model.step()
            
            is_dropped = False
            drop_frames_count = 0

            # Bandwidth Threshold Rule:
            if b_k < self.tau_audio:
                is_dropped = True
                # Severity-proportional dropout duration:
                # Greater deficit (tau - b_k) causes longer loss duration
                deficit_ratio = (self.tau_audio - b_k) / self.tau_audio  # between 0 and 1
                # Drop between 40% and 100% of the window
                drop_ratio = self.rng.uniform(0.40, 1.0) * deficit_ratio
                drop_frames_count = int(np.clip(drop_ratio * w_len_frames, 1, w_len_frames))
                
                # Pick a random onset within the window
                max_offset = max(0, w_len_frames - drop_frames_count)
                offset = self.rng.randint(0, max_offset + 1) if max_offset > 0 else 0
                
                drop_start = f_start + offset
                drop_end = drop_start + drop_frames_count
                
                # Step 4: Masking & Availability Indicators
                mask_audio[drop_start:drop_end] = 0.0
                corrupt_audio[drop_start:drop_end] = 0.0

            bandwidth_timeline.append({
                "window": k,
                "start_sec": round(w_start_time, 2),
                "end_sec": round(w_end_time, 2),
                "bandwidth_kbps": round(b_k, 2),
                "state": ["High", "Medium", "Low"][state_k],
                "dropped": is_dropped,
                "dropped_frames": drop_frames_count,
                "window_frames": w_len_frames
            })

        return corrupt_audio, mask_audio, bandwidth_timeline


if __name__ == "__main__":
    print("=" * 80)
    print("BANDWIDTH SIMULATION ENGINE - STANDALONE DEMO TEST")
    print("=" * 80)
    
    # Create a synthetic 15-second acoustic feature sequence (100 Hz, 74 features)
    duration_sec = 15.0
    fps = 100.0
    total_frames = int(duration_sec * fps)
    dummy_clean_audio = np.random.randn(total_frames, 74).astype(np.float32)
    
    print(f"\nCreated synthetic clean audio: {total_frames} frames ({duration_sec}s at {fps} Hz, 74 features)")
    print("Simulating time-windowed bandwidth corruption (Window Δt = 3.0s, Threshold τ = 32.0 kbps)...\n")
    
    corruptor = BandwidthCorruptor(tau_audio=32.0, delta_t=3.0, audio_fps=fps, seed=42)
    corrupt_audio, mask_audio, timeline = corruptor.corrupt_utterance(dummy_clean_audio, duration_sec)
    
    dropped_frames = int(np.sum(mask_audio == 0.0))
    drop_pct = (dropped_frames / total_frames) * 100.0
    
    print("=" * 80)
    print(f"{'Window':<8} {'Time Span':<16} {'Bandwidth B(t)':<18} {'State':<12} {'Status':<16} {'Frames Masked'}")
    print("-" * 80)
    for w in timeline:
        t_span = f"[{w['start_sec']:>4.1f}s - {w['end_sec']:>4.1f}s]"
        bw_str = f"{w['bandwidth_kbps']:>6.1f} kbps"
        if w["dropped"]:
            status_str = "DROPOUT DETECTED"
            mask_str = f"Dropped {w['dropped_frames']}/{w['window_frames']} frames"
        else:
            status_str = "INTACT (OK)"
            mask_str = "0 frames dropped (All 1.0)"
        print(f"{w['window']:<8} {t_span:<16} {bw_str:<18} {w['state']:<12} {status_str:<16} {mask_str}")
    
    print("=" * 80)
    print(f"Summary: Dropped {dropped_frames}/{total_frames} frames ({drop_pct:.2f}% drop rate)")
    print("=" * 80)
    print("\n[NOTE] To run the full dataset pipeline against actual CMU-MOSEI .csd files, run:")
    print("  python run_bandwidth_simulation_protocol.py\n")


