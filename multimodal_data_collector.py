#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi 5 Edge ML Motor Condition Monitoring Data Collector
Sensors:
  1. AMG8833 (Grid-EYE 8x8 Thermal Camera) - I2C Address: 0x69 (or 0x68)
  2. ADXL345 (3-Axis Digital Accelerometer) - I2C Address: 0x53

Description:
  This script utilizes multi-threading to collect high-frequency vibration data (ADXL345) 
  and synchronize it with low-frequency thermal grids (AMG8833). It extracts time-domain 
  and frequency-domain (FFT) features, saving raw tensors (.npz) for deep learning 
  and statistical features (.csv) for Isolation Forest anomaly detection.
"""

import argparse
import csv
import datetime
import json
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional visualization and scaling libraries
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from scipy.ndimage import zoom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class MultimodalMotorCollector:
    """
    Acquires synchronized data from AMG8833 and ADXL345 via I2C, computes 
    predictive maintenance features, and logs datasets for Edge AI training.
    """

    def __init__(
        self,
        output_dir: str = "dataset_motor",
        label: str = "normal",
        vib_samples_per_window: int = 128,
        vib_rate_hz: float = 200.0,
        window_interval_sec: float = 0.5,
        amg_addr: int = 0x69,
        adxl_addr: int = 0x53,
        upscale_size: int = 32,
        simulate: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.label = label
        self.vib_samples_per_window = vib_samples_per_window
        self.vib_rate_hz = vib_rate_hz
        self.window_interval_sec = window_interval_sec
        self.amg_addr = amg_addr
        self.adxl_addr = adxl_addr
        self.upscale_size = upscale_size
        self.simulate = simulate

        # Mutex to prevent I2C bus collisions between threads
        self.i2c_lock = threading.Lock()
        self.running = False

        # Circular buffer for vibration data: (timestamp, ax, ay, az)
        self.vib_buffer: deque = deque(maxlen=vib_samples_per_window * 4)

        # Prepare directory structure
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.raw_dir = self.output_dir / "raw_npz" / self.label
        self.thermal_img_dir = self.output_dir / "thermal_png" / self.label
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.thermal_img_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / f"features_{self.label}_{self.session_id}.csv"
        self.meta_path = self.output_dir / f"session_meta_{self.session_id}.json"

        self.amg = None
        self.adxl = None
        self._init_sensors()

    def _init_sensors(self) -> None:
        """Initializes the AMG8833 and ADXL345 sensors over the I2C bus."""
        if self.simulate:
            print("[INFO] Simulation mode active. Hardware checks bypassed.")
            return

        try:
            import board
            import busio
            import adafruit_amg88xx
            import adafruit_adxl34x
        except ImportError as exc:
            print(f"[ERROR] Missing CircuitPython libraries: {exc}")
            sys.exit(1)

        try:
            i2c = busio.I2C(board.SCL, board.SDA)
            self.amg = adafruit_amg88xx.AMG88XX(i2c, addr=self.amg_addr)
            self.adxl = adafruit_adxl34x.ADXL345(i2c, address=self.adxl_addr)

            # Set measurement range to +-16G and rate to 400Hz for motor analytics
            self.adxl.range = adafruit_adxl34x.Range.RANGE_16_G
            self.adxl.data_rate = adafruit_adxl34x.DataRate.RATE_400_HZ
            print(f"[SUCCESS] AMG8833 (0x{self.amg_addr:02X}) and ADXL345 (0x{self.adxl_addr:02X}) initialized.")
            time.sleep(0.2)
        except Exception as exc:
            print(f"[ERROR] Failed to initialize I2C sensors: {exc}")
            sys.exit(1)

    def _vibration_worker(self) -> None:
        """
        Background thread for high-frequency ADXL345 polling.
        Safely shares the I2C bus using a threading lock.
        """
        target_dt = 1.0 / self.vib_rate_hz
        sim_phase = 0.0

        while self.running:
            t_start = time.perf_counter()
            now_ts = time.time()

            if self.simulate:
                sim_phase += 2.0 * np.pi * 50.0 * target_dt
                fault_gain = 3.5 if self.label != "normal" else 1.0
                ax = fault_gain * 1.2 * np.sin(sim_phase) + np.random.normal(0, 0.2)
                ay = fault_gain * 0.9 * np.cos(sim_phase * 2) + np.random.normal(0, 0.2)
                az = 9.81 + fault_gain * 0.6 * np.sin(sim_phase * 3) + np.random.normal(0, 0.25)
            else:
                with self.i2c_lock:
                    ax, ay, az = self.adxl.acceleration

            self.vib_buffer.append((now_ts, float(ax), float(ay), float(az)))

            elapsed = time.perf_counter() - t_start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read_thermal_grid(self) -> np.ndarray:
        """Reads the 8x8 temperature matrix (°C) from the AMG8833."""
        if self.simulate:
            base_temp = 45.0 if self.label != "normal" else 30.0
            grid = np.random.normal(loc=base_temp, scale=1.2, size=(8, 8))
            grid[2:6, 2:6] += 8.5 if self.label != "normal" else 3.0
            return grid.astype(np.float32)

        with self.i2c_lock:
            pixels = self.amg.pixels
        return np.array(pixels, dtype=np.float32)

    def upscale_thermal(self, grid_8x8: np.ndarray) -> np.ndarray:
        """Upscales the 8x8 grid to a CNN-compatible size (e.g., 32x32) using bicubic interpolation."""
        if self.upscale_size == 8:
            return grid_8x8

        if HAS_CV2:
            return cv2.resize(grid_8x8, (self.upscale_size, self.upscale_size), interpolation=cv2.INTER_CUBIC)
        if HAS_SCIPY:
            scale = self.upscale_size / 8.0
            return zoom(grid_8x8, scale, order=3).astype(np.float32)

        repeat_factor = self.upscale_size // 8
        return np.kron(grid_8x8, np.ones((repeat_factor, repeat_factor), dtype=np.float32))

    @staticmethod
    def compute_vibration_features(vib_window: np.ndarray, fs: float) -> Dict[str, float]:
        """Computes time and frequency domain (FFT) features from an (N, 3) vibration window."""
        features: Dict[str, float] = {}
        axes = ["x", "y", "z"]

        vib_zero_mean = vib_window - np.mean(vib_window, axis=0, keepdims=True)

        for idx, axis_name in enumerate(axes):
            sig = vib_zero_mean[:, idx]
            rms = float(np.sqrt(np.mean(sig ** 2)) + 1e-8)
            peak = float(np.max(np.abs(sig)))
            p2p = float(np.ptp(sig))
            crest_factor = float(peak / rms)

            std = float(np.std(sig) + 1e-8)
            kurtosis = float(np.mean(((sig - np.mean(sig)) / std) ** 4))

            fft_vals = np.abs(np.fft.rfft(sig))
            fft_freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
            if len(fft_vals) > 1:
                dom_idx = int(np.argmax(fft_vals[1:]) + 1)
                dom_freq = float(fft_freqs[dom_idx])
                spectral_energy = float(np.sum(fft_vals ** 2) / len(sig))
            else:
                dom_freq = 0.0
                spectral_energy = 0.0

            features[f"vib_{axis_name}_rms"] = round(rms, 5)
            features[f"vib_{axis_name}_peak"] = round(peak, 5)
            features[f"vib_{axis_name}_p2p"] = round(p2p, 5)
            features[f"vib_{axis_name}_crest"] = round(crest_factor, 5)
            features[f"vib_{axis_name}_kurtosis"] = round(kurtosis, 5)
            features[f"vib_{axis_name}_dom_freq_hz"] = round(dom_freq, 2)
            features[f"vib_{axis_name}_spec_energy"] = round(spectral_energy, 5)

        mag = np.linalg.norm(vib_zero_mean, axis=1)
        features["vib_mag_rms"] = round(float(np.sqrt(np.mean(mag ** 2))), 5)
        features["vib_mag_max"] = round(float(np.max(mag)), 5)
        return features

    @staticmethod
    def compute_thermal_features(grid_8x8: np.ndarray) -> Dict[str, float]:
        """Extracts statistical temperature features from the 8x8 thermal matrix."""
        t_min, t_max = float(np.min(grid_8x8)), float(np.max(grid_8x8))
        t_mean, t_std = float(np.mean(grid_8x8)), float(np.std(grid_8x8))
        center_mean = float(np.mean(grid_8x8[2:6, 2:6]))
        hotspot_delta = t_max - t_mean

        return {
            "temp_min_c": round(t_min, 2),
            "temp_max_c": round(t_max, 2),
            "temp_mean_c": round(t_mean, 2),
            "temp_std_c": round(t_std, 2),
            "temp_center_mean_c": round(center_mean, 2),
            "temp_hotspot_delta_c": round(hotspot_delta, 2),
        }

    def collect(self, duration_sec: float, preview: bool = False) -> None:
        """Executes the synchronized data acquisition loop."""
        self.running = True
        vib_thread = threading.Thread(target=self._vibration_worker, daemon=True)
        vib_thread.start()

        print(f"[INFO] Populating vibration buffer ({self.vib_samples_per_window} samples)...")
        while len(self.vib_buffer) < self.vib_samples_per_window:
            time.sleep(0.05)

        print(f"\n[STARTING] Collecting '{self.label}' data for {duration_sec} seconds...\nPress Ctrl+C to stop.\n")

        start_time = time.time()
        sample_idx = 0
        csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
        csv_writer = None

        try:
            while (time.time() - start_time) < duration_sec:
                loop_start = time.perf_counter()
                now_iso = datetime.datetime.now().isoformat()
                sample_id = f"{self.label}_{self.session_id}_{sample_idx:05d}"

                thermal_8x8 = self.read_thermal_grid()
                thermal_upscaled = self.upscale_thermal(thermal_8x8)

                buffer_snapshot = list(self.vib_buffer)[-self.vib_samples_per_window :]
                vib_timestamps = np.array([row[0] for row in buffer_snapshot], dtype=np.float64)
                vib_xyz = np.array([row[1:] for row in buffer_snapshot], dtype=np.float32)

                effective_fs = self.vib_rate_hz
                if len(vib_timestamps) > 1 and (vib_timestamps[-1] - vib_timestamps[0]) > 0:
                    effective_fs = (len(vib_timestamps) - 1) / (vib_timestamps[-1] - vib_timestamps[0])

                npz_path = self.raw_dir / f"{sample_id}.npz"
                np.savez_compressed(
                    npz_path,
                    thermal_8x8=thermal_8x8,
                    vibration_xyz=vib_xyz,
                    effective_fs=np.float32(effective_fs),
                    label=self.label,
                )

                therm_feats = self.compute_thermal_features(thermal_8x8)
                vib_feats = self.compute_vibration_features(vib_xyz, fs=effective_fs)

                row_dict = {
                    "sample_id": sample_id,
                    "timestamp": now_iso,
                    "label": self.label,
                    "effective_vib_fs_hz": round(float(effective_fs), 1),
                    **therm_feats,
                    **vib_feats,
                }

                if csv_writer is None:
                    csv_writer = csv.DictWriter(csv_file, fieldnames=list(row_dict.keys()))
                    csv_writer.writeheader()

                csv_writer.writerow(row_dict)
                csv_file.flush()

                print(
                    f"[{sample_idx:04d}] Label: {self.label} | "
                    f"Max Temp: {therm_feats['temp_max_c']:.1f}°C | "
                    f"Vib RMS: {vib_feats['vib_mag_rms']:.3f} m/s² | "
                    f"Dom Freq (Z): {vib_feats['vib_z_dom_freq_hz']:.1f} Hz"
                )

                sample_idx += 1
                elapsed = time.perf_counter() - loop_start
                sleep_rem = self.window_interval_sec - elapsed
                if sleep_rem > 0:
                    time.sleep(sleep_rem)

        except KeyboardInterrupt:
            print("\n[INFO] Data collection stopped by user.")
        finally:
            self.running = False
            vib_thread.join(timeout=1.0)
            csv_file.close()
            print(f"\n[DONE] Saved {sample_idx} multimodal samples.\nFeatures: {self.csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Motor Condition Monitoring Collector")
    parser.add_argument("--output-dir", type=str, default="dataset_motor", help="Output directory")
    parser.add_argument("--label", type=str, default="normal", help="Operating condition label")
    parser.add_argument("--duration", type=float, default=60.0, help="Recording duration in seconds")
    parser.add_argument("--interval", type=float, default=0.5, help="Seconds between multimodal windows")
    parser.add_argument("--simulate", action="store_true", help="Run with synthetic data (no sensors required)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collector = MultimodalMotorCollector(
        output_dir=args.output_dir,
        label=args.label,
        window_interval_sec=args.interval,
        simulate=args.simulate,
    )
    collector.collect(duration_sec=args.duration)