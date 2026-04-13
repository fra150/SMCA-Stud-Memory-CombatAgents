"""
Countdown — Evolutionary Pressure Mechanism for the SMCA Arena.

The countdown is not just a deadline — it's a selective pressure that forces
the system to mature. When time runs out, the current champion wins automatically.
"""

import time
import threading
from typing import List, Dict, Callable, Optional, Any


class Countdown:
    """Evolutionary pressure timer for the SMCA Arena.
    
    Simulates biological urgency: time pressure forces rapid convergence
    toward the best available solution — not the perfect one, but the
    optimal one in context.
    """
    
    def __init__(self, total_seconds: float, 
                 pressure_thresholds: Optional[List[float]] = None):
        """Initialize the Countdown.
        
        Args:
            total_seconds: Total time for the countdown
            pressure_thresholds: List of threshold fractions (e.g., [0.5, 0.25, 0.1])
                                where callbacks fire
        """
        self.total_seconds = total_seconds
        self.thresholds = pressure_thresholds or [0.5, 0.25, 0.1]
        self.thresholds.sort(reverse=True)  # Ensure descending order
        
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._is_running = False
        self._is_paused = False
        self._pause_time: Optional[float] = None
        self._total_paused: float = 0.0
        
        # Threshold callbacks: threshold_value -> list of callbacks
        self._callbacks: Dict[float, List[Callable]] = {}
        self._triggered_thresholds: set = set()
        
        # Monitor thread
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor = False
        
        # Pressure history for visualization
        self.pressure_history: List[Dict[str, Any]] = []
        
    def start(self) -> None:
        """Start the countdown timer."""
        self._start_time = time.time()
        self._end_time = self._start_time + self.total_seconds
        self._is_running = True
        self._is_paused = False
        self._triggered_thresholds = set()
        self._total_paused = 0.0
        
        # Record initial state
        self.pressure_history.append({
            'time': 0.0,
            'remaining': self.total_seconds,
            'pressure': 0.0,
            'event': 'start'
        })
        
        # Start monitor thread for threshold callbacks
        if self._callbacks:
            self._stop_monitor = False
            self._monitor_thread = threading.Thread(
                target=self._monitor_thresholds, daemon=True
            )
            self._monitor_thread.start()
    
    def stop(self) -> float:
        """Stop the countdown and return elapsed time."""
        self._is_running = False
        self._stop_monitor = True
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
        
        elapsed = self.get_elapsed()
        
        self.pressure_history.append({
            'time': elapsed,
            'remaining': max(0, self.get_remaining()),
            'pressure': self.get_pressure_level(),
            'event': 'stop'
        })
        
        return elapsed
    
    def pause(self) -> None:
        """Pause the countdown."""
        if self._is_running and not self._is_paused:
            self._is_paused = True
            self._pause_time = time.time()
    
    def resume(self) -> None:
        """Resume the countdown."""
        if self._is_paused and self._pause_time:
            paused_duration = time.time() - self._pause_time
            self._total_paused += paused_duration
            self._end_time += paused_duration
            self._is_paused = False
            self._pause_time = None
    
    def get_remaining(self) -> float:
        """Get remaining time in seconds."""
        if not self._is_running or self._start_time is None:
            return self.total_seconds
        
        if self._is_paused and self._pause_time:
            # Don't count paused time
            effective_now = self._pause_time
        else:
            effective_now = time.time()
        
        remaining = self._end_time - effective_now
        return max(0.0, remaining)
    
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds."""
        if self._start_time is None:
            return 0.0
        return self.total_seconds - self.get_remaining()
    
    def get_fraction_remaining(self) -> float:
        """Get fraction of time remaining (1.0 = full, 0.0 = expired)."""
        if self.total_seconds <= 0:
            return 0.0
        return self.get_remaining() / self.total_seconds
    
    def get_pressure_level(self) -> float:
        """Get current pressure level (0.0 = relaxed, 1.0 = maximum pressure).
        
        Pressure increases exponentially as time runs out, simulating
        the biological urgency of the "Tombola effect".
        """
        fraction_remaining = self.get_fraction_remaining()
        if fraction_remaining >= 1.0:
            return 0.0
        if fraction_remaining <= 0.0:
            return 1.0
        
        # Exponential pressure curve: pressure = 1 - fraction^0.5
        # This makes pressure increase faster as time runs out
        pressure = 1.0 - (fraction_remaining ** 0.5)
        return min(max(pressure, 0.0), 1.0)
    
    def is_expired(self) -> bool:
        """Check if the countdown has expired."""
        return self._is_running and self.get_remaining() <= 0.0
    
    def is_running(self) -> bool:
        """Check if the countdown is currently running."""
        return self._is_running and not self.is_expired()
    
    def on_threshold(self, threshold: float, callback: Callable) -> None:
        """Register a callback for when a specific fraction of time remains.
        
        Args:
            threshold: Fraction of time remaining (e.g., 0.5 = 50%)
            callback: Function to call when threshold is crossed
        """
        if threshold not in self._callbacks:
            self._callbacks[threshold] = []
        self._callbacks[threshold].append(callback)
    
    def get_max_rounds(self, base_rounds: int) -> int:
        """Calculate maximum rounds allowed based on remaining time.
        
        As pressure increases, available rounds decrease — forcing convergence.
        
        Args:
            base_rounds: Maximum rounds with no time pressure
            
        Returns:
            Adjusted number of rounds
        """
        fraction = self.get_fraction_remaining()
        if fraction <= 0.0:
            return 0  # No more rounds — champion wins
        
        # Linear reduction: fraction * base_rounds, minimum 1
        adjusted = max(1, int(fraction * base_rounds))
        return adjusted
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive countdown status."""
        return {
            'total_seconds': self.total_seconds,
            'remaining': self.get_remaining(),
            'elapsed': self.get_elapsed(),
            'fraction_remaining': self.get_fraction_remaining(),
            'pressure_level': self.get_pressure_level(),
            'is_running': self._is_running,
            'is_paused': self._is_paused,
            'is_expired': self.is_expired(),
            'triggered_thresholds': list(self._triggered_thresholds),
            'pressure_history_length': len(self.pressure_history)
        }
    
    def _monitor_thresholds(self) -> None:
        """Background thread monitoring threshold crossings."""
        while not self._stop_monitor and self._is_running:
            if not self._is_paused:
                fraction = self.get_fraction_remaining()
                pressure = self.get_pressure_level()
                
                # Record pressure for visualization
                self.pressure_history.append({
                    'time': self.get_elapsed(),
                    'remaining': self.get_remaining(),
                    'pressure': pressure,
                    'event': 'tick'
                })
                
                # Check thresholds
                for threshold in self.thresholds:
                    if threshold not in self._triggered_thresholds and fraction <= threshold:
                        self._triggered_thresholds.add(threshold)
                        # Fire callbacks
                        for callback in self._callbacks.get(threshold, []):
                            try:
                                callback(threshold, fraction, pressure)
                            except Exception as e:
                                print(f"Countdown callback error at {threshold}: {e}")
                
                if fraction <= 0.0:
                    self._is_running = False
                    break
            
            time.sleep(0.1)  # Check every 100ms
    
    def __repr__(self) -> str:
        status = "EXPIRED" if self.is_expired() else (
            "PAUSED" if self._is_paused else (
                "RUNNING" if self._is_running else "READY"
            )
        )
        return (f"Countdown({self.total_seconds}s, {status}, "
                f"pressure={self.get_pressure_level():.2f})")
