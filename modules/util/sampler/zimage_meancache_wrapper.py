"""
MeanCache Wrapper for Z-Image Flow Matching Models

This module implements a lightweight wrapper specifically for Z-Image transformer,
intercepting forward calls to enable MeanCache acceleration.
"""

import torch
from typing import Any
import logging

logger = logging.getLogger(__name__)


class ZImageMeanCacheWrapper:
    """
    A wrapper for Z-Image transformer that enables MeanCache acceleration.
    
    This wrapper intercepts forward() calls and decides whether to:
    1. Compute the actual velocity (first few steps, or when needed)
    2. Use cached JVP-corrected velocity (when stable)
    """
    
    def __init__(
        self,
        transformer: torch.nn.Module,
        preset: str = "balanced",
        enabled: bool = True,
    ):
        """
        Initialize MeanCache wrapper for Z-Image.
        
        Args:
            transformer: The Z-Image transformer model
            preset: Preset profile ("quality", "balanced", "speed", "turbo")
            enabled: Whether MeanCache is enabled
        """
        self.transformer = transformer
        self.enabled = enabled
        self.preset = preset
        
        # Preset configurations  
        self.presets = {
            "quality": {"skip_threshold": 0.15, "target_skip_rate": 0.30},  # ~1.4x
            "balanced": {"skip_threshold": 0.12, "target_skip_rate": 0.40}, # ~1.67x
            "speed": {"skip_threshold": 0.10, "target_skip_rate": 0.45},    # ~1.8x
            "turbo": {"skip_threshold": 0.08, "target_skip_rate": 0.50},    # ~2.0x
        }
        
        self.config = self.presets.get(preset, self.presets["balanced"])
        
        # State tracking
        self.reset_state()
        
    def reset_state(self):
        """Reset internal state for a new sampling run."""
        self.prev_velocity = None
        self.prev_timestep = None
        self.prev_jvp = None
        
        self.total_steps = 0
        self.skipped_steps = 0
        self.computed_steps = 0
        
    def __call__(
        self,
        latent_list: list[torch.Tensor],
        timestep: torch.Tensor,
        prompt_embedding: torch.Tensor,
        **kwargs
    ) -> Any:
        """
        Forward pass with MeanCache acceleration for Z-Image.
        
        Args:
            latent_list: List of latent tensors
            timestep: Normalized timestep (0-1)
            prompt_embedding: Text encoder embeddings
            **kwargs: Additional arguments
            
        Returns:
            Model output (with .sample attribute)
        """
        if not self.enabled:
            return self.transformer(latent_list, timestep, prompt_embedding, **kwargs)
        
        self.total_steps += 1
        
        # First step always computes
        if self.prev_velocity is None:
            output = self._compute(latent_list, timestep, prompt_embedding, **kwargs)
            self.prev_velocity = self._extract_velocity(output)
            self.prev_timestep = timestep
            self.computed_steps += 1
            logger.debug(f"[MeanCache] Step {self.total_steps}: Initial computation")
            return output
        
        # Compute JVP approximation
        dt = timestep - self.prev_timestep
        
        # Try to use cached JVP-corrected velocity
        if self.prev_jvp is not None and self._should_skip():
            # Skip this step, use predicted velocity
            predicted_velocity = self.prev_velocity + dt * self.prev_jvp
            self.skipped_steps += 1
            logger.debug(f"[MeanCache] Step {self.total_steps}: SKIPPED (t={float(timestep):.4f})")
            
            # Return a fake output with the predicted velocity
            # Note: This is a simplified version that may need adjustment
            return self._create_output_from_velocity(predicted_velocity)
        
        # Compute actual velocity
        output = self._compute(latent_list, timestep, prompt_embedding, **kwargs)
        velocity = self._extract_velocity(output)
        self.computed_steps += 1
        logger.debug(f"[MeanCache] Step {self.total_steps}: Computed (t={float(timestep):.4f})")
        
        # Update JVP
        if dt.abs() > 1e-6:
            self.prev_jvp = (velocity - self.prev_velocity) / dt
            
        self.prev_velocity = velocity
        self.prev_timestep = timestep
        
        return output
    
    def _compute(self, latent_list, timestep, prompt_embedding, **kwargs):
        """Compute velocity from the underlying transformer."""
        return self.transformer(latent_list, timestep, prompt_embedding, **kwargs)
    
    def _extract_velocity(self, output):
        """Extract velocity from model output."""
        # Z-Image returns output with .sample attribute
        if hasattr(output, 'sample'):
            # Stack the list of tensors
            return -torch.stack(output.sample, dim=0)
        else:
            return output
    
    def _create_output_from_velocity(self, velocity):
        """
        Create a fake output structure from predicted velocity.
        
        NOTE: This is a simplified implementation. The actual Z-Image output
        structure may be more complex and require proper handling.
        """
        # Create a simple object with .sample attribute
        class FakeOutput:
            def __init__(self, sample):
                self.sample = sample
        
        # Un-stack the velocity back to list format
        velocity_list = list(velocity.unbind(dim=0))
        return FakeOutput(sample=velocity_list)
    
    def _should_skip(self) -> bool:
        """
        Decide whether to skip this step based on heuristic.
        
        Uses a simple target-based approach:
        - Skip more in the middle of trajectory (0.3 < t < 0.7)
        - Skip less at the beginning and end
        """
        current_skip_rate = self.skipped_steps / max(self.total_steps, 1)
        target_rate = self.config["target_skip_rate"]
        
        # Skip more aggressively in middle of trajectory
        if self.prev_timestep is not None:
            t_val = float(self.prev_timestep)
            if 0.3 < t_val < 0.7:
                adjusted_target = target_rate * 1.2
            else:
                adjusted_target = target_rate * 0.8
        else:
            adjusted_target = target_rate
        
        return current_skip_rate < adjusted_target
    
    def get_stats(self) -> dict:
        """Get sampling statistics."""
        skip_rate = self.skipped_steps / max(self.total_steps, 1)
        speedup = 1.0 / (1.0 - skip_rate) if skip_rate < 1.0 else 1.0
        
        return {
            "total_steps": self.total_steps,
            "skipped_steps": self.skipped_steps,
            "computed_steps": self.computed_steps,
            "skip_rate": skip_rate,
            "speedup": speedup,
        }
    
    def print_summary(self):
        """Print sampling summary to console."""
        stats = self.get_stats()
        logger.info(
            f"[MeanCache] Sampling complete ({self.preset}): "
            f"{stats['total_steps']} steps, "
            f"{stats['skipped_steps']} skipped, "
            f"{stats['computed_steps']} computed "
            f"({stats['skip_rate']*100:.1f}% skip rate, ~{stats['speedup']:.2f}x speedup)"
        )
