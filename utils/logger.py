"""Logging utilities and WandB integration."""

import wandb
from typing import Dict, Any, Optional


class Logger:
    """Logger class with WandB integration."""
    
    def __init__(self, project: str, config: Dict[str, Any], use_wandb: bool = True, run_name: Optional[str] = None):
        """Initialize logger.
        
        Args:
            project: WandB project name
            config: Configuration dictionary
            use_wandb: Whether to use WandB logging
            run_name: Optional name for the run
        """
        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(project=project, config=config, name=run_name)
    
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        if self.use_wandb:
            wandb.log(metrics, step=step)
        else:
            print(f"Step {step}: {metrics}" if step is not None else f"Metrics: {metrics}")
    
    def finish(self):
        """Finish logging."""
        if self.use_wandb:
            wandb.finish()
