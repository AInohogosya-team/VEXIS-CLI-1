"""
AI Agent System - CLI-based automation
Production-ready implementation with zero-defect policy

Version Note: VEXIS-1.1 uses simplified versioning (1.1 instead of 1.1.0.1)
with 0.0.1 increments omitted to reduce complexity while maintaining compatibility.
"""

__version__ = "1.1"
__author__ = "AI Agent Team"
__email__ = "contact@vexis-1.1.org"
__description__ = "CLI-based AI agent for command-line automation"

# Import core components for easy access
from .core_processing.command_parser import CommandParser
from .core_processing.two_phase_engine import TwoPhaseEngine
from .platform_abstraction.platform_detector import PlatformDetector
from .external_integration.vision_api_client import VisionAPIClient
from .external_integration.model_runner import ModelRunner

__all__ = [
    "CommandParser", 
    "TwoPhaseEngine",
    "PlatformDetector",
    "VisionAPIClient",
    "ModelRunner",
]
