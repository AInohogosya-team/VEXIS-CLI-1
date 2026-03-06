"""
Main Application Entry Point for VEXIS AI Agent System
Provides the main CLI interface for the AI agent
"""

import sys
import argparse
from typing import Optional, Dict, Any

from ..core_processing.two_phase_engine import TwoPhaseEngine
from ..utils.exceptions import AIAgentException
from ..utils.logger import get_logger, setup_logging
from ..utils.config import load_config


class MainAIAgent:
    """Main AI Agent application"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = load_config(config_path) if config_path else load_config()
        self.logger = get_logger("main_app")
        
        # Initialize two-phase engine
        engine_config = {
            "click_delay": getattr(self.config.engine, 'click_delay', 0.1),
            "typing_delay": getattr(self.config.engine, 'typing_delay', 0.05),
            "scroll_duration": getattr(self.config.engine, 'scroll_duration', 0.5),
            "drag_duration": getattr(self.config.engine, 'drag_duration', 0.3),
            "screenshot_quality": getattr(self.config.engine, 'screenshot_quality', 95),
            "screenshot_format": getattr(self.config.engine, 'screenshot_format', 'PNG'),
            "max_task_retries": getattr(self.config.engine, 'max_task_retries', 3),
            "max_command_retries": getattr(self.config.engine, 'max_command_retries', 3),
            "command_timeout": getattr(self.config.engine, 'command_timeout', 30),
            "task_timeout": getattr(self.config.engine, 'task_timeout', 300),
        }
        
        self.engine = TwoPhaseEngine(engine_config)
        self.logger.info("Main AI Agent initialized")
    
    def run(self, instruction: str, options: Dict[str, Any]) -> int:
        """Run AI Agent with instruction"""
        try:
            self.logger.info(
                "Starting Main AI Agent execution",
                instruction=instruction,
                options=options,
            )
            
            # Setup logging if requested
            if options.get("verbose"):
                setup_logging(level="DEBUG")
            elif options.get("log_file"):
                setup_logging(file_path=options["log_file"])
            
            # Validate instruction
            if not instruction or not instruction.strip():
                self.logger.error("Instruction cannot be empty")
                return 1
            
            # Execute instruction using two-phase engine
            execution_context = self.engine.execute_instruction(instruction)
            
            # Simple success check
            from ..core_processing.two_phase_engine import ExecutionPhase
            success = execution_context.phase == ExecutionPhase.COMPLETED
            
            # Print results
            if not options.get("quiet"):
                print(f"\n{'='*60}")
                print("AI AGENT EXECUTION SUMMARY")
                print(f"{'='*60}")
                print(f"Instruction: {instruction}")
                print(f"Success: {success}")
                print(f"Executed Commands: {len(execution_context.executed_commands)}")
                if execution_context.error:
                    print(f"Error: {execution_context.error}")
                print(f"{'='*60}")
            
            # Return exit code based on success
            return 0 if success else 1
                
        except AIAgentException as e:
            self.logger.error(f"Main AI Agent error: {e}")
            print(f"Error: {e}", file=sys.stderr)
            return 3
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            print(f"Unexpected error: {e}", file=sys.stderr)
            return 4


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="VEXIS AI Agent - Vision-based GUI automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Take a screenshot"
  %(prog)s "Click the button at coordinates (100, 200)"
  %(prog)s --verbose "Open web browser and search for AI"
        """
    )
    
    # Positional arguments
    parser.add_argument(
        "instruction",
        type=str,
        help="Natural language instruction for the AI agent"
    )
    
    # Configuration options
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    # Output options
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output except errors"
    )
    
    # Logging options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log to specified file"
    )
    
    return parser


def main():
    """Main entry point"""
    # Parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create Main AI Agent
    try:
        agent = MainAIAgent(args.config)
    except Exception as e:
        print(f"Failed to initialize Main AI Agent: {e}", file=sys.stderr)
        return 1
    
    # Prepare options
    options = {
        "verbose": args.verbose,
        "quiet": args.quiet,
        "log_file": args.log_file,
    }
    
    # Run Main AI Agent
    exit_code = agent.run(args.instruction, options)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
