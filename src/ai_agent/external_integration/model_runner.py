"""
Model Runner for CLI AI Agent System
Multi-Provider Support: 13+ AI providers available
"""

import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

from .multi_provider_vision_client import MultiProviderVisionAPIClient, APIRequest, APIProvider
from ..utils.exceptions import ValidationError
from ..utils.logger import get_logger
from ..utils.config import load_config


class TaskType(Enum):
    """Task types for CLI Architecture"""
    TASK_GENERATION = "task_generation"
    COMMAND_PARSING = "command_parsing"


@dataclass
class ModelRequest:
    """Model request structure"""
    task_type: TaskType
    prompt: str
    image_data: Optional[bytes] = None
    image_format: str = "PNG"
    context: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    max_tokens: int = 5000
    temperature: float = 1.0
    timeout: int = 30


@dataclass
class ModelResponse:
    """Model response structure"""
    success: bool
    content: str
    task_type: TaskType
    model: str
    provider: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    latency: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PromptTemplate:
    """Prompt template manager"""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates for CLI Architecture"""
        return {
            TaskType.TASK_GENERATION.value: """# CLI Task Planning Expert System

You are an expert CLI automation assistant specializing in breaking down complex user instructions into precise, executable command-line steps.

## Mission
Transform the user's natural language instruction into a sequential list of specific, actionable CLI commands that accomplish the exact goal requested.

## Context Analysis
- **User Instruction**: {instruction}
- **Operating System**: {os_info}
- **Current Working Directory**: {current_directory}

## Core Principles
1. **Be Specific, Not Verbose** - Each step must be an exact CLI command, not a description
2. **Think Sequentially** - Commands must build upon each other logically
3. **Respect System State** - Only create/navigate directories when explicitly needed
4. **Assume Standard Tools** - Use common CLI tools available on the target OS

## Command Generation Rules

### ✅ DO:
- Output numbered steps (1., 2., 3., etc.)
- Use exact CLI syntax with proper flags and arguments
- Include file paths, URLs, or specific values when mentioned
- Chain simple, logical operations
- Consider error handling (e.g., check if directory exists before cd)
- **NO STEP LIMITATIONS** - Generate as many steps as needed to complete the task fully

### ❌ DO NOT:
- Write conversational text or explanations
- Use markdown formatting or code blocks
- Include placeholder text like "[your-file-name]"
- Skip steps assuming user will do them manually
- Use Windows commands on macOS/Linux or vice-versa

## OS-Specific Guidelines

### macOS/Linux:
- Use `open -a` for applications
- Use standard Unix commands (ls, cd, mkdir, etc.)
- Respect Unix file paths and permissions

### Windows:
- Use `start` for applications
- Use Windows command syntax
- Respect Windows file paths and CMD/PowerShell differences

## Quality Examples

**Example 1 - Application Launch**
User: "open Chrome browser"
1. open -a Google Chrome

**Example 2 - Project Setup**
User: "create a new Python project with git"
1. mkdir my_python_project
2. cd my_python_project
3. git init
4. touch README.md
5. touch main.py

**Example 3 - File Operations**
User: "download a file from GitHub and extract it"
1. curl -O https://github.com/user/repo/archive/main.zip
2. unzip main.zip
3. cd repo-main

## Output Format
```
1. [exact CLI command]
2. [exact CLI command]
3. [exact CLI command]
```

## Critical Constraint
Output ONLY the numbered command list. No explanations, no markdown, no conversational text. Each line must start with a number followed by a period and a space, then the exact command.

**IMPORTANT**: Generate as many steps as required - there are NO limitations on the number of steps. Break down complex tasks into as many individual commands as needed for complete execution.

Now analyze the user instruction and generate the precise command sequence:""",

            TaskType.COMMAND_PARSING.value: """# CLI Command Generation Expert

You are an expert CLI assistant that generates precise, contextually-aware commands based on task requirements and system state.

## Current Situation
- **Task**: {task_description}
- **Operating System**: {os_info}
- **Current Directory**: {current_directory}

## Recent Activity Context
{previous_terminal_actions}

## Last Command Output
{last_command_output}

## Decision Framework

### Step 1: Analyze Completion Status
- Is the task already fully completed based on previous actions?
- Are there any remaining subtasks or requirements?
- Did the last command succeed or fail?

### Step 2: Determine Next Action
- **If task is complete**: Output `END`
- **If previous step failed and needs retry**: Output `REGENERATE_STEP`
- **If task needs continuation**: Generate the next specific command

### Step 3: Command Generation Principles
- Build upon previous successful commands
- Don't repeat commands that were already executed
- Use exact syntax with proper paths and flags
- Consider the current working directory
- Account for command outputs/errors

## Command Selection Guidelines

### Continue Task When:
- Previous command succeeded but task isn't complete
- Need to perform the next logical step
- Must handle output from previous command

### Use END When:
- Task objective has been fully achieved
- All required files are created/modified
- User's original request is satisfied

### Use REGENERATE_STEP When:
- Previous command failed due to syntax/path issues
- Need to retry with different approach
- Command had no effect on task progress

## Output Format
```
Reasoning: [Brief analysis of current situation and why this action is needed]
Target: [What this command aims to accomplish]
Command: [The exact CLI command or END/REGENERATE_STEP]
```

## Contextual Examples

### Example 1 - Continuing Success
Previous: `mkdir project` (SUCCESS)
Task: "create Python project"
```
Reasoning: Directory created successfully, now need to initialize git repository
Target: Git repository initialization
Command: git init
```

### Example 2 - Task Complete
Previous: `python3 main.py` (SUCCESS - script ran)
Task: "run Python script"
```
Reasoning: Script executed successfully, task objective achieved
Target: Task completion
Command: END
```

### Example 3 - Retry Failed Command
Previous: `cd non_existent_dir` (FAILED - no such directory)
Task: "navigate to project directory"
```
Reasoning: Previous cd command failed because directory doesn't exist, need to create it first
Target: Directory creation before navigation
Command: REGENERATE_STEP
```

## Critical Instructions
1. Always consider the full context before deciding
2. Be precise about what needs to happen next
3. Use plain text only - no markdown formatting
4. Base decisions on actual command outcomes, not assumptions

Now analyze the current situation and generate the appropriate response:"""
        }


class ModelRunner:
    """CLI Architecture Model Runner: Ollama Cloud Models"""

    # Valid Ollama model names
    DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
    DEFAULT_GOOGLE_MODEL = "gemini-3.1-pro"

    def __init__(self, config: Optional[Dict[str, Any]] = None, auto_install_sdks: bool = False):
        self.config = config or load_config().api.__dict__
        self.logger = get_logger(__name__)
        
        # Initialize multi-provider vision client with SDK installation support
        self.vision_client = MultiProviderVisionAPIClient(self.config, auto_install_sdks=auto_install_sdks)
        self.prompt_template = PromptTemplate()

        self.logger.info(
            "Model runner initialized",
            preferred_provider=self.config.get("preferred_provider"),
            default_model=self.DEFAULT_OLLAMA_MODEL,
        )

    def run_model(self, request: ModelRequest) -> ModelResponse:
        """Run AI model for CLI Architecture"""
        start_time = time.time()

        try:
            # Validate request
            self._validate_request(request)

            # Format prompt
            prompt = self._format_prompt(request)
            
            # Get system instructions for API request
            system_instructions = self._get_system_instructions(request.task_type)

            # Use configured model directly - no provider preference logic
            model_name = self.config.get("local_model")
            if not model_name:
                raise ValidationError("No model configured. Please select a model first.")
            
            # Determine provider from model configuration
            provider_name = "ollama"  # Default to ollama for local models
            
            # Create API request
            api_request = APIRequest(
                prompt=prompt,
                image_data=request.image_data,
                image_format=request.image_format,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                model=model_name,
                provider=provider_name,
                system_instruction=system_instructions
            )

            # Make API call
            api_response = self.vision_client.generate_response(api_request)

            # Create model response
            model_response = ModelResponse(
                success=api_response.success,
                content=api_response.content,
                task_type=request.task_type,
                model=api_response.model or model_name,
                provider=api_response.provider or preferred_provider,
                tokens_used=api_response.tokens_used,
                cost=api_response.cost,
                latency=time.time() - start_time,
                error=api_response.error,
            )

            if api_response.success:
                self.logger.info(
                    "Model execution successful",
                    task_type=request.task_type.value,
                    model=model_response.model,
                    latency=model_response.latency,
                )
            else:
                self.logger.error(
                    "Model execution failed",
                    task_type=request.task_type.value,
                    error=model_response.error,
                )
                
                # Enhanced error handling for authentication issues
                if "Authentication required" in model_response.error:
                    try:
                        from ..utils.ollama_error_handler import handle_ollama_error
                        context = {
                            'model_name': model_response.model,
                            'operation': 'model_execution'
                        }
                        handle_ollama_error(model_response.error, context, display_to_user=True)
                        
                        # Prompt user to sign in
                        import sys
                        if sys.stdin.isatty():  # Only prompt if running in terminal
                            try:
                                choice = input("\nWould you like to sign in to Ollama now? (y/n): ").lower().strip()
                                if choice in ['y', 'yes']:
                                    import subprocess
                                    print("\n🔐 Opening Ollama sign-in...")
                                    result = subprocess.run(["ollama", "signin"], capture_output=False, text=True)
                                    if result.returncode == 0:
                                        print("✓ Sign-in initiated. Please complete it in your browser.")
                                        print("Then try running your command again.")
                                    else:
                                        print("✗ Failed to initiate sign-in.")
                            except (KeyboardInterrupt, EOFError):
                                print("\nOperation cancelled.")
                    except ImportError:
                        pass  # Fallback to just logging the error

            return model_response

        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Model execution failed: {e}")
            return ModelResponse(
                success=False,
                content="",
                task_type=request.task_type,
                model="",
                provider="",
                latency=time.time() - start_time,
                error=str(e),
            )

    def _validate_request(self, request: ModelRequest):
        """Validate model request"""
        if not request.prompt:
            raise ValidationError("Prompt cannot be empty", "prompt", request.prompt)

        if request.max_tokens < 1 or request.max_tokens > 7000:
            raise ValidationError("Invalid max_tokens", "max_tokens", request.max_tokens)

        if not (0.0 <= request.temperature <= 2.0):
            raise ValidationError("Invalid temperature", "temperature", request.temperature)

        if request.task_type not in TaskType:
            raise ValidationError("Invalid task type", "task_type", request.task_type)

    def _format_prompt(self, request: ModelRequest) -> str:
        """Format prompt based on task type and context"""
        template = self.prompt_template.get_template(request.task_type)

        format_vars = {
            "instruction": request.prompt,
            "task_description": request.prompt,
        }

        if request.context:
            format_vars.update(request.context)
            if request.task_type == TaskType.COMMAND_PARSING:
                format_vars.setdefault("previous_terminal_actions", "No previous actions")
                format_vars.setdefault("last_command_output", "No previous output")
                format_vars.setdefault("current_directory", "Unknown")

        format_vars.setdefault("os_info", "Unknown OS")

        try:
            formatted_prompt = template.format(**format_vars)
            
            # Add system instructions for better AI behavior
            system_instructions = self._get_system_instructions(request.task_type)
            if system_instructions:
                # For providers that support system instructions, we'll set them in the config
                # For others, we prepend to the prompt
                if hasattr(request, 'parameters') and request.parameters:
                    request.parameters['system_instruction'] = system_instructions
                else:
                    # Prepend system instructions to prompt for providers that don't support separate system messages
                    formatted_prompt = f"{system_instructions}\n\n{formatted_prompt}"
            
            return formatted_prompt
        except KeyError as e:
            self.logger.warning(f"Template variable missing: {e}")
            return request.prompt
        except Exception as e:
            self.logger.error(f"Template formatting error: {e}")
            return request.prompt

    def _get_system_instructions(self, task_type: TaskType) -> str:
        """Get system instructions for better AI behavior"""
        base_instructions = """# VEXIS-CLI AI Agent System Instructions

You are operating as part of the VEXIS-CLI automation system. Your responses directly impact system execution and user experience.

## Behavioral Guidelines
1. **Precision Over Verbosity** - Be exact and concise
2. **Context Awareness** - Always consider previous actions and current state
3. **Error Resilience** - Handle failures gracefully and suggest alternatives
4. **Safety First** - Never suggest destructive commands without clear warnings
5. **User Intent Focus** - Stay focused on accomplishing the user's original goal

## Response Standards
- Use clear, unambiguous language
- Provide specific, actionable outputs
- Avoid conversational filler or unnecessary explanations
- Maintain consistency with previous interactions
- Respect the established workflow and command patterns

## Quality Assurance
- Double-check command syntax before outputting
- Verify file paths and parameters are valid
- Consider edge cases and potential failure points
- Ensure outputs match the expected format exactly"""
        
        if task_type == TaskType.TASK_GENERATION:
            return f"""{base_instructions}

## Task Generation Specific Guidelines
- Break complex tasks into logical, sequential steps
- Each step must be a complete, executable command
- Consider dependencies between steps
- Include necessary setup and cleanup operations
- Validate that the sequence accomplishes the user's goal

## Error Prevention
- Check for required tools or dependencies
- Include error handling commands where appropriate
- Use absolute paths when directory navigation is involved
- Consider permission issues and access rights"""

        elif task_type == TaskType.COMMAND_PARSING:
            return f"""{base_instructions}

## Command Parsing Specific Guidelines
- Analyze the full context before generating commands
- Build upon previous successful actions
- Recognize when tasks are complete vs. need continuation
- Distinguish between retry scenarios vs. new approaches
- Maintain awareness of current working directory and system state

## Decision Making
- Use END only when the original task is fully satisfied
- Use REGENERATE_STEP when previous attempts failed and need retry
- Generate new commands when progress is still needed
- Consider command outputs and error messages in decisions

## Context Integration
- Factor in all previous command outcomes
- Respect the current file system state
- Account for any errors or warnings from prior steps
- Ensure commands are logically consistent with the workflow"""
        
        return base_instructions

    def generate_tasks(self, instruction: str, os_info: str, context: Optional[Dict[str, Any]] = None) -> ModelResponse:
        """Phase 1: Generate task list from instruction and OS environment"""
        enhanced_context = context or {}
        enhanced_context["os_info"] = os_info

        request = ModelRequest(
            task_type=TaskType.TASK_GENERATION,
            prompt=instruction,
            context=enhanced_context,
        )

        return self.run_model(request)

    def parse_command(self, task_description: str, os_info: str, context: Optional[Dict[str, Any]] = None) -> ModelResponse:
        """Phase 2: Parse task description into CLI command"""
        enhanced_context = context or {}
        enhanced_context["os_info"] = os_info

        request = ModelRequest(
            task_type=TaskType.COMMAND_PARSING,
            prompt=task_description,
            context=enhanced_context,
        )

        return self.run_model(request)
    
    def install_missing_sdks(self, providers: Optional[List[str]] = None, interactive: bool = True) -> Dict[str, bool]:
        """Install missing SDKs for specified providers"""
        return self.vision_client.install_missing_sdks(providers, interactive)
    
    def show_sdk_status(self, providers: Optional[List[str]] = None):
        """Show SDK installation status"""
        self.vision_client.show_sdk_status(providers)


def get_model_runner() -> ModelRunner:
    """Get model runner instance (always fresh to use latest settings)"""
    # Always create a new instance to ensure latest settings are loaded
    return ModelRunner()
