#!/usr/bin/env python3
"""
Pinocchio Handler Module
------------------------
Handles interactions with the Pinocchio AI assistant system for Choreo.
This module provides functions to activate different Pinocchio agents and
coordinate their workflow.
"""

import os
import sys
import json
import subprocess
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# Import the SessionLogger
from .session_logger import SessionLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'pinocchio.log'))
    ]
)
logger = logging.getLogger('pinocchio')

# Constants
PROJECT_ROOT = Path(__file__).parent.parent.parent
PINOCCHIO_DIR = PROJECT_ROOT / "tools" / "pinocchio"
CONFIG_FILE = PINOCCHIO_DIR / ".cursor" / "pinocchio_config.json"
KNOWLEDGE_DIR = PINOCCHIO_DIR / ".cursor" / "knowledge"
RULES_DIR = PINOCCHIO_DIR / ".cursor" / "rules"
STATE_FILE = PINOCCHIO_DIR / ".cursor" / "state.json"

# Agent state management
class AgentState:
    """Class to manage agent state and context."""
    
    def __init__(self):
        self.current_agent = None
        self.agent_history = []
        self.context = {}
        self.last_updated = None
        self.load_state()
    
    def load_state(self):
        """Load agent state from file."""
        try:
            if STATE_FILE.exists():
                with open(STATE_FILE, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                    self.current_agent = state_data.get('current_agent')
                    self.agent_history = state_data.get('agent_history', [])
                    self.context = state_data.get('context', {})
                    self.last_updated = state_data.get('last_updated')
                    logger.info(f"Loaded state: current agent is {self.current_agent}")
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            # Initialize with default values
            self.current_agent = None
            self.agent_history = []
            self.context = {}
            self.last_updated = None
    
    def save_state(self):
        """Save agent state to file."""
        try:
            state_data = {
                'current_agent': self.current_agent,
                'agent_history': self.agent_history,
                'context': self.context,
                'last_updated': time.time()
            }
            
            # Create directory if it doesn't exist
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2)
                logger.info(f"Saved state: current agent is {self.current_agent}")
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    def set_current_agent(self, agent_type):
        """Set the current active agent."""
        self.current_agent = agent_type
        self.agent_history.append(agent_type)
        self.last_updated = time.time()
        self.save_state()
    
    def update_context(self, key, value):
        """Update a context value."""
        self.context[key] = value
        self.last_updated = time.time()
        self.save_state()
    
    def get_context(self, key, default=None):
        """Get a context value."""
        return self.context.get(key, default)
    
    def clear_context(self):
        """Clear all context data."""
        self.context = {}
        self.save_state()
    
    def get_agent_history(self, limit=5):
        """Get recent agent history."""
        return self.agent_history[-limit:] if self.agent_history else []

# Initialize agent state
agent_state = AgentState()

# Color printing
def print_color(text, color="green"):
    """Print colored text to the console."""
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m"
    }
    print(f"{colors.get(color, colors['green'])}{text}{colors['reset']}")

def check_pinocchio_available():
    """Check if Pinocchio system is available and properly configured."""
    global CONFIG_FILE
    
    # Check if Pinocchio directory exists
    if not PINOCCHIO_DIR.exists():
        logger.error(f"Pinocchio directory not found at {PINOCCHIO_DIR}")
        print_color(f"❌ Pinocchio directory not found at {PINOCCHIO_DIR}", "red")
        return False
    
    # Check if config file exists
    if not CONFIG_FILE.exists():
        logger.error(f"Configuration file not found at {CONFIG_FILE}")
        print_color(f"❌ Configuration file not found at {CONFIG_FILE}", "red")
        # Try to find any JSON config file
        config_files = list(PINOCCHIO_DIR.glob("**/*.json"))
        if config_files:
            CONFIG_FILE = config_files[0]
            logger.info(f"Found alternative configuration file: {CONFIG_FILE}")
            print_color(f"✅ Found alternative configuration file: {CONFIG_FILE}", "yellow")
        else:
            return False
    
    # Check knowledge and rules directories
    if not KNOWLEDGE_DIR.exists():
        logger.error(f"Knowledge directory not found at {KNOWLEDGE_DIR}")
        print_color(f"❌ Knowledge directory not found at {KNOWLEDGE_DIR}", "red")
        return False
    
    if not RULES_DIR.exists():
        logger.error(f"Rules directory not found at {RULES_DIR}")
        print_color(f"❌ Rules directory not found at {RULES_DIR}", "red")
        return False
    
    logger.info("Pinocchio system is available")
    print_color("✅ Pinocchio system is available", "green")
    return True

def load_pinocchio_config():
    """Load Pinocchio configuration."""
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            config = json.load(f)
            logger.info(f"Configuration loaded: {config.get('name', 'Pinocchio')} v{config.get('version', '1.0')}")
            print_color(f"✅ Configuration loaded: {config.get('name', 'Pinocchio')} v{config.get('version', '1.0')}", "green")
            return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        print_color(f"❌ Error loading configuration: {e}", "red")
        return None

def validate_knowledge_files(config):
    """Validate that all required knowledge files exist."""
    print_color("\n📚 Knowledge Files Check:", "yellow")
    
    knowledge_files = config.get("knowledge_files", {})
    all_files_exist = True
    missing_files = []
    
    for key, filename in knowledge_files.items():
        file_path = KNOWLEDGE_DIR / filename
        if file_path.exists():
            logger.info(f"Knowledge file found: {key}: {filename}")
            print_color(f"  ✅ {key}: {filename}")
        else:
            logger.warning(f"Knowledge file missing: {key}: {filename}")
            print_color(f"  ❌ {key}: {filename} (missing)", "red")
            missing_files.append((key, filename))
            all_files_exist = False
    
    # Update agent state with missing files
    if missing_files:
        agent_state.update_context("missing_knowledge_files", missing_files)
    
    return all_files_exist

def validate_rule_files(config):
    """Validate that all required rule files exist."""
    print_color("\n📜 Rule Files Check:", "yellow")
    
    components = config.get("components", {})
    all_files_exist = True
    missing_files = []
    
    for component, details in components.items():
        rule_file = details.get("rule_file")
        if rule_file:
            file_path = RULES_DIR / rule_file
            if file_path.exists():
                logger.info(f"Rule file found: {component}: {rule_file}")
                print_color(f"  ✅ {component}: {rule_file}")
            else:
                logger.warning(f"Rule file missing: {component}: {rule_file}")
                print_color(f"  ❌ {component}: {rule_file} (missing)", "red")
                missing_files.append((component, rule_file))
                all_files_exist = False
    
    # Update agent state with missing files
    if missing_files:
        agent_state.update_context("missing_rule_files", missing_files)
    
    return all_files_exist

def load_rule_file(agent_type, config):
    """Load the rule file for a specific agent."""
    components = config.get("components", {})
    if agent_type not in components:
        logger.error(f"Agent '{agent_type}' not found in configuration")
        return None
    
    rule_file = components[agent_type].get("rule_file")
    if not rule_file:
        logger.error(f"No rule file specified for agent '{agent_type}'")
        return None
    
    file_path = RULES_DIR / rule_file
    if not file_path.exists():
        logger.error(f"Rule file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            rule_content = f.read()
            logger.info(f"Loaded rule file for {agent_type}: {rule_file}")
            return rule_content
    except Exception as e:
        logger.error(f"Error loading rule file: {e}")
        return None

def load_knowledge_file(knowledge_key, config):
    """Load a knowledge file by key."""
    knowledge_files = config.get("knowledge_files", {})
    if knowledge_key not in knowledge_files:
        logger.error(f"Knowledge key '{knowledge_key}' not found in configuration")
        return None
    
    filename = knowledge_files[knowledge_key]
    file_path = KNOWLEDGE_DIR / filename
    if not file_path.exists():
        logger.error(f"Knowledge file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Handle different file formats
            if filename.endswith('.json'):
                knowledge_content = json.load(f)
            else:
                knowledge_content = f.read()
            logger.info(f"Loaded knowledge file: {knowledge_key}: {filename}")
            return knowledge_content
    except Exception as e:
        logger.error(f"Error loading knowledge file: {e}")
        return None

def get_agent_prompt(agent_type, config, params, context=None):
    """Generate a prompt for a specific agent."""
    # Load agent rule
    rule_content = load_rule_file(agent_type, config)
    if not rule_content:
        return None
    
    # Create agent prompt
    agent_description = config.get("components", {}).get(agent_type, {}).get("description", "")
    
    prompt = f"""# Pinocchio {agent_type.capitalize()} Agent

{agent_description}

## Task
{params}

"""
    
    # Add context if provided
    if context:
        prompt += f"\n## Context\n{json.dumps(context, indent=2)}\n"
    
    # Add rule content
    prompt += f"\n## Agent Rules\n{rule_content}\n"
    
    return prompt

def activate_agent(agent_type, params):
    """Activate a specific Pinocchio agent."""
    if not check_pinocchio_available():
        return "Pinocchio system is not available."
    
    config = load_pinocchio_config()
    if not config:
        return "Failed to load Pinocchio configuration."
    
    # Validate knowledge and rule files
    knowledge_valid = validate_knowledge_files(config)
    rules_valid = validate_rule_files(config)
    
    if not (knowledge_valid and rules_valid):
        return "Pinocchio system is not fully configured. Some required files are missing."
    
    # Find the agent in the configuration
    components = config.get("components", {})
    if agent_type not in components:
        return f"Agent '{agent_type}' not found in Pinocchio configuration."
    
    agent_config = components[agent_type]
    logger.info(f"Activating {agent_type} agent: {agent_config.get('description', '')}")
    print_color(f"\n🤖 Activating {agent_type} agent: {agent_config.get('description', '')}", "cyan")
    
    # Update agent state
    agent_state.set_current_agent(agent_type)
    
    # Get previous context if this is part of a pipeline
    context = agent_state.context
    
    # Generate agent prompt
    prompt = get_agent_prompt(agent_type, config, params, context)
    if not prompt:
        return f"Failed to generate prompt for {agent_type} agent."
    
    # In a real implementation, we would pass this prompt to the LLM
    # For now, we'll just log it and return a simulated response
    logger.info(f"Generated prompt for {agent_type} agent")
    
    # Simulate agent response
    response = f"""[Pinocchio {agent_type.capitalize()} Agent]

I've analyzed your request: "{params}"

Based on my role as the {agent_type} agent and the Choreo knowledge base, here's my response:

"""
    
    if agent_type == "analyzer":
        response += "I've analyzed the code and identified potential issues with syntax and optimization opportunities."
    elif agent_type == "generator":
        response += "I've generated Choreo code based on your requirements, ensuring it follows all syntax rules."
    elif agent_type == "optimizer":
        response += "I've optimized the code for better performance while maintaining correctness and following syntax constraints."
    elif agent_type == "debugger":
        response += "I've identified and fixed issues in the code, ensuring it follows Choreo syntax rules."
    elif agent_type == "coordinator":
        response += "I'm coordinating the workflow between different agents to process your request."
    
    # Update context with agent's response
    agent_state.update_context(f"{agent_type}_response", response)
    agent_state.update_context("last_params", params)
    
    return response

def activate_pinocchio_pipeline(params):
    """Activate the full Pinocchio pipeline with coordinator."""
    if not check_pinocchio_available():
        return "Pinocchio system is not available."
    
    config = load_pinocchio_config()
    if not config:
        return "Failed to load Pinocchio configuration."
    
    logger.info(f"Activating Pinocchio pipeline with parameters: {params}")
    print_color("\n🔄 Activating Pinocchio pipeline", "cyan")
    print_color(f"Parameters: {params}", "blue")
    
    # Find the coordinator in the configuration
    components = config.get("components", {})
    if "coordinator" not in components:
        return "Coordinator agent not found in Pinocchio configuration."
    
    # Clear previous context to start fresh
    agent_state.clear_context()
    
    # First, activate the coordinator
    coordinator_response = activate_agent("coordinator", params)
    
    # Based on the task type, determine the pipeline flow
    # This is a simplified pipeline; in a real implementation, 
    # the coordinator would determine the flow dynamically
    
    # Simple pipeline: analyzer -> generator -> optimizer -> debugger
    print_color("\n📊 Pipeline Stage 1: Analysis", "magenta")
    analyzer_response = activate_agent("analyzer", params)
    
    print_color("\n📝 Pipeline Stage 2: Generation", "magenta")
    generator_response = activate_agent("generator", params)
    
    print_color("\n⚡ Pipeline Stage 3: Optimization", "magenta")
    optimizer_response = activate_agent("optimizer", params)
    
    print_color("\n🔍 Pipeline Stage 4: Debugging", "magenta")
    debugger_response = activate_agent("debugger", params)
    
    # Final coordinator review
    print_color("\n🔄 Pipeline Stage 5: Final Review", "magenta")
    final_response = activate_agent("coordinator", "Review and finalize the pipeline results")
    
    # Compile the final response
    pipeline_result = f"""# Pinocchio Pipeline Results

## Analysis
{analyzer_response}

## Generation
{generator_response}

## Optimization
{optimizer_response}

## Debugging
{debugger_response}

## Final Review
{final_response}
"""
    
    return pipeline_result

def choreo_development_mode(params):
    """Enter Choreo development mode."""
    logger.info(f"Entering Choreo development mode with parameters: {params}")
    print_color("\n🛠️ Entering Choreo development mode", "cyan")
    print_color(f"Parameters: {params}", "blue")
    
    # Load development rules
    dev_rules_path = PROJECT_ROOT / ".cursor" / "rules" / "development_rules.mdc"
    if dev_rules_path.exists():
        try:
            with open(dev_rules_path, 'r', encoding='utf-8') as f:
                dev_rules = f.read()
                print_color("✅ Development rules loaded", "green")
        except Exception as e:
            logger.error(f"Error loading development rules: {e}")
            dev_rules = "Development rules not available."
            print_color("❌ Error loading development rules", "red")
    else:
        dev_rules = "Development rules file not found."
        print_color("❌ Development rules file not found", "red")
    
    # In a real implementation, we would use the development rules
    # to guide the agent's behavior
    
    return f"""# Choreo Development Mode

I'm ready to assist with Choreo compiler development. I'll follow the project's development rules and coding conventions.

## Your Request
{params}

## Development Approach
Based on your request, I'll help you navigate and modify the Choreo codebase. I'll ensure that any changes follow the project's coding style and architecture.
"""

# Specific agent activation functions
def activate_pinocchio_analyze(params):
    """Activate the Pinocchio analyzer agent."""
    logger.info(f"Activating analyzer agent with parameters: {params}")
    return activate_agent("analyzer", params)

def activate_pinocchio_generate(params):
    """Activate the Pinocchio generator agent."""
    logger.info(f"Activating generator agent with parameters: {params}")
    return activate_agent("generator", params)

def activate_pinocchio_convert(params):
    """Activate the Pinocchio converter agent."""
    logger.info(f"Activating converter agent with parameters: {params}")
    return activate_agent("converter", params)

def activate_pinocchio_optimize(params):
    """Activate the Pinocchio optimizer agent."""
    logger.info(f"Activating optimizer agent with parameters: {params}")
    return activate_agent("optimizer", params)

def activate_pinocchio_debug(params):
    """Activate the Pinocchio debugger agent."""
    logger.info(f"Activating debugger agent with parameters: {params}")
    return activate_agent("debugger", params)

# Command handler functions for Cursor integration
def handle_choreo_development_mode(params):
    """Handler for choreo:dev command."""
    return choreo_development_mode(params)

def handle_activate_pinocchio_pipeline(params):
    """Handler for pinocchio:pipeline command."""
    return activate_pinocchio_pipeline(params)

def handle_activate_pinocchio_analyze(params):
    """Handler for pinocchio:analyze command."""
    return activate_pinocchio_analyze(params)

def handle_activate_pinocchio_generate(params):
    """Handler for pinocchio:generate command."""
    return activate_pinocchio_generate(params)

def handle_activate_pinocchio_convert(params):
    """Handler for pinocchio:convert command."""
    return activate_pinocchio_convert(params)

def handle_activate_pinocchio_optimize(params):
    """Handler for pinocchio:optimize command."""
    return activate_pinocchio_optimize(params)

def handle_activate_pinocchio_debug(params):
    """Handler for pinocchio:debug command."""
    return activate_pinocchio_debug(params)

def handle_list_sessions(params):
    """Handler for pinocchio:sessions command."""
    logger.info(f"Listing Pinocchio sessions with parameters: {params}")
    print_color("\n📋 Pinocchio Sessions", "cyan")
    
    # Get recent sessions
    sessions = SessionLogger.list_sessions(limit=10)
    
    if not sessions:
        print_color("No sessions found.", "yellow")
        return "No Pinocchio sessions found."
    
    # Format sessions for display
    result = "# Recent Pinocchio Sessions\n\n"
    
    for i, session in enumerate(sessions):
        session_time = datetime.fromtimestamp(session.get("start_time", 0)).strftime("%Y-%m-%d %H:%M:%S")
        status = "Completed" if session.get("end_time") else "In Progress"
        interactions = session.get("num_interactions", 0)
        
        result += f"## Session {i+1}: {session.get('session_id')}\n"
        result += f"- **Time**: {session_time}\n"
        result += f"- **Status**: {status}\n"
        result += f"- **Interactions**: {interactions}\n"
        result += f"- **Request**: {session.get('user_request', 'N/A')[:100]}...\n\n"
    
    return result

def handle_show_session(params):
    """Handler for pinocchio:session command."""
    logger.info(f"Showing Pinocchio session with parameters: {params}")
    
    # Extract session ID from params
    if not params:
        print_color("❌ No session ID provided.", "red")
        return "Please provide a session ID. Example: pinocchio:session session-20240601-123456-abcdef12"
    
    session_id = params.strip()
    print_color(f"\n📋 Showing Pinocchio Session: {session_id}", "cyan")
    
    # Load the session
    session = SessionLogger.load_session(session_id)
    if not session:
        print_color(f"❌ Session not found: {session_id}", "red")
        return f"Session not found: {session_id}"
    
    # Format session details for display
    start_time = datetime.fromtimestamp(session.start_time).strftime("%Y-%m-%d %H:%M:%S")
    end_time = datetime.fromtimestamp(session.end_time).strftime("%Y-%m-%d %H:%M:%S") if session.end_time else "In Progress"
    
    result = f"# Pinocchio Session: {session.session_id}\n\n"
    result += f"- **Start Time**: {start_time}\n"
    result += f"- **End Time**: {end_time}\n"
    result += f"- **User Request**: {session.user_request}\n\n"
    
    result += "## Agent Interactions\n\n"
    for i, interaction in enumerate(session.agent_interactions):
        agent = interaction.get("agent", "unknown")
        timestamp = datetime.fromtimestamp(interaction.get("timestamp", 0)).strftime("%H:%M:%S")
        input_text = interaction.get("input", "")[:100] + "..." if len(interaction.get("input", "")) > 100 else interaction.get("input", "")
        
        result += f"### Interaction {i+1}: {agent.capitalize()} Agent ({timestamp})\n"
        result += f"- **Input**: {input_text}\n"
        result += f"- **Output**: {interaction.get('output', '')[:200]}...\n\n"
    
    if session.final_result:
        result += "## Final Result\n\n"
        result += f"{json.dumps(session.final_result, indent=2)}\n"
    
    return result

def handle_pinocchio_upgrade(params):
    """Handler for pinocchio:upgrade command."""
    logger.info(f"Upgrading Pinocchio system with parameters: {params}")
    print_color("\n🔄 Pinocchio System Upgrade", "cyan")
    print_color(f"Parameters: {params}", "blue")
    
    # Load design principles
    CURSOR_DIR = Path(__file__).parent.parent
    design_principles_path = CURSOR_DIR / "rules" / "pinocchio_design_principles.mdc"
    if design_principles_path.exists():
        try:
            with open(design_principles_path, 'r', encoding='utf-8') as f:
                design_principles = f.read()
                print_color("✅ Design principles loaded", "green")
        except Exception as e:
            logger.error(f"Error loading design principles: {e}")
            design_principles = "Design principles not available."
            print_color("❌ Error loading design principles", "red")
    else:
        design_principles = "Design principles file not found."
        print_color("❌ Design principles file not found", "red")
    
    # In a real implementation, we would use the design principles
    # to guide the system upgrade
    
    return f"""# Pinocchio System Upgrade

I'm ready to help you develop, update, or evolve the Pinocchio multi-agent system according to its design principles.

## Your Request
{params}

## Development Approach
Based on your request, I'll help you enhance the Pinocchio system while adhering to its core design principles, including:

1. Single-Agent Multi-Role Architecture
2. Clear Role Separation
3. Consistent Interaction Patterns
4. State Preservation
5. Extensibility
6. User Experience

Let me know what specific aspect of the system you'd like to improve or extend.
"""

# Main function for standalone usage
def main():
    """Main function when script is run directly."""
    if not check_pinocchio_available():
        print_color("❌ Pinocchio system is not available.", "red")
        return 1
    
    config = load_pinocchio_config()
    if not config:
        print_color("❌ Failed to load Pinocchio configuration.", "red")
        return 1
    
    knowledge_valid = validate_knowledge_files(config)
    rules_valid = validate_rule_files(config)
    
    # Summary
    print_color("\n📋 Pinocchio System Status:", "blue")
    print_color(f"  Configuration: {'✅ Valid' if config else '❌ Invalid'}")
    print_color(f"  Knowledge Files: {'✅ All present' if knowledge_valid else '❌ Some missing'}")
    print_color(f"  Rule Files: {'✅ All present' if rules_valid else '❌ Some missing'}")
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        params = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else ""
        
        if command == "analyze":
            print(activate_pinocchio_analyze(params))
        elif command == "generate":
            print(activate_pinocchio_generate(params))
        elif command == "convert":
            print(activate_pinocchio_convert(params))
        elif command == "optimize":
            print(activate_pinocchio_optimize(params))
        elif command == "debug":
            print(activate_pinocchio_debug(params))
        elif command == "pipeline":
            print(activate_pinocchio_pipeline(params))
        elif command == "dev":
            print(choreo_development_mode(params))
        else:
            print_color(f"❌ Unknown command: {command}", "red")
            print_color("Available commands: analyze, generate, convert, optimize, debug, pipeline, dev", "yellow")
            return 1
    else:
        print_color("❌ No command specified.", "red")
        print_color("Available commands: analyze, generate, convert, optimize, debug, pipeline, dev", "yellow")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
