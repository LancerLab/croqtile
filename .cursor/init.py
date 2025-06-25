#!/usr/bin/env python3
"""
Cursor Initialization Script for Choreo
--------------------------------------
This script initializes the Cursor environment for the Choreo project,
including setting up the Pinocchio AI assistant system.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'init.log'))
    ]
)
logger = logging.getLogger('cursor_init')

# Constants
CURSOR_DIR = Path(__file__).parent
PROJECT_ROOT = CURSOR_DIR.parent
PINOCCHIO_DIR = PROJECT_ROOT / "tools" / "pinocchio"
RULES_DIR = CURSOR_DIR / "rules"
HANDLERS_DIR = CURSOR_DIR / "handlers"

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

def check_directories():
    """Check if required directories exist and create them if necessary."""
    directories = [
        RULES_DIR,
        HANDLERS_DIR
    ]
    
    for directory in directories:
        if not directory.exists():
            try:
                directory.mkdir(parents=True)
                logger.info(f"Created directory: {directory}")
                print_color(f"✅ Created directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                print_color(f"❌ Failed to create directory {directory}: {e}", "red")
                return False
    
    return True

def check_pinocchio():
    """Check if Pinocchio system is available."""
    if not PINOCCHIO_DIR.exists():
        logger.warning(f"Pinocchio directory not found at {PINOCCHIO_DIR}")
        print_color(f"⚠️ Pinocchio directory not found at {PINOCCHIO_DIR}", "yellow")
        return False
    
    # Check for key Pinocchio files
    config_file = PINOCCHIO_DIR / ".cursor" / "pinocchio_config.json"
    if not config_file.exists():
        logger.warning(f"Pinocchio configuration file not found at {config_file}")
        print_color(f"⚠️ Pinocchio configuration file not found", "yellow")
        return False
    
    logger.info("Pinocchio system found")
    print_color("✅ Pinocchio system found")
    return True

def sync_knowledge_files():
    """Sync knowledge files from Pinocchio to Cursor rules."""
    if not check_pinocchio():
        return False
    
    try:
        # Load Pinocchio config
        config_file = PINOCCHIO_DIR / ".cursor" / "pinocchio_config.json"
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Get knowledge files
        knowledge_files = config.get("knowledge_files", {})
        knowledge_dir = PINOCCHIO_DIR / ".cursor" / "knowledge"
        
        # Check if knowledge directory exists
        if not knowledge_dir.exists():
            logger.warning(f"Pinocchio knowledge directory not found at {knowledge_dir}")
            print_color(f"⚠️ Pinocchio knowledge directory not found", "yellow")
            return False
        
        # Create MDC summaries for key knowledge files
        knowledge_keys = [
            "choreo_syntax_rules",
            "choreo_common_errors",
            "advanced_patterns",
            "hpc_hardware_performance_rules"
        ]
        
        for key in knowledge_keys:
            if key in knowledge_files:
                file_name = knowledge_files[key]
                source_file = knowledge_dir / file_name
                target_file = RULES_DIR / f"{key.lower()}.mdc"
                
                if source_file.exists():
                    # Create a summary MDC file that references the original JSON
                    try:
                        with open(target_file, 'w', encoding='utf-8') as f:
                            f.write(f"""# {key.replace('_', ' ').title()}

This file provides a summary of the {key.replace('_', ' ')} for the Choreo project.
The full knowledge base is available in JSON format at:

`{source_file}`

## Usage

This knowledge can be accessed by the Pinocchio agents through the knowledge base system.
The Cursor agent can also access this information directly.

## Key Points

- This is a summary of the {key.replace('_', ' ')}
- For detailed information, refer to the original JSON file
- The Pinocchio system uses this knowledge to guide its behavior

## Integration

This file is part of the Cursor-Pinocchio integration system, which allows
the Cursor agent to leverage the specialized knowledge and capabilities of
the Pinocchio multi-agent system.
""")
                        logger.info(f"Created summary for {key} at {target_file}")
                        print_color(f"✅ Created summary for {key}")
                    except Exception as e:
                        logger.error(f"Failed to create summary for {key}: {e}")
                        print_color(f"❌ Failed to create summary for {key}: {e}", "red")
                else:
                    logger.warning(f"Knowledge file not found: {source_file}")
                    print_color(f"⚠️ Knowledge file not found: {source_file}", "yellow")
        
        return True
    except Exception as e:
        logger.error(f"Error syncing knowledge files: {e}")
        print_color(f"❌ Error syncing knowledge files: {e}", "red")
        return False

def check_agent_config():
    """Check if agent_config.json exists and is valid."""
    config_file = CURSOR_DIR / "agent_config.json"
    if not config_file.exists():
        logger.warning("agent_config.json not found")
        print_color("⚠️ agent_config.json not found", "yellow")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
            # Check for required fields
            if "project_rules" not in config:
                logger.warning("project_rules not found in agent_config.json")
                print_color("⚠️ project_rules not found in agent_config.json", "yellow")
            
            if "custom_commands" not in config:
                logger.warning("custom_commands not found in agent_config.json")
                print_color("⚠️ custom_commands not found in agent_config.json", "yellow")
            
            logger.info("agent_config.json is valid")
            print_color("✅ agent_config.json is valid")
            return True
    except Exception as e:
        logger.error(f"Error checking agent_config.json: {e}")
        print_color(f"❌ Error checking agent_config.json: {e}", "red")
        return False

def main():
    """Main initialization function."""
    print_color("\n🚀 Initializing Cursor for Choreo Project", "cyan")
    
    # Check directories
    if not check_directories():
        print_color("❌ Failed to initialize directories", "red")
        return 1
    
    # Check Pinocchio
    pinocchio_available = check_pinocchio()
    if pinocchio_available:
        print_color("✅ Pinocchio system is available", "green")
        
        # Sync knowledge files
        if sync_knowledge_files():
            print_color("✅ Knowledge files synced", "green")
        else:
            print_color("⚠️ Some knowledge files could not be synced", "yellow")
    else:
        print_color("⚠️ Pinocchio system is not available - some features will be limited", "yellow")
    
    # Check agent config
    if check_agent_config():
        print_color("✅ Agent configuration is valid", "green")
    else:
        print_color("⚠️ Agent configuration may have issues", "yellow")
    
    print_color("\n✅ Cursor initialization complete", "cyan")
    print_color("You can now use the Cursor agent with the Choreo project", "cyan")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 