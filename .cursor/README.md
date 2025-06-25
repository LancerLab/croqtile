# Cursor Configuration for Choreo

This directory contains the configuration files for Cursor AI agents working with the Choreo project.

## Directory Structure

- `agent_config.json` - Main configuration file for Cursor agents
- `rules/` - Contains development rules and knowledge summaries for the Choreo project
- `handlers/` - Contains command handler implementations
- `init.py` - Initialization script for the Cursor environment

## Custom Commands

The following custom commands are available:

- `choreo:dev` - Develop the Choreo compiler project directly
- `pinocchio:pipeline` - Activate the Pinocchio system pipeline
- `pinocchio:analyze` - Analyze Choreo code for issues
- `pinocchio:generate` - Generate Choreo code based on description
- `pinocchio:convert` - Convert TopSCC code to Choreo
- `pinocchio:optimize` - Optimize existing Choreo code
- `pinocchio:debug` - Help debug Choreo code

## Pinocchio Multi-Agent System

This configuration integrates with the Pinocchio AI assistant system located in `tools/pinocchio`. The integration is handled by the `handlers/pinocchio_handler.py` module, which provides access to the various Pinocchio agents.

### Agent Roles

The Pinocchio system implements a simulated multi-agent architecture with the following specialized agents:

1. **Coordinator Agent** (`coordinator`)
   - Manages the overall workflow
   - Delegates tasks to specialized agents
   - Synthesizes results into coherent solutions
   - Provides the final response to the user

2. **Analyzer Agent** (`analyzer`)
   - Analyzes and understands existing Choreo code
   - Identifies syntax issues and optimization opportunities
   - Validates code against Choreo's syntax rules
   - Provides detailed analysis reports

3. **Generator Agent** (`generator`)
   - Generates Choreo code from algorithmic descriptions
   - Translates code from other languages to Choreo
   - Ensures generated code follows syntax rules
   - Creates readable and maintainable code

4. **Optimizer Agent** (`optimizer`)
   - Analyzes code for optimization opportunities
   - Applies performance improvements while maintaining correctness
   - Tailors optimizations to specific hardware targets
   - Balances readability and performance

5. **Debugger Agent** (`debugger`)
   - Identifies and fixes syntax errors
   - Debugs logical errors in Choreo programs
   - Diagnoses performance issues
   - Proposes and implements fixes

### How the Multi-Agent System Works

The Pinocchio system uses a single LLM to simulate multiple agents by:

1. **Role Switching** - The system switches between different agent roles by loading specific rule files that define each agent's behavior, knowledge, and constraints.

2. **Context Preservation** - State and context are maintained between agent switches, allowing information to be passed from one agent to another.

3. **Pipeline Workflow** - The `pinocchio:pipeline` command activates a complete workflow that passes the user's request through multiple agents in sequence:
   - Coordinator (initial analysis)
   - Analyzer (code analysis)
   - Generator (code generation)
   - Optimizer (performance optimization)
   - Debugger (error checking)
   - Coordinator (final review)

4. **Knowledge Integration** - All agents access a shared knowledge base containing Choreo syntax rules, common errors, advanced patterns, and hardware performance guidelines.

### Implementation Details

- Agent rules are stored as MDC files in `tools/pinocchio/.cursor/rules/`
- Knowledge files are stored as JSON in `tools/pinocchio/.cursor/knowledge/`
- Agent state is maintained in `tools/pinocchio/.cursor/state.json`
- Each agent has a specific prompt template that includes its role definition and relevant knowledge

## Development Rules

The development rules for the Choreo project are defined in `rules/development_rules.mdc`. These rules guide the Cursor agents in understanding the project structure, coding conventions, and best practices.

## Knowledge Summaries

The `rules/` directory contains MDC summaries of key knowledge files:

- `choreo_syntax.mdc` - Summary of Choreo syntax rules
- `common_errors.mdc` - Common errors in Choreo code
- `advanced_patterns.mdc` - Advanced coding patterns for Choreo
- `dma_patterns.mdc` - Patterns for DMA operations in Choreo

These summaries reference the full JSON knowledge files in the Pinocchio system.

## Usage

To use the custom commands, simply type them in the Cursor chat interface. For example:

```
choreo:dev I want to add a new feature to the typeinfer.cpp file
```

or

```
pinocchio:analyze 
```co
__co__ void foo(u32 x) {
  u32 y = x + 1;
  return y;
}
```

## Initialization

The `init.py` script initializes the Cursor environment by:

1. Creating necessary directories
2. Checking for the Pinocchio system
3. Syncing knowledge files from Pinocchio to Cursor
4. Validating the agent configuration

Run it with:

```
python .cursor/init.py
``` 