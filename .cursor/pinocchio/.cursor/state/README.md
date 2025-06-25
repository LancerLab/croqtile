# Pinocchio State Directory

This directory contains state information for the Pinocchio multi-agent system.

## Files

- `state.json` - Current state of the Pinocchio system, including:
  - Current active agent
  - Agent history
  - Context data shared between agents
  - Last update timestamp

- `sessions/` - Directory containing session history files
  - Each file represents a complete interaction session
  - Sessions are named with timestamp and session ID

## State Management

The Pinocchio system uses this directory to maintain state between agent activations. This allows for:

1. **Context Continuity** - Agents can access information from previous agents
2. **Session History** - Complete interaction logs are preserved
3. **Pipeline Tracking** - The system can track which agents have been activated in a pipeline

## Implementation Details

The state management is implemented in `.cursor/handlers/pinocchio_handler.py` through the `AgentState` class, which provides:

- Loading and saving state
- Context updates
- Agent history tracking
- State clearing

## Usage

State files should not be modified manually. They are managed automatically by the Pinocchio system. 