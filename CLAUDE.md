# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AIOps Agent is a conversational root cause analysis system for microservice anomalies. It uses a Supervisor-Agent architecture with LangGraph workflows to analyze metrics data using IAF-RCL and KE-FPC causal discovery algorithms (wrapped as `rcd_tool` and `pc_tool`).

## Running the Application

```bash
# Install dependencies
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env with your DEEPSEEK_API_KEY

# Interactive mode (recommended)
python -m app.main

# Single request mode
python -m app.main --request "分析 ./data/sample_metrics.csv 文件的根因"

# LangGraph Studio (visualization/debug)
langgraph dev
# Visit http://localhost:8123
```

## Common Commands

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_csv_tool.py

# Check Python dependencies
pip list | grep -E "langgraph|langchain|deepseek"
```

## Architecture

### Three-Level Graph Structure

```
Main Graph (main_graph.py)
  ├── Supervisor Agent (supervisor_agent.py) - Intent recognition and routing
  └── Diagnose Agent (diagnose_agent.py) - ReAct loop for tool execution
```

### Key Components

**Main Graph** (`app/agents/main_graph.py`)
- Entry point that coordinates supervisor and diagnose subgraphs
- Handles state conversion between graphs
- Routes based on `action`: "call_diagnose", "have_diagnose_result", "interrupted", "respond"
- Catches `GraphInterrupt` from diagnose agent for human-in-the-loop
- Both subgraphs are expandable in LangGraph Studio

**Supervisor Agent** (`app/agents/supervisor_agent.py`)
- Simple intent recognition using LLM (prompt loaded from `app/prompts/supervisor_system.md`)
- Routes to diagnose_agent for analysis tasks or responds directly
- NO parameter extraction - diagnose_agent handles all parameters
- Handles "interrupted" action: relays the question from diagnose agent to the user
- Keywords triggering diagnosis (fallback when LLM fails): "异常", "故障", "根因", "诊断", "分析", "rcd", "pc", "因果", "传播"

**Diagnose Agent** (`app/agents/diagnose_agent.py`) — ReAct Loop

```
START → model (LLM) ──有 tool_calls──→ tools (ToolNode) → extract_results → model
              │                                                             ↑
              └──无 tool_calls──→ final (报告合成) → END                    │
              └──iteration >= max──→ final ─────────────────────────────────┘
```

- **model_node**: Loads `diagnose_system.md`, binds `diagnose_tools` to LLM, invokes LLM
- **tools**: LangGraph built-in `ToolNode(diagnose_tools)` executes tool calls
- **extract_results_node**: Parses ToolMessages, updates state (csv_data, rcd_result, pc_result), tracks errors, triggers `interrupt()` for ask_user
- **final_response_node**: LLM synthesizes results using `diagnose_refine.md`
- Tools available:
  - `read_csv` - Load metrics data (cached after first load)
  - `rcd_algorithm` - Fast root cause inference (requires inject_time)
  - `pc_algorithm` - Causal discovery (requires CSV data)
  - `ask_user` - Request missing parameters via LangGraph interrupt()
- CSV data is cached globally to avoid re-reading files
- Tool errors are tracked but don't stop execution
- Message compression (>2000 chars) reduces token usage

### Algorithm Wrappers

**IAF-RCL Wrapper** (`app/tools/rcd_wrapper.py`)
- Wraps the IAF-RCL implementation from `app/tools/rcd/`
- Parameters: inject_time (required), gamma (default: 5)
- Returns ranked list of root cause metrics

**KE-FPC Wrapper** (`app/tools/pc_wrapper.py`)
- Wraps KE-FPC causal discovery (causal-learn PC backend) with correlation fallback
- Parameters: alpha (default: 0.05)
- Returns root causes and causal graph edges
- Falls back to correlation-based analysis if causal-learn unavailable

### Configuration

**Environment Variables** (`.env`)
```
LLM_PROVIDER=deepseek          # or openai, anthropic
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
LLM_TEMPERATURE=0.7
LANGGRAPH_DEBUG=true
```

**Model Factory** (`app/config/model_config.py`)
- `get_deepseek_llm(**kwargs)` - Get configured DeepSeek LLM instance
- `get_llm(provider, **kwargs)` - Generic LLM getter
- Temperature defaults: intent=0.1, analysis=0, refinement=0.3

### Causal Discovery Tool (`app/tools/CausalDiscovery/`)

Separate causal discovery system with its own configuration:
- See `app/tools/CausalDiscovery/CLAUDE.md` for detailed documentation
- Contains PC/PCMCI algorithms, preprocessing, knowledge management
- Runs independently via `python app/tools/CausalDiscovery/run.py`

## State Management

**MainState** - Coordinates between subgraphs
- `user_input`, `messages`, `action`
- Parameters: `csv_file_path`, `inject_time`, `gamma`, `alpha`, `dataset_type`
- Results: `diagnose_result`, `interrupt_data`

**SupervisorAgentState** - Intent recognition
- Extends MainState fields
- `response_message`, `continue_conversation`, `interrupt_data`

**DiagnoseAgentState** - ReAct loop state
- Messages with tool call history (HumanMessage, AIMessage, ToolMessage)
- Task description and parameters
- Results: `csv_data`, `rcd_result`, `pc_result`, `integrated_result`
- Control: `iteration_count`, `max_iterations`, `tool_errors`
- Interrupt: `interrupted` (bool), `interrupt_data` (dict) — set when ask_user triggers

## Important Implementation Details

### ReAct Loop
The diagnose agent uses LangGraph's built-in `ToolNode` for tool execution. The loop is:
1. `model_node` invokes LLM with bound tools → returns AIMessage with `tool_calls`
2. `route_model` routes to `tools` if tool_calls present, or `final` if done
3. `ToolNode` executes the tool and produces ToolMessages
4. `extract_results_node` parses results, updates state, detects interrupts
5. Loop back to `model_node` (up to `max_iterations` iterations)

### CSV Data Caching
CSV data is cached globally in `diagnose_agent.py` to avoid re-reading files within a single analysis session. The cache key is the resolved file path.

### Tool Parameter Validation
Tools validate their own parameters BEFORE execution. If required parameters are missing, they return a JSON response with `error: "missing_parameter:xxx"` and `user_prompt` for asking the user. Pydantic `Field(...)` enforces required parameters at the tool schema level.

### Human-in-the-Loop
When `ask_user` tool is called, `extract_results_node` detects `requires_user_input` and calls LangGraph's `interrupt()`. The `main_graph.py` catches `GraphInterrupt` and sets `action="interrupted"`, routing back to the supervisor to relay the question to the user.

### Error Handling
- Tool errors are tracked in `state["tool_errors"]` list
- Errors don't stop execution; LLM can retry, skip, or call ask_user
- Final report includes both successful results and errors

### Message Compression
Large tool messages (>1000 chars) are compressed in `model_node` before passing to LLM, keeping essential keys and truncating large lists to reduce token usage.

### Prompt Files
- `app/prompts/diagnose_system.md` — ReAct tool-calling protocol for the diagnose agent
- `app/prompts/diagnose_refine.md` — Template for synthesizing final reports from tool results
- `app/prompts/supervisor_system.md` — Intent recognition instructions for the supervisor

### Path Resolution
`app/utils/path_resolver.py` handles data path resolution with support for relative paths like `data/file.csv` and `./data/file.csv`.

## Adding New Tools

1. Create tool function with Pydantic input schema
2. Add to `diagnose_tools` list in `diagnose_agent.py`
3. Update system prompt in `app/prompts/diagnose_system.md`
4. Handle result extraction in `extract_results_node`

Example:
```python
class MyToolInput(BaseModel):
    param: str = Field(..., description="Description")

def my_tool_func(param: str) -> str:
    result = do_something(param)
    return json.dumps({"success": True, "data": result})

# Add to diagnose_tools
StructuredTool.from_function(
    func=my_tool_func,
    name="my_tool",
    description="Tool description",
    args_schema=MyToolInput
)
```

## Testing

Tests are located in `tests/`:
- `test_csv_tool.py` - CSV reading functionality
- `test_models.py` - LLM model factory
- `test_main.py` - Main application entry
- `test_integration.py` - End-to-end workflow tests
- `test_new_workflow.py` - Supervisor-agent workflow

## Troubleshooting

**IAF-RCL/KE-FPC algorithms not working**
- Check `app/tools/rcd/` and ensure dependencies are installed
- Verify causal-learn is available: `pip show causal-learn`

**LangGraph routing issues**
- Enable debug: `LANGGRAPH_DEBUG=true` in `.env`
- Check action values in state using `langgraph dev`
- Verify graph structure: `diagnose_agent.get_graph()` should contain nodes `["model", "tools", "extract_results", "final"]`

**ReAct loop issues**
- Check that LLM returns proper tool_calls (not all providers support function calling)
- Verify `diagnose_tools` are correctly bound: `llm.bind_tools(diagnose_tools)` in `model_node`
- If LLM doesn't call tools, check `diagnose_system.md` prompt for clear tool instructions

**Human-in-the-loop not working**
- `ask_user` tool must set `requires_user_input: true` in response
- `extract_results_node` must call `interrupt()` for `main_graph.py` to catch `GraphInterrupt`
- Ensure `main_graph.py` has the `except GraphInterrupt` handler in `diagnose_node`

**LLM not responding**
- Verify API key in `.env`
- Check base URL and model name
- Reduce `LLM_MAX_TOKENS` if hitting limits
