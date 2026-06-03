# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

AIOps Agent is a conversational root cause analysis system for microservice anomalies. It uses a four-agent architecture with a Plan-and-Execute supervisor: Supervisor (planner → executor → finalize), Detection Agent (3-Sigma anomaly detection), Diagnose Agent (IAF-RCL/KE-FPC root cause analysis), and Report Agent (NL report generation).

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

### Three-Agent Architecture (Plan-and-Execute)

```
Main Graph: START → supervisor → END

Supervisor Agent (Plan-and-Execute):
  START → planner(LLM生成计划)
              │
              ├── empty plan → direct_reply(reply字段) → END
              │
              └── has steps → executor(调度执行) ──→ executor(循环) ──→ finalize → END
  Registry: detection adapter, diagnose adapter, report adapter
  Memory: step_results[step_id] → dict

Detection Agent (ReAct, 独立):
  model → [tool_calls] → ToolNode → extract_results → model → ... → final(结构化异常摘要) → END
  Tools: csv_reader_tool, three_sigma_tool
  输出: { anomalies, summary, abnormal_kpi, inject_time }

Diagnose Agent (ReAct, 独立):
  model → [tool_calls] → ToolNode → extract_results → model → ... → final(结构化算法结果) → END
  Tools: csv_reader_tool, rcd_tool, pc_tool, graph_visualization_tool
  输出: { root_causes, edges, graph_visualizations, rcd_result, pc_result }

Report Agent (单节点):
  START → generate_report_node → END
  输入: detection/diagnose 结构化数据 → LLM → 自然语言报告
  输出: { final_response (NL report) }
```

### Key Components

**Main Graph** (`app/agents/main_graph.py`)
- Simple wrapper: START → supervisor_node → END
- Converts MainState to PlanExecuteState and back
- Propagates sub-agent results from step_results and AI messages

**Supervisor Agent** (`app/agents/supervisor_plan_execute.py`) — Plan-and-Execute
- **planner node**: Analyzes user request via LLM (temperature=0, no bind_tools), generates `List[PlanStep]`
- **executor node**: Iterates through plan steps, invokes subgraphs via `SubAgentAdapter` interface (detection, diagnose, report)
- **finalize node**: Extracts report result from step_results, generates HTML report, attaches topology visualization
- **direct_reply node**: For non-agent queries (uses planner's `reply` field from JSON output)
- **direct_reply node**: For simple conversational responses (no sub-agent needed)
- Uses `step_results: Dict[int, Dict]` as generic memory between steps
- Routing: `route_after_planner` (empty plan → direct_reply), `route_after_executor` (more steps → loop)

**Subgraph Registry** (`app/agents/subgraph_registry.py`)
- `SubAgentAdapter` interface: `build_input(step_input, step_results)` → `extract_result(subgraph_output)`
- `DetectionAdapter`: Builds detection subgraph input, extracts anomaly results
- `DiagnoseAdapter`: Builds diagnose subgraph input with optional detection summary from prior step
- `get_adapter(agent_name)` and `get_subgraph(agent_name)` for lookup

**Detection Agent** (`app/agents/detection_agent.py`) — ReAct
- Loads CSV data and runs 3-Sigma anomaly detection
- Returns structured results: anomalies, abnormal_kpi, inject_time, csv_file_path
- Final node builds natural-language summary (no LLM call)

**Diagnose Agent** (`app/agents/diagnose_agent.py`) — ReAct
- Runs IAF-RCL and KE-FPC algorithms, generates graph visualizations
- Returns **structured data** (not human-readable report) — report generation is the supervisor's job
- Final node packages rcd_result, pc_result, graph_visualizations as JSON in `integrated_result`
- Tools: csv_reader_tool, rcd_tool, pc_tool, graph_visualization_tool

### Generic ReAct Nodes (`app/agents/nodes/react_nodes.py`)

Reusable node factories for all ReAct agents:
- `create_model_node(llm, system_prompt)` — Invokes LLM with bound tools
- `extract_results_node(state)` — Parses ToolMessages, updates state fields
- `create_final_response_node(refine_prompt_path, llm)` — LLM-based result synthesis (used by standalone diagnose agents)
- `route_after_model(state)` → "tools" or "final"
- `route_after_extract(state)` → "model", "interrupt", or "final"
- `compress_messages(messages)` — Reduces token usage for long ToolMessages

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
- Temperature defaults: planner=0, detection=0, diagnose=0, synthesis=0.3

**Agent Config** (`app/config/agents.yaml`)
- YAML configuration for all three agents
- Specifies state_schema, system_prompt, tools, model settings, max_iterations
- Supervisor uses `type: plan_execute` with `sub_agents: [detection, diagnose]`

### Causal Discovery Tool (`app/tools/CausalDiscovery/`)

Separate causal discovery system with its own configuration:
- See `app/tools/CausalDiscovery/AGENTS.md` for detailed documentation
- Contains PC/PCMCI algorithms, preprocessing, knowledge management
- Runs independently via `python app/tools/CausalDiscovery/run.py`

## State Management

**MainState** - Main graph state
- `user_input`, `messages`, `action`
- Parameters: `csv_file_path`, `inject_time`, `abnormal_kpi`
- Results: `diagnose_result`

**PlanExecuteState** - Plan-Execute supervisor state
- Messages with plan execution history
- Plan: `plan: List[PlanStep]`, `current_step_index`, `plan_reasoning`
- Results: `step_results: Dict[int, Dict]` (generic memory, keyed by step_id)
- Output: `final_response`, `continue_conversation`

**PlanStep** - Single plan step
- `step_id`, `name`, `agent` ("detection"|"diagnose"|"direct_reply")
- `input: Dict[str, Any]`, `status`, `error`

**DetectionAgentState** - Detection agent state
- Messages, task_description, tool results
- `three_sigma_result`, `abnormal_kpi`, `inject_time`
- Output: `final_response`

**ReactAgentState** - Generic ReAct state (used by diagnose agent)
- Messages, tool_results, tool_errors
- Legacy fields: `rcd_result`, `pc_result`, `csv_file_path`, `inject_time`, `abnormal_kpi`
- Output: `final_response`, `integrated_result`

## Prompt Template System (`app/utils/prompt_template.py`)

Centralizes result formatting logic:
- `render_template(template_path, variables)` — Load .md template and replace `{{VAR}}` placeholders
- `format_detection_summary(detection_result)` — Format detection results for template injection
- `format_diagnose_summary(diagnose_result, ...)` — Format diagnose results for template injection
- `_format_inject_time(inject_time)` — Format Unix timestamp as human-readable string

## Prompt Files

- `app/prompts/supervisor_planner.md` — Supervisor planner prompt (sub-agent descriptions, decision rules, JSON output format, reply field for direct_reply)
- `app/prompts/supervisor_synthesis.md` — Supervisor report synthesis prompt template (legacy, not used by current finalize node)
- `app/prompts/detection_system.md` — Detection agent system prompt
- `app/prompts/diagnose_system.md` — Diagnose agent system prompt (no detailed tool params)
- `app/prompts/diagnose_refine.md` — Legacy refine template (output format spec migrated to supervisor_synthesis.md)

## Important Implementation Details

### Plan-Execute Flow
1. Planner analyzes user request, generates `List[PlanStep]` (JSON output from LLM)
2. If plan is empty → direct_reply node (uses planner's `reply` field as response)
3. Executor iterates through plan steps (detection → diagnose → report):
   - `adapter.build_input(step.input, step_results)` → subgraph input state
   - `subgraph.invoke(input)` → subgraph output
   - `adapter.extract_result(output)` → standardized result
   - Result stored in `step_results[step_id]`
4. Finalize node extracts report step result, generates HTML report, attaches topology visualization
5. Topology visualization attached via `build_final_report_message()`

### Step-to-Step Result Passing
- Each step's result is stored in `step_results[step_id]`
- Later steps can reference prior results via `step.input["from_step"]`
- The DiagnoseAdapter automatically enriches task_description with detection summary when `from_step` is set
- Shared parameters (csv_file_path, inject_time, abnormal_kpi) flow through step_results

### ReAct Loop (Sub-agents)
Detection and Diagnose agents use the same generic ReAct pattern:
1. `model_node` invokes LLM with bound tools → returns AIMessage with `tool_calls`
2. Route to `tools` if tool_calls present, or to final if done
3. `ToolNode` executes the tool and produces ToolMessages
4. `extract_results_node` parses results, updates state, detects interrupts
5. Loop back to `model_node` (up to `max_iterations` iterations)

### CSV Data Caching
CSV data is cached globally to avoid re-reading files within a single analysis session. The cache key is the resolved file path.

### Tool Parameter Validation
Tools validate their own parameters BEFORE execution. If required parameters are missing, they return a JSON response with `error: "missing_parameter:xxx"`. Pydantic `Field(...)` enforces required parameters at the tool schema level.

### Error Handling
- Tool errors are tracked in `state["tool_errors"]` list
- Executor marks step as "failed" but continues to next step
- Final report includes both successful results and errors

### Message Compression
Large tool messages (>2000 chars) are compressed in `model_node` before passing to LLM, keeping essential keys and truncating large lists to reduce token usage.

### Path Resolution
`app/utils/path_resolver.py` handles data path resolution with support for relative paths like `data/file.csv` and `./data/file.csv`.

## Adding New Sub-Agents

1. Create state TypedDict in `app/models/`
2. Create SubAgentAdapter in `app/agents/subgraph_registry.py` (define `build_input` / `extract_result`)
3. Register adapter in `REGISTRY` dict and add `get_subgraph()` case
4. Create system prompt `.md` in `app/prompts/`
5. Add agent config in `app/config/agents.yaml`
6. Update `supervisor_planner.md` to describe the new agent

## Testing

Tests are located in `tests/`:
- `test_csv_tool.py` - CSV reading functionality
- `test_models.py` - LLM model factory
- `test_main.py` - Main application entry
- `test_integration.py` - End-to-end workflow tests + adapter tests
- `test_new_workflow.py` - Plan-Execute routing, graph structure, prompt formatting
- `test_detection_agent.py` - Detection agent + adapter contract tests

## Troubleshooting

**IAF-RCL/KE-FPC algorithms not working**
- Check `app/tools/rcd/` and ensure dependencies are installed
- Verify causal-learn is available: `pip show causal-learn`

**LangGraph routing issues**
- Enable debug: `LANGGRAPH_DEBUG=true` in `.env`
- Check state values using `langgraph dev`
- Verify graph structure: each agent's `get_graph()` should contain expected nodes

**Supervisor planner not generating valid plans**
- Check that LLM returns valid JSON (test with `langgraph dev`)
- Verify `supervisor_planner.md` prompt is loaded correctly
- Check `_parse_plan_json()` handles markdown-wrapped JSON

**Executor not invoking subgraphs**
- Verify `get_subgraph(agent_name)` returns the correct LazyGraph
- Check that `adapter.build_input()` produces valid subgraph state
- Ensure subgraph state schema matches adapter output

**ReAct loop issues (sub-agents)**
- Check that LLM returns proper tool_calls (not all providers support function calling)
- Verify tools are correctly bound: `llm.bind_tools(tools)` in `model_node`
- If LLM doesn't call tools, check system prompt for clear tool instructions

**LLM not responding**
- Verify API key in `.env`
- Check base URL and model name
- Reduce `LLM_MAX_TOKENS` if hitting limits
