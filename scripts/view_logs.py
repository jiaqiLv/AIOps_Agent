"""View LLM conversation logs

This script provides utilities to view and analyze LLM conversation logs.
"""

import json
import sys
from pathlib import Path
from typing import Optional

LOG_DIR = Path("log/llm_conversations")


def print_log_entry(filepath: str, verbose: bool = False) -> None:
    """Print a single log entry in a readable format."""
    with open(filepath, 'r', encoding='utf-8') as f:
        entry = json.load(f)

    print(f"\n{'='*80}")
    print(f"File: {Path(filepath).name}")
    print(f"{'='*80}")
    print(f"Timestamp: {entry.get('timestamp')}")
    print(f"Agent: {entry.get('agent')}")
    print(f"Iteration: {entry.get('iteration')}")

    if entry.get('metadata'):
        print(f"Metadata: {json.dumps(entry.get('metadata'), ensure_ascii=False)}")

    # Input summary
    input_data = entry.get('input', {})
    print(f"\n[INPUT]")
    print(f"  Messages: {input_data.get('message_count', 0)}")
    print(f"  Approx tokens: {input_data.get('approx_tokens', 0):,}")

    if verbose:
        print(f"\n  Messages detail:")
        for i, msg in enumerate(input_data.get('messages', [])[:5]):  # Show first 5
            msg_type = msg.get('type', 'Unknown')
            content = msg.get('content', '')
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"    [{i}] {msg_type}: {content}")

        if len(input_data.get('messages', [])) > 5:
            print(f"    ... and {len(input_data.get('messages', [])) - 5} more messages")

    # Response
    if entry.get('response'):
        resp = entry['response']
        print(f"\n[RESPONSE]")
        print(f"  Type: {resp.get('type')}")

        if resp.get('tool_calls_count'):
            print(f"  Tool calls: {resp.get('tool_calls_count')}")
            if verbose:
                for tc in resp.get('tool_calls', []):
                    print(f"    - {tc.get('name')}")

        if resp.get('content'):
            content = resp['content']
            if len(content) > 500:
                preview = content[:500] + f"...[truncated, total {len(content)} chars]"
            else:
                preview = content
            print(f"  Content: {preview}")

    # Error
    if entry.get('error'):
        print(f"\n[ERROR]")
        print(f"  {entry.get('error')}")


def list_recent_logs(agent: Optional[str] = None, limit: int = 10) -> list:
    """List recent log files."""
    pattern = f"{agent}_*" if agent else "*.jsonl"

    if not LOG_DIR.exists():
        print(f"Log directory not found: {LOG_DIR}")
        return []

    log_files = sorted(LOG_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(f) for f in log_files[:limit]]


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="View LLM conversation logs")
    parser.add_argument('-l', '--list', action='store_true', help="List recent log files")
    parser.add_argument('-v', '--verbose', action='store_true', help="Show verbose output")
    parser.add_argument('-a', '--agent', help="Filter by agent name (e.g., diagnose, supervisor)")
    parser.add_argument('-n', '--number', type=int, default=5, help="Number of recent logs to show (default: 5)")
    parser.add_argument('files', nargs='*', help="Specific log files to view")

    args = parser.parse_args()

    if args.list:
        logs = list_recent_logs(agent=args.agent, limit=args.number)
        if logs:
            print(f"\nRecent logs in {LOG_DIR}:")
            for log in logs:
                print(f"  - {Path(log).name}")
        return

    files_to_view = args.files
    if not files_to_view:
        # View recent logs
        files_to_view = list_recent_logs(agent=args.agent, limit=args.number)

    for filepath in files_to_view:
        try:
            print_log_entry(filepath, verbose=args.verbose)
        except Exception as e:
            print(f"\nError reading {filepath}: {e}")


if __name__ == "__main__":
    main()
