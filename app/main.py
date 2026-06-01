"""Main entry point for the AIOps Agent"""

import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from app.agents.main_graph import main_graph
from app.utils.logger import get_logger
from app.utils.llm_logger import reset_trace_logger

logger = get_logger(__name__)


def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("AIOps 根因分析智能助手 v1.0.0")
    print("=" * 60)
    print()
    print("我可以帮您进行异常检测和故障根因分析：")
    print("- 使用 3-Sigma 算法检测异常指标")
    print("- 使用 IAF-RCL 算法识别根因指标")
    print("- 使用 KE-FPC 算法生成故障传播图")
    print("- 综合多个算法结果进行精确定位")
    print()
    print("请提供您的监控数据 CSV 文件。")
    print()
    print("示例：")
    print('  "2026年1月5日5:48发生了故障，请检测异常指标"')
    print('  "分析 ./data/metrics.csv 文件的根因"')
    print('  "诊断 data.csv，注入时间: 100"')
    print('  "分析 metrics.csv，gamma: 5, alpha: 0.05"')
    print()
    print("输入 'quit' 或 'exit' 退出")
    print("=" * 60)
    print()


def run_conversation():
    """
    Run the main conversation loop.

    Simplified flow:
    1. Accept user input
    2. Pass to supervisor (which handles everything)
    3. Display response
    4. Continue until user exits
    """
    load_dotenv()
    print_banner()

    # Initialize conversation state
    messages = []
    continue_conversation = True

    while continue_conversation:
        try:
            # Get user input
            user_input = input("\n您: ").strip()

            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                print("\n感谢使用 AIOps 智能助手，再见！")
                break

            if not user_input:
                print("请输入您的问题或请求。")
                continue

            # Add user message to history
            messages.append(HumanMessage(content=user_input))

            # Start a fresh trace session for this request
            reset_trace_logger()

            # Prepare state for main graph
            # Simplified state: only user_input is required
            initial_state = {
                "user_input": user_input,
                "messages": messages,
                "continue_conversation": True
            }

            # Execute workflow
            logger.info("Executing main workflow")
            result = main_graph.invoke(initial_state)

            # Update messages from result
            messages = result.get("messages", [])

            # Get and print assistant's response (last AI message)
            for msg in reversed(messages):
                if isinstance(msg, AIMessage):
                    print(f"\n助手: {msg.content}")
                    break

            # Check if conversation should continue
            continue_conversation = result.get("continue_conversation", False)

        except KeyboardInterrupt:
            print("\n\n对话被中断。再见！")
            break
        except Exception as e:
            logger.error(f"Error in conversation loop: {e}", exc_info=True)
            print(f"\n发生错误: {e}")
            print("请重试或输入 'quit' 退出。")


def run_single_request(user_request: str) -> str:
    """
    Run a single diagnosis request (non-interactive mode).

    Args:
        user_request: User's request

    Returns:
        Response message
    """
    load_dotenv()

    logger.info(f"Processing single request: {user_request}")

    # Start a fresh trace session for this request
    reset_trace_logger()

    messages = [HumanMessage(content=user_request)]

    initial_state = {
        "user_input": user_request,
        "messages": messages,
        "continue_conversation": False
    }

    try:
        result = main_graph.invoke(initial_state)

        # Extract the final response
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage):
                return msg.content

        return "处理完成，但没有返回结果。"

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return f"处理请求时出错: {e}"


def main():
    """
    Main entry function.

    Supports both interactive and single-request modes.
    """
    import argparse

    parser = argparse.ArgumentParser(description="AIOps 智能助手")
    parser.add_argument(
        "--request", "-r",
        type=str,
        help="单次请求模式：直接处理请求"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="交互式对话模式（默认）"
    )

    args = parser.parse_args()

    if args.request:
        # Single request mode
        result = run_single_request(args.request)
        print(result)
        return 0 if result else 1
    else:
        # Interactive mode (default)
        run_conversation()
        return 0


if __name__ == "__main__":
    sys.exit(main())
