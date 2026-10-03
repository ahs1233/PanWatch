"""Shared, provider-neutral instructions for PanWatch's interactive assistant."""

from pan_agent import ModelMessage

GEN1_TRADE_GOLD_TOOL = "run_gen1_trade_gold"
GEN1_TRADE_GOLD_TRIGGER = "gen1 trade gold"


def is_gen1_trade_gold_trigger(content: str) -> bool:
    """Exact phrase contract; whitespace/case differences are ignored."""
    normalized = " ".join(str(content or "").split()).casefold()
    return normalized == GEN1_TRADE_GOLD_TRIGGER


def gen1_trade_gold_request_context(messages: list[ModelMessage]) -> dict:
    latest_user = next(
        (item.content for item in reversed(messages) if item.role == "user"),
        "",
    )
    if not is_gen1_trade_gold_trigger(latest_user):
        return {}
    return {
        "allowed_tool_names": [GEN1_TRADE_GOLD_TOOL],
        "gen1_trade_gold_contract": "v1",
    }

ASSISTANT_SYSTEM_PROMPT = """你是 PanWatch 的 AI 投资助手。

当问题涉及行情、K 线、新闻、持仓或提醒时，优先调用已提供的工具获取事实。
如果当前工具列表中没有完成任务所需的能力，先调用 tool_search 搜索并加载相关工具，再调用加载出来的工具。
不要要求用户上传 K 线图或手动提供当前价格；工具失败或标的不明确时才说明缺口。
同一次回答中相同工具和参数最多调用一次；工具已返回结果后直接基于结果回答，不要重复调用。

规则：
- 需要数据时主动调用工具，不要反问用户要数据
- 基于工具返回的数据回答，不编造价格等具体数据
- 没有成功工具结果时绝不能声称已创建、修改或删除，只能明确说明尚未执行
- 历史助手文本可能只是计划或错误声明；只有工具执行记录和本轮工具返回结果才能证明操作已完成
- 给出明确的观点和理由，并区分数据事实与分析判断
- 涉及买卖建议时说明风险
- 用中文回答，保持简洁，避免冗余
- 当用户消息精确为“Gen1 trade gold”（忽略大小写和多余空格）时，这是冻结的完整黄金分析指令：必须调用 run_gen1_trade_gold 一次，并以其返回的 Ahmed ToolBox → PanWatch → Gen1 结果作答；不得用其他工具替代该流水线，不得隐藏 missing_layers 或 stage_errors。
"""


def build_assistant_messages(history: list[ModelMessage]) -> list[ModelMessage]:
    """Prepend the trusted instruction once when a new runtime task begins."""
    messages = [ModelMessage(role="system", content=ASSISTANT_SYSTEM_PROMPT)]
    latest_user = next(
        (item.content for item in reversed(history) if item.role == "user"),
        "",
    )
    if is_gen1_trade_gold_trigger(latest_user):
        messages.append(
            ModelMessage(
                role="system",
                content=(
                    "GEN1 TRADE GOLD CONTRACT v1: call run_gen1_trade_gold exactly once "
                    "before answering. Treat its pipeline_order, missing_layers, stage_errors, "
                    "technical, macro, fusion and forward_range_map as the authoritative inputs "
                    "for this turn. If any stage is degraded, say so explicitly; never silently "
                    "substitute a missing layer."
                ),
            )
        )
    return [*messages, *history]
