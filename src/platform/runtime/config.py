"""从环境和项目配置文件读取运行期设置的技术边界。

该模块可同时被 HTTP、后台任务和平台适配器使用；它不包含任何投资或产品决策。
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings

from src.platform.marketdata.models import MarketCode


class Settings(BaseSettings):
    """环境变量配置"""

    # Runtime profile. The Ahmed fork runs as an XAU/USD research terminal by
    # default; "legacy" restores the original stock-centric background jobs.
    panwatch_profile: str = "xau"

    # AI
    # This fork defaults to Atria's OpenAI-compatible endpoint. Only the
    # secret key is required at deploy time; the provider/model are
    # bootstrapped into PanWatch automatically on startup.
    ai_base_url: str = "https://api.atria-asi.ai/v1"
    ai_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("AI_API_KEY", "ATRIA_API_KEY"),
    )
    ai_model: str = "Atria-Dawn-Preview"

    # Assistant context engineering. The compression model is optional: when
    # unset, the host reuses the configured default assistant model.
    context_compression_model_id: int | None = Field(default=None, ge=1)
    context_compression_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    context_summary_max_tokens: int = Field(default=800, ge=128, le=4_000)
    context_max_tokens: int = Field(default=12_000, ge=256)
    context_soft_limit_tokens: int = Field(default=8_400, ge=128)
    context_hard_limit_tokens: int = Field(default=10_200, ge=256)
    context_keep_recent_messages: int = Field(default=8, ge=1, le=100)
    tool_research_enabled: bool = True

    # Durable Ahmed Research Engine / PanWatch belief store.
    # Production should point this at external PostgreSQL. Empty keeps the
    # local SQLite compatibility fallback.
    research_database_url: str = Field(
        default="",
        validation_alias=AliasChoices(
            "PANWATCH_RESEARCH_DATABASE_URL",
            "RESEARCH_DATABASE_URL",
        ),
    )

    # Ahmed ToolBox MCP. Empty URL keeps the integration disabled and leaves
    # PanWatch's built-in tools unchanged.
    ahmed_toolbox_url: str = ""
    ahmed_toolbox_token: str = ""
    ahmed_toolbox_timeout_seconds: float = Field(default=5.0, ge=0.5, le=30.0)

    # Automatic falsification/counter-research loop. Disabled by default in
    # generic installs; production explicitly enables it after durable-store
    # and external-tool verification.
    auto_research_enabled: bool = False
    auto_research_interval_minutes: int = Field(default=30, ge=15, le=1440)
    auto_research_max_probes: int = Field(default=2, ge=1, le=8)
    auto_research_max_sources_per_probe: int = Field(default=2, ge=1, le=5)
    auto_research_max_findings_per_document: int = Field(default=2, ge=1, le=5)
    auto_research_max_tool_calls: int = Field(default=8, ge=2, le=30)
    auto_research_tool_timeout_seconds: int = Field(default=25, ge=5, le=60)
    auto_research_extraction_timeout_seconds: int = Field(default=40, ge=15, le=120)
    auto_research_extraction_max_chars: int = Field(default=6000, ge=1500, le=12000)
    auto_research_probe_cooldown_minutes: int = Field(default=180, ge=15, le=10080)
    auto_research_bootstrap_xau_claims: bool = True

    # General, domain-neutral claim acquisition. A bounded scheduler may use
    # configured topics or active claims as seeds; the core can also ingest
    # arbitrary document/transcript text directly.
    claim_acquisition_enabled: bool = False
    claim_acquisition_interval_minutes: int = Field(default=120, ge=30, le=10080)
    claim_acquisition_max_topics: int = Field(default=1, ge=1, le=5)
    claim_acquisition_max_documents: int = Field(default=2, ge=1, le=8)
    claim_acquisition_max_claims_per_document: int = Field(default=3, ge=1, le=10)
    claim_acquisition_max_tool_calls: int = Field(default=6, ge=2, le=30)
    claim_acquisition_extraction_timeout_seconds: int = Field(default=45, ge=15, le=120)
    claim_acquisition_extraction_max_chars: int = Field(default=7000, ge=1500, le=16000)
    claim_acquisition_topics: str = ""
    claim_acquisition_bootstrap_from_active_claims: bool = True

    # XAU weekly paper league. These settings are explicit so paper-risk rules
    # are inspectable and never confused with the research decision fusion.
    xau_paper_enabled: bool = True
    xau_paper_initial_capital: float = Field(default=10_000.0, gt=0)
    xau_paper_risk_pct: float = Field(default=0.01, gt=0, le=0.10)
    xau_paper_reward_risk: float = Field(default=2.0, gt=0.25, le=10.0)
    xau_paper_max_leverage: float = Field(default=1.0, ge=0.1, le=100.0)
    xau_paper_max_spread_bps: float = Field(default=3.0, gt=0.0, le=100.0)
    xau_paper_max_hold_minutes: int = Field(default=240, ge=15, le=1440)
    # Multi-speed XAU runtime. Fast deterministic cognition stays in the hot path;
    # macro/LLM context is cached separately in the slow path.
    xau_fast_scan_seconds: int = Field(default=10, ge=5, le=60)
    xau_cognition_enabled: bool = True
    xau_cognition_min_confidence: float = Field(default=0.58, ge=0.50, le=0.90)

    xau_paper_scan_seconds: int = Field(default=15, ge=5, le=3600)
    xau_paper_timezone: str = "Asia/Baghdad"
    xau_paper_database_url: str = ""

    # Research-only historical replay. Runs off the hot path and never creates
    # paper/live fills; it only refreshes bounded episodic research memory.
    xau_replay_enabled: bool = True
    xau_replay_interval_minutes: int = Field(default=360, ge=60, le=1440)
    xau_replay_horizon_minutes: int = Field(default=60, ge=15, le=240)
    xau_replay_step_minutes: int = Field(default=5, ge=1, le=60)
    xau_replay_bar_limit: int = Field(default=1000, ge=60, le=1000)
    xau_replay_lookback_days: int = Field(default=5, ge=1, le=30)

    # Telegram
    notify_telegram_bot_token: str = ""
    notify_telegram_chat_id: str = ""

    # 代理
    http_proxy: str = ""

    # 通知策略（可通过 UI 的“系统设置”覆盖）
    # 静默时间段（本地时区），格式: HH:MM-HH:MM，空为关闭；跨夜示例: 23:00-07:00
    notify_quiet_hours: str = ""
    # 通知失败重试次数（不含首次尝试）
    notify_retry_attempts: int = 2
    # 重试退避秒数（基数），实际会按 1x,2x,... 递增
    notify_retry_backoff_seconds: float = 2.0
    # 幂等窗口覆盖（JSON），示例: {"news_digest":60,"daily_report":720}
    notify_dedupe_ttl_overrides: str = ""

    # SSL 证书（企业环境）
    ca_cert_file: str = ""

    # 调度
    # day_of_week 使用 POSIX cron 语义(1-5=周一到周五)
    daily_report_cron: str = "30 15 * * 1-5"

    # 默认时区（用于调度、时间展示等）。
    # 统一使用一个环境变量控制：TZ（默认 Asia/Shanghai）。
    # 建议使用 IANA 时区名，如 Asia/Shanghai, America/New_York。
    app_timezone: str = Field(
        default="Asia/Shanghai",
        validation_alias=AliasChoices("TZ", "APP_TIMEZONE"),
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        # .env 里可能有 HTTPS_PROXY 等未声明字段(httpx/系统标准变量),忽略不报错
        "extra": "ignore",
    }

    @model_validator(mode="after")
    def validate_context_thresholds(self) -> "Settings":
        if not self.context_soft_limit_tokens < self.context_hard_limit_tokens <= self.context_max_tokens:
            raise ValueError(
                "context thresholds must satisfy soft_limit < hard_limit <= max_tokens"
            )
        return self


@dataclass
class StockConfig:
    """自选股配置"""

    symbol: str
    name: str
    market: MarketCode


@dataclass
class AppConfig:
    """应用完整配置"""

    settings: Settings
    watchlist: list[StockConfig] = field(default_factory=list)


def load_watchlist(path: str | Path = "config/watchlist.yaml") -> list[StockConfig]:
    """从 YAML 加载自选股列表"""
    path = Path(path)
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    stocks = []
    for market_group in data.get("markets", []):
        market_code = MarketCode(market_group["code"])
        for stock in market_group.get("stocks", []):
            stocks.append(
                StockConfig(
                    symbol=stock["symbol"],
                    name=stock["name"],
                    market=market_code,
                )
            )

    return stocks


def load_config() -> AppConfig:
    """加载完整配置"""
    settings = Settings()
    watchlist = load_watchlist()
    return AppConfig(settings=settings, watchlist=watchlist)
