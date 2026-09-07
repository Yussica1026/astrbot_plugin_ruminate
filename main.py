"""astrbot_plugin_ruminate · 自主循环·独处沉思插件

让 AI 拥有自主循环：用户离开时用 LLM 想一轮，存进草稿池；
用户回来时把最近的沉思注入上下文，让对话自然接上"我不在时也在想事"。

功能：
- 独处沉思：用户离开超过阈值后触发，LLM 生成内心活动存入草稿池
- 当日回顾：深夜（23-5 点）自动回顾当天，每天最多一轮
- 草稿池注入：用户回来时把最近沉思拼进 system prompt

配置（metadata.yaml -> 面板插件配置）：
- target_user_id: 要监测的用户 ID（QQ 号等），留空则监测所有私聊
- check_interval_minutes: 心跳间隔分钟数，默认 30
- chat_archive_dir: 聊天存档目录（可选，留空不注入对话上下文）
"""

import asyncio
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from astrbot.api.all import *
from astrbot.api import filter
from astrbot.api.event import AstrMessageEvent
from astrbot.api.message_components import Plain

from . import ruminate

logger = logger if "logger" in dir() else None

# 统一在 __init__ 里从 self.logger 取


class RuminateStar(Star):
    """自主循环·独处沉思。用户在时陪聊，离开时自己思考存草稿池。"""

    def __init__(self, context: Context, config: dict) -> None:
        super().__init__(context)
        self.config = config or {}
        self.logger = context.get_logger() if hasattr(context, "get_logger") else None
        if self.logger is None:
            import logging
            self.logger = logging.getLogger("ruminate")

        self._last_msg_ts: float = 0.0
        self._ruminate_lock = asyncio.Lock()

        # ---- 配置 ----
        env_data_dir = os.environ.get("ASTRBOT_DATA_DIR", "") or os.environ.get("ASTRBOT_HOME", "")
        data_dir = Path(env_data_dir) if env_data_dir else Path.cwd() / "data"
        plugin_data_dir = data_dir / "plugin_data" / "astrbot_plugin_ruminate"
        plugin_data_dir.mkdir(parents=True, exist_ok=True)

        self.target_user_id = str(self.config.get("target_user_id", "") or "")
        self.check_interval = int(self.config.get("check_interval_minutes", 30) or 30)

        # ---- 初始化核心模块 ----
        ruminate.init_table()
        ruminate.set_db_path(str(plugin_data_dir / "ruminations.db"))
        archive_dir = self.config.get("chat_archive_dir", "")
        if archive_dir:
            ruminate.set_chat_archive_dir(str(archive_dir))

        # 从 AstrBot 当前 provider 读 LLM 配置（动态，不写死任何 key）
        base, key, model = self._load_current_llm_config()
        if base and key and model:
            ruminate.set_llm_config(base, key, model)
            self.logger.info(f"[ruminate] LLM 渠道已绑定: {model}")
        else:
            self.logger.warning("[ruminate] 未获取到 LLM 渠道，沉思生成将不可用")

        # 启动心跳
        self._heartbeat_task = asyncio.ensure_future(self._heartbeat_loop())
        self.logger.info("[ruminate] 自主循环心跳已启动")

    # ---------------- LLM 渠道 ----------------

    def _load_current_llm_config(self):
        """读取 AstrBot 当前默认 provider 渠道，返回 (base, key, model)。失败返回空。"""
        try:
            env_data_dir = os.environ.get("ASTRBOT_DATA_DIR", "") or os.environ.get("ASTRBOT_HOME", "")
            base_dir = Path(env_data_dir) if env_data_dir else Path.cwd()
            cmd_config_path = base_dir / "cmd_config.json"
            if not cmd_config_path.exists():
                return "", "", ""
            import json
            with open(cmd_config_path, encoding="utf-8-sig") as f:
                cfg = json.load(f)
            pid = cfg.get("provider_settings", {}).get("default_provider_id", "")
            if not pid:
                return "", "", ""
            src_id, _, model = pid.partition("/")
            if not src_id or not model:
                return "", "", ""
            for s in cfg.get("provider_sources", []):
                if s.get("id") == src_id:
                    key_field = s.get("key", "")
                    if isinstance(key_field, list):
                        key = key_field[0] if key_field else ""
                    else:
                        key = key_field
                    base = s.get("api_base", "")
                    if key and base:
                        return base, key, model
            return "", "", ""
        except Exception:
            return "", "", ""

    # ---------------- 心跳 ----------------

    async def _heartbeat_loop(self):
        await asyncio.sleep(60)  # 等插件完全加载
        if self._last_msg_ts <= 0:
            self._last_msg_ts = time.time()
        while True:
            try:
                absent_hours = (time.time() - self._last_msg_ts) / 3600.0
                now_server = datetime.now(timezone(timedelta(hours=3)))

                # 深夜回顾优先
                if ruminate.should_review(absent_hours, now_server):
                    await self._try_generate("review", absent_hours)
                elif ruminate.should_ruminate(absent_hours, now_server):
                    await self._try_generate("solitude", absent_hours)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"[ruminate] 心跳异常（静默）: {e}")
            await asyncio.sleep(self.check_interval * 60)

    async def _try_generate(self, kind: str, absent_hours: float) -> None:
        if self._ruminate_lock.locked():
            return
        async with self._ruminate_lock:
            try:
                content = await ruminate.gen_rumination(kind, absent_hours)
                rid = ruminate.save_rumination(kind, content)
                if rid:
                    self.logger.info(f"[ruminate] 第{rid}条[{kind}]已存入草稿池: {content[:60]}")
            except Exception as e:
                self.logger.error(f"[ruminate] 生成异常（静默）: {e}")

    # ---------------- 消息接入 ----------------

    @filter.on_llm_request()
    async def on_llm_request(self, event: AstrMessageEvent, req: ProviderRequest):
        """用户发消息时刷新在场时间，并把草稿池注入上下文。"""
        try:
            self._last_msg_ts = time.time()

            # 只注入目标用户（若配置了）
            if self.target_user_id:
                sender_id = str(event.get_sender_id() if hasattr(event, "get_sender_id") else "")
                if sender_id and sender_id != self.target_user_id:
                    return

            recent = ruminate.get_recent_ruminations(3)
            if not recent:
                return
            block = "\n\n## 独处时想过的（草稿池）\n"
            for r in recent:
                kind_label = "当日回顾" if r["kind"] == "review" else "独处沉思"
                ts = str(r.get("created_at", ""))[5:16]
                block += f"- [{kind_label} {ts}] {r['content'][:200]}\n"
            req.system_prompt = (req.system_prompt or "") + block
        except Exception as e:
            self.logger.error(f"[ruminate] 注入异常: {e}")

    # ---------------- 指令 ----------------

    @filter.command("ruminate")
    async def ruminate_command(self, event: AstrMessageEvent):
        """查看草稿池：/ruminate 或 /ruminate 数量"""
        try:
            argv = event.get_message_str().strip().split()
            n = 5
            if len(argv) > 1 and argv[1].isdigit():
                n = min(int(argv[1]), 20)
            recent = ruminate.get_recent_ruminations(n)
            if not recent:
                yield event.plain_result("草稿池是空的。她一直在，还没机会独处。")
                return
            lines = ["草稿池最近几条："]
            for r in recent:
                kind_label = "当日回顾" if r["kind"] == "review" else "独处沉思"
                ts = str(r.get("created_at", ""))[5:16]
                lines.append(f"[{kind_label} {ts}] {r['content']}")
            yield event.plain_result("\n\n".join(lines))
        except Exception as e:
            yield event.plain_result(f"读取失败: {e}")
