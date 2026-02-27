"""
conftest.py - 测试 fixtures（v4 HITL 闭环升级）

session 级 fixture：
- ensure_test_seed_data: 确保测试用户和 Bot 存在
- mock_backend: Mock Java 后端 HTTP 调用
"""
import os
import sys
import logging
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Any, Optional

# 确保 aiservice 目录在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logger = logging.getLogger(__name__)

# 固定测试 ID
TEST_USER_ID = 99999
TEST_BOT_ID = 99999
TEST_BOT_NAME = "TestBot_HITL"
TEST_BOT_CODE = """
import java.util.Scanner;

public class TestBot {
    public static void main(String[] args) {
        String input = System.getenv("INPUT");
        System.out.println(0);
    }

    Integer nextMove(String input) {
        return 0;
    }
}
""".strip()


def _seed_mysql_if_available() -> Dict[str, Any]:
    """在可用时向 MySQL 写入测试用户/Bot；不可用时安全降级。"""
    try:
        from db_client import get_mysql_connection  # 本地模块
    except Exception as e:
        return {"seeded": False, "reason": f"db_client unavailable: {e}"}

    try:
        with get_mysql_connection() as conn:
            with conn.cursor() as cur:
                # 表存在性检查
                cur.execute("SHOW TABLES LIKE 'user'")
                if not cur.fetchone():
                    return {"seeded": False, "reason": "table user not found"}
                cur.execute("SHOW TABLES LIKE 'bot'")
                if not cur.fetchone():
                    return {"seeded": False, "reason": "table bot not found"}

                # 确保动作表存在（幂等）
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS bot_action_log (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        bot_id INT NOT NULL,
                        action_id VARCHAR(128) NOT NULL,
                        action_type VARCHAR(32) NOT NULL DEFAULT 'code_update',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE KEY uk_bot_action (bot_id, action_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )

                # 用户存在性
                cur.execute("SELECT id FROM user WHERE id = %s LIMIT 1", (TEST_USER_ID,))
                user_exists = cur.fetchone() is not None
                if not user_exists:
                    cur.execute(
                        """
                        INSERT INTO user (id, username, password, photo, rating)
                        VALUES (%s, %s, %s, %s, %s)
                        """,
                        (
                            TEST_USER_ID,
                            f"test_hitl_user_{TEST_USER_ID}",
                            "test_password_hash",
                            "https://example.com/test.png",
                            1500,
                        ),
                    )

                # Bot 存在性
                cur.execute("SELECT id FROM bot WHERE id = %s LIMIT 1", (TEST_BOT_ID,))
                bot_exists = cur.fetchone() is not None
                if not bot_exists:
                    cur.execute(
                        """
                        INSERT INTO bot (id, user_id, title, description, code, is_default)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            TEST_BOT_ID,
                            TEST_USER_ID,
                            TEST_BOT_NAME,
                            "Auto-seeded test bot for HITL tests",
                            TEST_BOT_CODE,
                            0,
                        ),
                    )

            conn.commit()
            return {
                "seeded": True,
                "user_created": not user_exists,
                "bot_created": not bot_exists,
            }
    except Exception as e:
        return {"seeded": False, "reason": str(e)}


@pytest.fixture(scope="session", autouse=True)
def ensure_test_seed_data():
    """数据库为空时自动插入测试样例；不可达时仅记录日志，不阻塞单测。"""
    result = _seed_mysql_if_available()
    if result.get("seeded"):
        logger.info(
            "Seed ready: user_id=%d bot_id=%d user_created=%s bot_created=%s",
            TEST_USER_ID,
            TEST_BOT_ID,
            result.get("user_created"),
            result.get("bot_created"),
        )
    else:
        logger.warning(
            "Seed skipped: user_id=%d bot_id=%d reason=%s",
            TEST_USER_ID,
            TEST_BOT_ID,
            result.get("reason"),
        )
    yield


@dataclass
class MockBackendState:
    """追踪 mock 后端的状态"""
    update_calls: list
    action_log: dict  # action_id → bool (幂等追踪)
    should_fail_update: bool = False

    def reset(self):
        self.update_calls = []
        self.action_log = {}
        self.should_fail_update = False


@pytest.fixture(scope="session")
def test_ids():
    """返回固定的测试 ID"""
    logger.info("Test seed: user_id=%d, bot_id=%d", TEST_USER_ID, TEST_BOT_ID)
    return {
        "user_id": TEST_USER_ID,
        "bot_id": TEST_BOT_ID,
        "bot_name": TEST_BOT_NAME,
        "bot_code": TEST_BOT_CODE,
    }


@pytest.fixture
def backend_state():
    """每个测试独立的后端状态"""
    return MockBackendState(update_calls=[], action_log={})


@pytest.fixture
def mock_backend(backend_state):
    """Mock Java 后端的 HTTP 调用。
    
    拦截 requests.post 到 /ai/bot/manage/update，
    模拟幂等行为（action_id 判重）。
    """
    original_post = None

    def mock_post(url, json=None, **kwargs):
        if "/ai/bot/manage/update" in url:
            backend_state.update_calls.append(json)
            action_id = json.get("actionId") if json else None

            if backend_state.should_fail_update:
                resp = MagicMock()
                resp.status_code = 500
                resp.json.return_value = {"success": False, "error": "DB_ERROR"}
                return resp

            # 幂等判重
            if action_id and action_id in backend_state.action_log:
                resp = MagicMock()
                resp.status_code = 200
                resp.json.return_value = {
                    "success": True,
                    "duplicate": True,
                    "message": "操作已执行 (幂等)",
                }
                return resp

            if action_id:
                backend_state.action_log[action_id] = True

            resp = MagicMock()
            resp.status_code = 200
            resp.json.return_value = {
                "success": True,
                "duplicate": False,
                "message": "Bot 代码更新成功",
            }
            return resp

        if "/ai/bot/compile-check" in url:
            resp = MagicMock()
            resp.status_code = 404
            return resp

        # 其他 URL pass-through
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"success": True}
        return resp

    with patch("agent.executor.http_requests.post", side_effect=mock_post) as m:
        yield backend_state


@pytest.fixture
def mock_backend_for_verifier():
    """Mock 编译检查端点（返回 404 表示不可用）"""
    def mock_post(url, json=None, **kwargs):
        resp = MagicMock()
        if "/ai/bot/compile-check" in url:
            resp.status_code = 404
        else:
            resp.status_code = 200
            resp.json.return_value = {"success": True}
        return resp

    with patch("agent.verifier.http_requests.post", side_effect=mock_post):
        yield


@pytest.fixture(autouse=True)
def clear_completed_actions():
    """每个测试前清空进程内幂等缓存"""
    from agent.executor import _completed_actions
    _completed_actions.clear()
    yield
    _completed_actions.clear()
