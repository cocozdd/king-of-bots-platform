"""
数据库客户端 - PostgreSQL (pgvector) + MySQL 连接

职责：
- PostgreSQL: 向量检索（ai_corpus 表）
- MySQL: 对战记录查询（record 表）

安全设计：
- 读操作：Python 直连数据库
- 写操作：通过 Java API（确保事务一致性）
"""
import os
import logging
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# PostgreSQL 连接（用于向量检索）
_pg_pool = None

# MySQL 连接（用于对战记录）
_mysql_pool = None


def get_pg_config() -> Dict[str, Any]:
    """获取 PostgreSQL 配置"""
    return {
        "host": os.getenv("PGVECTOR_HOST", "127.0.0.1"),
        "port": int(os.getenv("PGVECTOR_PORT", "15432")),
        "database": os.getenv("PGVECTOR_DATABASE", "kob_ai"),
        "user": os.getenv("PGVECTOR_USER", "cocodzh"),
        "password": os.getenv("PGVECTOR_PASSWORD", "123"),
    }


def get_mysql_config() -> Dict[str, Any]:
    """获取 MySQL 配置"""
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "database": os.getenv("MYSQL_DATABASE", "kob"),
        "user": os.getenv("MYSQL_USER", "root"),
        "password": os.getenv("MYSQL_PASSWORD", "123456"),
        "charset": "utf8mb4",
    }


def init_pg_pool():
    """初始化 PostgreSQL 连接池"""
    global _pg_pool
    if _pg_pool is not None:
        return _pg_pool
    
    try:
        import psycopg2
        from psycopg2 import pool
        
        config = get_pg_config()
        _pg_pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            host=config["host"],
            port=config["port"],
            database=config["database"],
            user=config["user"],
            password=config["password"],
        )
        logger.info("PostgreSQL 连接池初始化成功: %s:%d/%s",
                    config["host"], config["port"], config["database"])
        return _pg_pool
    except ImportError:
        logger.warning("psycopg2 未安装，PostgreSQL 功能不可用")
        return None
    except Exception as e:
        logger.error("PostgreSQL 连接池初始化失败: %s", e)
        return None


def init_mysql_pool():
    """初始化 MySQL 连接池"""
    global _mysql_pool
    if _mysql_pool is not None:
        return _mysql_pool
    
    try:
        import pymysql
        from dbutils.pooled_db import PooledDB
        
        config = get_mysql_config()
        _mysql_pool = PooledDB(
            creator=pymysql,
            maxconnections=10,
            mincached=1,
            maxcached=5,
            blocking=True,
            host=config["host"],
            port=config["port"],
            database=config["database"],
            user=config["user"],
            password=config["password"],
            charset=config["charset"],
            cursorclass=pymysql.cursors.DictCursor,
        )
        logger.info("MySQL 连接池初始化成功: %s:%d/%s",
                    config["host"], config["port"], config["database"])
        return _mysql_pool
    except ImportError:
        logger.warning("pymysql 或 dbutils 未安装，MySQL 功能不可用")
        return None
    except Exception as e:
        logger.error("MySQL 连接池初始化失败: %s", e)
        return None


@contextmanager
def get_pg_connection():
    """获取 PostgreSQL 连接（上下文管理器）"""
    pool = init_pg_pool()
    if pool is None:
        raise RuntimeError("PostgreSQL 连接池未初始化")
    
    conn = pool.getconn()
    try:
        yield conn
    finally:
        pool.putconn(conn)


@contextmanager
def get_mysql_connection():
    """获取 MySQL 连接（上下文管理器）"""
    pool = init_mysql_pool()
    if pool is None:
        raise RuntimeError("MySQL 连接池未初始化")
    
    conn = pool.connection()
    try:
        yield conn
    finally:
        conn.close()


def pg_query(sql: str, params: tuple = None) -> List[Dict[str, Any]]:
    """执行 PostgreSQL 查询"""
    try:
        with get_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        logger.error("PostgreSQL 查询失败: %s, SQL: %s", e, sql[:200])
        return []


def mysql_query(sql: str, params: tuple = None) -> List[Dict[str, Any]]:
    """执行 MySQL 查询"""
    try:
        with get_mysql_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()
    except Exception as e:
        logger.error("MySQL 查询失败: %s, SQL: %s", e, sql[:200])
        return []


# ==================== 业务查询方法 ====================

def vector_search(query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    向量相似度检索
    
    Args:
        query_embedding: 查询向量（1536维）
        top_k: 返回数量
        
    Returns:
        检索结果列表，包含 id, title, content, category, score
    """
    embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
    sql = """
        SELECT id, title, content, category,
               1 - (embedding <=> %s::vector) as score
        FROM ai_corpus
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """
    return pg_query(sql, (embedding_str, embedding_str, top_k))


def keyword_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    关键词检索（PostgreSQL 全文搜索）
    
    使用 plainto_tsquery 避免中文特殊字符导致的语法错误
    """
    # 清洗查询词：移除可能导致 tsquery 错误的字符
    cleaned_query = "".join(c for c in query if c.isalnum() or c.isspace())
    if not cleaned_query.strip():
        return []
    
    sql = """
        SELECT id, title, content, category,
               ts_rank_cd(
                   setweight(to_tsvector('simple', coalesce(title, '')), 'A') ||
                   setweight(to_tsvector('simple', coalesce(content, '')), 'B'),
                   plainto_tsquery('simple', %s)
               ) as score
        FROM ai_corpus
        WHERE to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(content, ''))
              @@ plainto_tsquery('simple', %s)
        ORDER BY score DESC
        LIMIT %s
    """
    return pg_query(sql, (cleaned_query, cleaned_query, top_k))


def get_battle_records(user_id: int, limit: int = 10) -> List[Dict[str, Any]]:
    """
    查询用户对战记录
    
    Args:
        user_id: 用户ID
        limit: 返回数量
        
    Returns:
        对战记录列表
    """
    sql = """
        SELECT id, a_id, a_sx, a_sy, b_id, b_sx, b_sy,
               a_steps, b_steps, map, loser, createtime
        FROM record
        WHERE a_id = %s OR b_id = %s
        ORDER BY createtime DESC
        LIMIT %s
    """
    return mysql_query(sql, (user_id, user_id, limit))


def get_user_stats(user_id: int) -> Dict[str, Any]:
    """
    获取用户统计数据
    
    Args:
        user_id: 用户ID
        
    Returns:
        统计数据：总场次、胜场、负场、胜率
    """
    sql = """
        SELECT
            COUNT(*) as total_games,
            SUM(CASE WHEN (a_id = %s AND loser = 'B') OR (b_id = %s AND loser = 'A') THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN (a_id = %s AND loser = 'A') OR (b_id = %s AND loser = 'B') THEN 1 ELSE 0 END) as losses
        FROM record
        WHERE a_id = %s OR b_id = %s
    """
    results = mysql_query(sql, (user_id, user_id, user_id, user_id, user_id, user_id))
    if not results:
        return {"total_games": 0, "wins": 0, "losses": 0, "win_rate": 0.0}
    
    stats = results[0]
    total = stats.get("total_games", 0) or 0
    wins = stats.get("wins", 0) or 0
    losses = stats.get("losses", 0) or 0
    win_rate = (wins / total * 100) if total > 0 else 0.0
    
    return {
        "total_games": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 2),
    }


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """获取用户信息"""
    sql = "SELECT id, username, rating FROM user WHERE id = %s"
    results = mysql_query(sql, (user_id,))
    return results[0] if results else None


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """
    通过用户名获取用户信息
    
    Args:
        username: 用户名
        
    Returns:
        用户信息字典，包含 id, username, rating；未找到返回 None
    """
    sql = "SELECT id, username, rating FROM user WHERE username = %s"
    results = mysql_query(sql, (username,))
    return results[0] if results else None


def get_bot_by_id(bot_id: int) -> Optional[Dict[str, Any]]:
    """获取 Bot 信息"""
    sql = "SELECT id, user_id, title, description, content, rating FROM bot WHERE id = %s"
    results = mysql_query(sql, (bot_id,))
    return results[0] if results else None


def get_user_bots(user_id: int) -> List[Dict[str, Any]]:
    """获取用户的所有 Bot"""
    sql = """
        SELECT id, title, description, rating, createtime, modifytime
        FROM bot
        WHERE user_id = %s
        ORDER BY modifytime DESC
    """
    return mysql_query(sql, (user_id,))


# ==================== 健康检查 ====================

def check_pg_health() -> Dict[str, Any]:
    """检查 PostgreSQL 连接状态"""
    try:
        result = pg_query("SELECT 1 as ok")
        return {"status": "ok", "connected": bool(result)}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def check_mysql_health() -> Dict[str, Any]:
    """检查 MySQL 连接状态"""
    try:
        result = mysql_query("SELECT 1 as ok")
        return {"status": "ok", "connected": bool(result)}
    except Exception as e:
        return {"status": "error", "error": str(e)}
