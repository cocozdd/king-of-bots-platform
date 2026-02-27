CREATE TABLE IF NOT EXISTS bot_action_log (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    bot_id INT NOT NULL,
    action_id VARCHAR(128) NOT NULL,
    action_type VARCHAR(32) NOT NULL DEFAULT 'code_update',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_bot_action (bot_id, action_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
