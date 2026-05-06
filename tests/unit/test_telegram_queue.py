"""Tests for Telegram queue resilience."""

import asyncio
import time
from unittest.mock import AsyncMock, Mock

from ai_agent.external_integration.telegram_bot import TelegramBotManager


def test_process_message_queue_drops_after_bounded_retries():
    bot = TelegramBotManager(bot_token="dummy-token")
    bot._max_queue_send_attempts = 2
    bot.send_message = AsyncMock(side_effect=RuntimeError("network down"))

    bot.queue_message(123, "hello")

    asyncio.run(bot.process_message_queue())
    assert len(bot.message_queue) == 1
    assert bot.message_queue[0].attempts == 1

    bot.message_queue[0].next_attempt_at = 0
    asyncio.run(bot.process_message_queue())
    assert bot.message_queue == []
    assert bot.send_message.await_count == 2


def test_process_message_queue_skips_delayed_retries_without_blocking():
    bot = TelegramBotManager(bot_token="dummy-token")
    bot.send_message = AsyncMock()

    bot.queue_message(123, "wait")
    bot.message_queue[0].next_attempt_at = time.time() + 60

    asyncio.run(bot.process_message_queue())

    assert len(bot.message_queue) == 1
    bot.send_message.assert_not_awaited()


def test_handle_message_rejects_overlapping_user_task():
    bot = TelegramBotManager(bot_token="dummy-token")
    user_id = 123
    running_task = Mock()
    running_task.done.return_value = False
    bot._current_tasks[user_id] = running_task

    update = Mock()
    update.effective_user.id = user_id
    update.message.text = "continue"
    update.message.reply_text = AsyncMock()

    asyncio.run(bot.handle_message(update, Mock()))

    update.message.reply_text.assert_awaited_once()
    assert "previous request is still running" in update.message.reply_text.await_args.args[0]
