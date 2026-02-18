"""Tests for engram.runtime — Mode state machine and scaffolds."""

import pytest

from engram.runtime.modes import Mode, ModeManager, MODE_CONFIG
from engram.runtime.mind import LocalMind
from engram.runtime.actions import ActionSystem


class TestMode:
    def test_values(self):
        assert Mode.QUIET_PRESENCE.name == "QUIET_PRESENCE"
        assert Mode.ACTIVE_CONVERSATION.name == "ACTIVE_CONVERSATION"
        assert Mode.DEEP_WORK.name == "DEEP_WORK"
        assert Mode.SLEEP.name == "SLEEP"

    def test_mode_configs_exist(self):
        for mode in Mode:
            assert mode in MODE_CONFIG
            cfg = MODE_CONFIG[mode]
            assert "check_interval" in cfg
            assert "can_initiate" in cfg


class TestModeManager:
    def test_default_mode(self):
        mm = ModeManager()
        assert mm.current == Mode.QUIET_PRESENCE

    def test_set_mode(self):
        mm = ModeManager()
        result = mm.set_mode("active", "test")
        assert result["changed"] is True
        assert result["old"] == "QUIET_PRESENCE"
        assert result["new"] == "ACTIVE_CONVERSATION"
        assert mm.current == Mode.ACTIVE_CONVERSATION

    def test_set_same_mode(self):
        mm = ModeManager(default_mode="active")
        result = mm.set_mode("active")
        assert result["changed"] is False

    def test_set_unknown_mode(self):
        mm = ModeManager()
        with pytest.raises(ValueError, match="Unknown mode"):
            mm.set_mode("nonexistent")

    def test_history_tracked(self):
        mm = ModeManager()
        mm.set_mode("active", "start conversation")
        mm.set_mode("deep_work", "need focus")
        assert len(mm.history) == 2
        assert mm.history[0]["mode"] == "QUIET_PRESENCE"
        assert mm.history[0]["exit_reason"] == "start conversation"

    def test_config(self):
        mm = ModeManager(default_mode="active")
        cfg = mm.config()
        assert cfg["can_initiate"] is True
        assert cfg["check_interval"] == 1

    def test_can_initiate(self):
        mm = ModeManager(default_mode="active")
        assert mm.can_initiate() is True
        mm.set_mode("sleep")
        assert mm.can_initiate() is False

    def test_check_interval(self):
        mm = ModeManager(default_mode="sleep")
        assert mm.check_interval() == 300

    def test_status(self):
        mm = ModeManager()
        status = mm.status()
        assert status["mode"] == "QUIET_PRESENCE"
        assert "since" in status
        assert "duration_seconds" in status
        assert "config" in status

    def test_name_aliases(self):
        mm = ModeManager()
        mm.set_mode("active_conversation")
        assert mm.current == Mode.ACTIVE_CONVERSATION


class TestLocalMind:
    def test_stub(self):
        mind = LocalMind()
        assert mind is not None


class TestActionSystem:
    def test_stub(self):
        actions = ActionSystem()
        assert actions is not None
