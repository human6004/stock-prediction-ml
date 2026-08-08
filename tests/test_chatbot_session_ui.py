"""Behavior contracts for the single /chat UI and tab-scoped transcript."""

from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="cần Node.js để chạy assertion JS"
)


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _run_node(source: str) -> None:
    result = subprocess.run(
        ["node", "-e", textwrap.dedent(source)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_chat_assets_load_only_on_chat_page():
    base = _read("templates/base.html")
    page = _read("templates/chat.html")
    assert "chat-client.js" not in base
    assert "chat-ui.css" not in base
    assert page.count("chat-client.js") == 1
    assert page.count("chat-ui.css") == 1


def test_transport_sends_last_six_messages_without_conversation_state():
    client = _read("static/chat-client.js")
    assert "session.transcript.slice(-6)" in client
    assert 'fetch("/api/chat"' in client
    assert "AbortController" in client
    assert "70000" in client
    assert "conversation_state" not in client
    assert "release_status" not in client


def test_only_successful_exchange_is_persisted():
    client = _read("static/chat-client.js")
    send = client.index("function sendMessage(message, options)")
    ok_check = client.index("if (!response.ok)", send)
    persist = client.index("saveExchange(message, data)", ok_check)
    assert ok_check < persist
    assert "saveExchange(message, data)" not in client[send:ok_check]


@requires_node
def test_session_runtime_bounds_history_recovers_and_does_not_persist_failures():
    _run_node(
        r"""
        const assert = require("node:assert/strict");
        const modulePath = require.resolve("./static/chat-client.js");
        const values = new Map();
        global.window = {
            sessionStorage: {
                getItem(key) { return values.has(key) ? values.get(key) : null; },
                setItem(key, value) { values.set(key, value); },
                removeItem(key) { values.delete(key); }
            },
            setTimeout() { return 1; },
            clearTimeout() {}
        };
        require(modulePath);
        const kit = window.ChatClientKit;
        const key = kit.SESSION_KEY;

        (async () => {
            values.set(key, JSON.stringify({transcript: [{role: "user", content: "hỏng"}]}));
            assert.deepEqual(kit.loadSession(), {transcript: []});
            assert.equal(values.has(key), false);

            const requests = [];
            global.fetch = async (_url, options) => {
                requests.push(JSON.parse(options.body));
                return {
                    ok: true,
                    json: async () => ({
                        answer: "Đã trả lời.", sources: [], warnings: [],
                        data_as_of: "2026-07-20",
                        model_trained_through: "2026-04-10"
                    })
                };
            };
            for (let index = 0; index < 21; index += 1) {
                await kit.sendMessage("Câu " + index);
            }
            const saved = JSON.parse(values.get(key));
            assert.equal(saved.transcript.length, 40);
            assert.equal(requests.at(-1).history.length, 6);
            assert.equal("conversation_state" in requests.at(-1), false);
            assert.equal(saved.transcript.at(-1).data_as_of, "2026-07-20");

            const beforeError = values.get(key);
            global.fetch = async () => ({
                ok: false,
                json: async () => ({error: {message: "Provider lỗi"}})
            });
            await assert.rejects(kit.sendMessage("Không lưu"));
            assert.equal(values.get(key), beforeError);

            kit.clearSession();
            assert.equal(values.has(key), false);
        })().catch((error) => process.nextTick(() => { throw error; }));
        """
    )


def test_only_one_chat_ui_and_llm_text_uses_safe_dom():
    base = _read("templates/base.html")
    page = _read("templates/chat.html")
    client = _read("static/chat-client.js")
    assert "chat-dock" not in base
    assert 'id="chat-form"' in page
    assert "textContent" in page + client
    assert "innerHTML" not in page + client
    assert "renderRichText" not in page + client
    assert "buildSourceDetails" not in page + client
