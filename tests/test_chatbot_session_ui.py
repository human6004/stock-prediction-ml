"""Contracts for the shared, tab-scoped chatbot session UI."""

from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]

requires_node = pytest.mark.skipif(
    shutil.which("node") is None,
    reason="cần Node.js trên PATH để chạy assertion JS",
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


def test_chat_client_is_loaded_globally_once():
    base = _read("templates/base.html")
    page = _read("templates/chat.html")

    assert base.count("chat-client.js") == 1
    assert "chat-client.js" not in page


def test_session_is_tab_scoped_bounded_and_recovers_from_corruption():
    client = _read("static/chat-client.js")

    assert 'SESSION_KEY = "hose-chat-session-v1"' in client
    assert "MAX_TRANSCRIPT = 40" in client
    assert "window.sessionStorage.getItem(SESSION_KEY)" in client
    assert "window.sessionStorage.setItem(SESSION_KEY" in client
    assert "window.sessionStorage.removeItem(SESSION_KEY)" in client
    assert ".slice(-MAX_TRANSCRIPT)" in client
    assert "catch (error)" in client


def test_transport_sends_six_messages_and_optional_conversation_state():
    client = _read("static/chat-client.js")

    assert "function sendMessage(message, options)" in client
    assert "session.transcript.slice(-6)" in client
    assert "payload.conversation_state = session.conversation_state" in client
    assert 'fetch("/api/chat"' in client
    assert "AbortController" in client
    # timeout client phải >= TOTAL_DEADLINE_SECONDS(60) của server, không được ngắn hơn
    assert "70000" in client


def test_only_successful_exchange_is_persisted():
    client = _read("static/chat-client.js")
    send = client.index("function sendMessage(message, options)")
    ok_check = client.index("if (!response.ok)", send)
    persist = client.index("saveExchange(message, data)", ok_check)

    assert ok_check < persist
    assert "saveExchange(message, data)" not in client[send:ok_check]


@requires_node
def test_session_runtime_reload_bounds_transport_clear_and_corruption_recovery():
    _run_node(
        r"""
        const assert = require("node:assert/strict");
        const modulePath = require.resolve("./static/chat-client.js");
        const values = new Map();
        const storage = {
            getItem(key) { return values.has(key) ? values.get(key) : null; },
            setItem(key, value) { values.set(key, value); },
            removeItem(key) { values.delete(key); }
        };
        global.window = {
            sessionStorage: storage,
            setTimeout() { return 1; },
            clearTimeout() {},
            matchMedia() { return {matches: false}; }
        };

        function loadKit() {
            delete require.cache[modulePath];
            require(modulePath);
            return window.ChatClientKit;
        }

        (async () => {
            let kit = loadKit();
            const key = kit.SESSION_KEY;
            storage.setItem(key, JSON.stringify({
                transcript: [],
                conversation_state: {
                    active_symbols: ["INVALID"], topic: null,
                    ranking_order: null, last_result_symbols: []
                }
            }));
            assert.equal(kit.loadSession().conversation_state, null);
            assert.equal(storage.getItem(key), null);

            const requests = [];
            global.fetch = async (_url, options) => {
                requests.push(JSON.parse(options.body));
                return {
                    ok: true,
                    json: async () => ({
                        answer: "Đã trả lời.", sources: [], warnings: [],
                        release_status: "current",
                        conversation_state: {
                            active_symbols: ["FPT"], topic: "signal",
                            ranking_order: null, last_result_symbols: ["FPT"]
                        }
                    })
                };
            };
            for (let index = 0; index < 21; index += 1) {
                await kit.sendMessage("Câu " + index);
            }
            const saved = JSON.parse(storage.getItem(key));
            assert.equal(saved.transcript.length, 40);
            assert.equal(requests.at(-1).history.length, 6);
            assert.deepEqual(requests.at(-1).conversation_state.active_symbols, ["FPT"]);

            kit = loadKit();
            assert.equal(kit.loadSession().transcript.length, 40);

            const beforeError = storage.getItem(key);
            global.fetch = async () => ({
                ok: false,
                json: async () => ({error: {message: "Provider lỗi"}})
            });
            await assert.rejects(kit.sendMessage("Không lưu lượt này"));
            assert.equal(storage.getItem(key), beforeError);

            kit.clearSession();
            assert.equal(storage.getItem(key), null);
        })().catch((error) => {
            process.nextTick(() => { throw error; });
        });
        """
    )


def test_dock_and_full_page_share_restore_clear_and_rich_metadata():
    base = _read("templates/base.html")
    page = _read("templates/chat.html")

    for source in (base, page):
        assert "kit.restoreSession(" in source
        assert "kit.sendMessage(" in source
        assert "kit.buildSourceDetails(" in source
    assert "kit.clearSession()" in base
    assert "kit.clearSession()" in page
    assert "conversation_state" in _read("static/chat-client.js")


def test_llm_content_is_only_rendered_through_safe_dom_apis():
    sources = "\n".join(
        _read(path)
        for path in (
            "static/chat-client.js",
            "templates/base.html",
            "templates/chat.html",
        )
    )

    assert "document.createElement" in sources
    assert "textContent" in sources
    assert "innerHTML" not in sources
