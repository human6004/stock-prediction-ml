/* Transport and tab-scoped transcript for the single /chat page. */
(function () {
    "use strict";

    var MAX_MESSAGE = 1000;
    var MAX_TRANSCRIPT = 40;
    var SESSION_KEY = "hose-chat-session-v1";
    var REQUEST_TIMEOUT_MS = 70000;

    function asText(value) {
        return value === null || value === undefined ? "" : String(value);
    }

    function emptySession() {
        return {transcript: []};
    }

    function normalizeSources(value) {
        if (!Array.isArray(value)) {
            return [];
        }
        return value.slice(0, 20).map(function (source) {
            var item = source && typeof source === "object" ? source : {};
            return {
                kind: asText(item.kind || "artifact").slice(0, 80),
                symbols: Array.isArray(item.symbols)
                    ? item.symbols.slice(0, 10).map(function (symbol) {
                        return asText(symbol).slice(0, 20);
                    })
                    : [],
                as_of: asText(item.as_of).slice(0, 40)
            };
        });
    }

    function normalizeWarnings(value) {
        if (!Array.isArray(value)) {
            return [];
        }
        return value.slice(0, 20).map(function (warning) {
            return {
                message: asText(
                    warning && warning.message !== undefined ? warning.message : warning
                ).slice(0, MAX_MESSAGE)
            };
        });
    }

    function normalizeEntry(entry, role) {
        if (!entry || entry.role !== role || typeof entry.content !== "string"
                || !entry.content.trim()) {
            throw new Error("invalid_session");
        }
        var clean = {
            role: role,
            content: entry.content.slice(0, MAX_MESSAGE)
        };
        if (role === "assistant") {
            clean.sources = normalizeSources(entry.sources);
            clean.warnings = normalizeWarnings(entry.warnings);
            clean.data_as_of = asText(entry.data_as_of).slice(0, 40) || null;
            clean.model_trained_through = (
                asText(entry.model_trained_through).slice(0, 40) || null
            );
        }
        return clean;
    }

    function normalizeSession(value) {
        if (!value || !Array.isArray(value.transcript)) {
            throw new Error("invalid_session");
        }
        var rows = value.transcript.slice(-MAX_TRANSCRIPT);
        if (rows.length % 2 !== 0) {
            throw new Error("invalid_session");
        }
        return {
            transcript: rows.map(function (entry, index) {
                return normalizeEntry(entry, index % 2 === 0 ? "user" : "assistant");
            })
        };
    }

    function clearSession() {
        try {
            window.sessionStorage.removeItem(SESSION_KEY);
        } catch (error) {
            /* Storage blocked: current page still works. */
        }
    }

    function loadSession() {
        try {
            var raw = window.sessionStorage.getItem(SESSION_KEY);
            return raw ? normalizeSession(JSON.parse(raw)) : emptySession();
        } catch (error) {
            clearSession();
            return emptySession();
        }
    }

    function storeSession(session) {
        try {
            window.sessionStorage.setItem(SESSION_KEY, JSON.stringify(session));
        } catch (error) {
            /* Quota/private mode: do not break the current response. */
        }
    }

    function saveExchange(message, data) {
        var session = loadSession();
        session.transcript.push(
            {role: "user", content: asText(message).slice(0, MAX_MESSAGE)},
            {
                role: "assistant",
                content: asText(data.answer).slice(0, MAX_MESSAGE),
                sources: normalizeSources(data.sources),
                warnings: normalizeWarnings(data.warnings),
                data_as_of: asText(data.data_as_of).slice(0, 40) || null,
                model_trained_through: (
                    asText(data.model_trained_through).slice(0, 40) || null
                )
            }
        );
        session.transcript = session.transcript.slice(-MAX_TRANSCRIPT);
        storeSession(session);
        return session;
    }

    function restoreSession(renderEntry) {
        var session = loadSession();
        if (typeof renderEntry === "function") {
            session.transcript.forEach(renderEntry);
        }
        return session;
    }

    function apiErrorMessage(data) {
        var message = data && data.error && typeof data.error.message === "string"
            ? data.error.message.trim()
            : "";
        return message ? message.slice(0, MAX_MESSAGE) : "Không thể gọi trợ lý lúc này.";
    }

    function sendMessage(message, options) {
        var opts = options || {};
        var session = loadSession();
        var payload = {
            message: asText(message).trim().slice(0, MAX_MESSAGE),
            history: session.transcript.slice(-6).map(function (entry) {
                return {role: entry.role, content: entry.content};
            })
        };
        var controller = new AbortController();
        var timeout = window.setTimeout(function () {
            controller.abort();
        }, opts.timeoutMs || REQUEST_TIMEOUT_MS);

        return fetch("/api/chat", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload),
            signal: controller.signal
        }).then(function (response) {
            return response.json().catch(function () {
                throw new Error("invalid_response");
            }).then(function (data) {
                if (!response.ok) {
                    var error = new Error("api_error");
                    error.publicMessage = apiErrorMessage(data);
                    throw error;
                }
                if (!data || typeof data.answer !== "string" || !data.answer.trim()) {
                    throw new Error("invalid_response");
                }
                try {
                    saveExchange(message, data);
                } catch (error) {
                    /* A storage failure must not discard a successful answer. */
                }
                return data;
            });
        }).finally(function () {
            window.clearTimeout(timeout);
        });
    }

    function enhanceComposer(options) {
        var opts = options || {};
        var input = opts.input;
        var form = opts.form;
        var counter = opts.counter;
        var max = opts.max || MAX_MESSAGE;
        function updateCounter() {
            if (counter && input) {
                counter.textContent = input.value.length + "/" + max + " ký tự";
            }
        }
        if (input && form) {
            input.addEventListener("keydown", function (event) {
                if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    form.requestSubmit();
                }
            });
            input.addEventListener("input", updateCounter);
        }
        updateCounter();
        return {updateCounter: updateCounter};
    }

    function createEmptyStateToggle(transcript, emptyElement) {
        return function () {
            var hasMessage = !!transcript.querySelector(".chat-entry");
            emptyElement.hidden = hasMessage;
            transcript.classList.toggle("is-empty", !hasMessage);
        };
    }

    window.ChatClientKit = {
        SESSION_KEY: SESSION_KEY,
        MAX_MESSAGE: MAX_MESSAGE,
        MAX_TRANSCRIPT: MAX_TRANSCRIPT,
        loadSession: loadSession,
        restoreSession: restoreSession,
        clearSession: clearSession,
        sendMessage: sendMessage,
        enhanceComposer: enhanceComposer,
        createEmptyStateToggle: createEmptyStateToggle
    };
})();
