/* chat-dock.js — khung chat nổi dùng trên mọi trang trừ /chat.
   Dùng lại ChatClientKit (transport + sessionStorage) của trang /chat, chỉ khác
   phần render: bong bóng gọn, không hiện danh sách nguồn (bấm ⤢ để xem đầy đủ).
   Mọi nội dung từ server chỉ ghi vào DOM qua textContent, không dựng HTML thô. */
(function () {
    "use strict";

    var kit = window.ChatClientKit;
    var dock = document.getElementById("chat-dock");
    if (!kit || !dock) {
        return;
    }

    var panel = document.getElementById("chat-dock-panel");
    var toggle = document.getElementById("chat-dock-toggle");
    var closeBtn = document.getElementById("chat-dock-close");
    var clearBtn = document.getElementById("chat-dock-clear");
    var transcript = document.getElementById("chat-dock-transcript");
    var empty = document.getElementById("chat-dock-empty");
    var form = document.getElementById("chat-dock-form");
    var input = document.getElementById("chat-dock-message");
    var sendBtn = document.getElementById("chat-dock-send");
    if (!panel || !toggle || !transcript || !form || !input) {
        return;
    }

    var busy = false;

    function syncEmpty() {
        if (empty) {
            empty.hidden = !!transcript.querySelector(".chat-dock-entry");
        }
    }

    /* Mốc dữ liệu + cảnh báo gộp thành một dòng chú thích dưới câu trả lời. */
    function buildNote(data) {
        var meta = [];
        if (data.data_as_of) {
            meta.push("Dữ liệu " + data.data_as_of);
        }
        if (data.model_trained_through) {
            meta.push("Model " + data.model_trained_through);
        }
        var lines = meta.length ? [meta.join(" · ")] : [];
        (Array.isArray(data.warnings) ? data.warnings : []).forEach(function (warning) {
            var message = warning && warning.message ? String(warning.message).trim() : "";
            if (message) {
                lines.push("⚠ " + message);
            }
        });
        return lines.join("\n");
    }

    function appendEntry(role, content, note) {
        var item = document.createElement("li");
        item.className = "chat-dock-entry chat-dock-entry-" + role;
        var text = document.createElement("p");
        text.className = "chat-dock-text";
        text.textContent = content;
        item.appendChild(text);
        if (note) {
            var meta = document.createElement("p");
            meta.className = "chat-dock-note";
            meta.textContent = note;
            item.appendChild(meta);
        }
        transcript.appendChild(item);
        transcript.scrollTop = transcript.scrollHeight;
        syncEmpty();
        return item;
    }

    function appendPending() {
        var item = appendEntry("assistant", "Đang trả lời…", "");
        item.classList.add("is-pending");
        return item;
    }

    function setBusy(value) {
        busy = value;
        input.disabled = value;
        if (sendBtn) {
            sendBtn.disabled = value;
        }
        if (clearBtn) {
            clearBtn.disabled = value;
        }
        form.setAttribute("aria-busy", value ? "true" : "false");
    }

    function setOpen(open) {
        dock.dataset.open = open ? "true" : "false";
        panel.hidden = !open;
        toggle.setAttribute("aria-expanded", open ? "true" : "false");
        toggle.setAttribute("aria-label", open ? "Đóng trợ lý HOSE" : "Mở trợ lý HOSE");
        if (open && !busy) {
            input.focus();
        }
    }

    kit.restoreSession(function (entry) {
        appendEntry(entry.role, entry.content, entry.role === "assistant" ? buildNote(entry) : "");
    });
    syncEmpty();
    kit.enhanceComposer({form: form, input: input});

    toggle.addEventListener("click", function () {
        setOpen(dock.dataset.open !== "true");
    });

    if (closeBtn) {
        closeBtn.addEventListener("click", function () {
            setOpen(false);
            toggle.focus();
        });
    }

    if (clearBtn) {
        clearBtn.addEventListener("click", function () {
            if (busy) {
                return;
            }
            kit.clearSession();
            while (transcript.firstChild) {
                transcript.removeChild(transcript.firstChild);
            }
            syncEmpty();
            input.focus();
        });
    }

    form.addEventListener("submit", function (event) {
        event.preventDefault();
        if (busy) {
            return;
        }
        var message = input.value.trim();
        if (!message) {
            return;
        }
        appendEntry("user", message, "");
        input.value = "";
        setBusy(true);
        var pending = appendPending();
        kit.sendMessage(message).then(function (data) {
            transcript.removeChild(pending);
            appendEntry("assistant", data.answer, buildNote(data));
        }).catch(function (error) {
            transcript.removeChild(pending);
            var reason = error && error.publicMessage
                ? error.publicMessage
                : "Không thể gọi trợ lý lúc này. Thử lại sau.";
            appendEntry("assistant", reason, "");
        }).finally(function () {
            setBusy(false);
            syncEmpty();
            input.focus();
        });
    });

    /* Link "Trợ lý" trên nav mở dock tại chỗ; không JS thì vẫn điều hướng /chat. */
    document.querySelectorAll('[data-nav="chat"]').forEach(function (link) {
        link.addEventListener("click", function (event) {
            event.preventDefault();
            setOpen(true);
        });
    });

    document.addEventListener("keydown", function (event) {
        if (event.key === "Escape" && dock.dataset.open === "true") {
            setOpen(false);
            toggle.focus();
        }
    });
})();
