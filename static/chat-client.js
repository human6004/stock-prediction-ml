/* chat-client.js — session, transport và VIEW dùng chung cho trợ lý HOSE.

   MỤC ĐÍCH TÁI SỬ DỤNG (đọc trước khi sửa)
   ----------------------------------------
   Cả trang /chat và khung chat nổi dùng cùng session trong tab, transport và
   phần dựng DOM. Không hardcode id phần tử; template truyền tham chiếu vào.

   Session chỉ chứa transcript + conversation_state, tối đa 40 mục. Số liệu
   trong transcript chỉ làm ngữ cảnh hội thoại; backend vẫn dựng data context.

   RÀNG BUỘC BẢO MẬT — TUYỆT ĐỐI: chỉ dựng DOM bằng document.createElement +
   textContent, không bao giờ gán chuỗi HTML thô vào phần tử và không dùng API
   chèn HTML dạng chuỗi. Test grep thẳng vào source nên ngay cả tên các API đó
   cũng không được viết ra, kể cả trong comment. Nội dung câu trả lời do LLM
   sinh ra nên phải coi là dữ liệu không tin cậy: chỉ đi qua textContent.

   Cặp đôi CSS: static/chat-ui.css. */

(function () {
    "use strict";

    var MAX_MESSAGE = 1000;
    var SESSION_KEY = "hose-chat-session-v1";
    var MAX_TRANSCRIPT = 40;
    var REQUEST_TIMEOUT_MS = 70000;  // ≥ TOTAL_DEADLINE_SECONDS(60) + network margin
    // Còn <= 100 ký tự thì đổi sang màu cảnh báo và báo cho screen reader.
    var WARN_REMAINING = 100;
    // Chặn spam aria-live: chỉ đọc lại sau khoảng nghỉ này.
    var LIVE_THROTTLE_MS = 1200;

    var BULLET_RE = /^\s*[-*\u2022]\s+/;
    var ORDERED_RE = /^\s*\d+[.)]\s+/;
    var FENCE_RE = /^\s*```/;

    function clearNode(node) {
        while (node && node.firstChild) {
            node.removeChild(node.firstChild);
        }
    }

    function asText(value) {
        return value === null || value === undefined ? "" : String(value);
    }

    function emptySession() {
        return {transcript: [], conversation_state: null};
    }

    function normalizeStringArray(value, max) {
        if (!Array.isArray(value) || value.length > max) {
            throw new Error("invalid_session");
        }
        var normalized = [];
        value.forEach(function (item) {
            if (typeof item !== "string" || !item.trim()) {
                throw new Error("invalid_session");
            }
            var symbol = item.trim().toUpperCase();
            if (!/^[A-Z]{2,5}$/.test(symbol) || normalized.indexOf(symbol) !== -1) {
                throw new Error("invalid_session");
            }
            normalized.push(symbol);
        });
        return normalized;
    }

    function normalizeConversationState(value) {
        if (value === null || value === undefined) {
            return null;
        }
        if (!value || typeof value !== "object" || Array.isArray(value)) {
            throw new Error("invalid_session");
        }
        var stateKeys = [
            "active_symbols", "topic", "ranking_order", "last_result_symbols"
        ];
        if (Object.keys(value).length !== stateKeys.length
                || stateKeys.some(function (key) {
                    return !Object.prototype.hasOwnProperty.call(value, key);
                })) {
            throw new Error("invalid_session");
        }
        var topics = [
            "signal", "comparison", "ranking", "model", "dataset",
            "feature", "project", "limitations"
        ];
        var orders = ["highest_up_score", "lowest_up_score"];
        var topic = value.topic === null ? null : value.topic;
        var order = value.ranking_order === null ? null : value.ranking_order;
        if ((topic !== null && topics.indexOf(topic) === -1)
                || (order !== null && orders.indexOf(order) === -1)) {
            throw new Error("invalid_session");
        }
        return {
            active_symbols: normalizeStringArray(value.active_symbols, 2),
            topic: topic,
            ranking_order: order,
            last_result_symbols: normalizeStringArray(value.last_result_symbols, 10)
        };
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

    function normalizeEntry(entry, expectedRole) {
        if (!entry || entry.role !== expectedRole || typeof entry.content !== "string"
                || !entry.content.trim()) {
            throw new Error("invalid_session");
        }
        var clean = {
            role: expectedRole,
            content: entry.content.slice(0, MAX_MESSAGE)
        };
        if (expectedRole === "assistant") {
            clean.sources = normalizeSources(entry.sources);
            clean.warnings = normalizeWarnings(entry.warnings);
            clean.release_status = asText(entry.release_status).slice(0, 40) || null;
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
        var transcript = rows.map(function (entry, index) {
            return normalizeEntry(entry, index % 2 === 0 ? "user" : "assistant");
        });
        return {
            transcript: transcript,
            conversation_state: normalizeConversationState(value.conversation_state)
        };
    }

    function clearSession() {
        try {
            window.sessionStorage.removeItem(SESSION_KEY);
        } catch (error) {
            // Storage có thể bị browser chặn; UI vẫn dùng được trong lượt hiện tại.
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
            // Quota/private mode: không làm hỏng lượt chat đang chạy.
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
                release_status: asText(data.release_status).slice(0, 40) || null
            }
        );
        session.transcript = session.transcript.slice(-MAX_TRANSCRIPT);
        if (data.conversation_state !== undefined) {
            session.conversation_state = normalizeConversationState(data.conversation_state);
        }
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
        if (session.conversation_state) {
            payload.conversation_state = session.conversation_state;
        }

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
                // Lưu transcript là việc phụ; nếu nó lỗi (storage quota,
                // dữ liệu lạ) thì vẫn phải trả câu trả lời đã nhận được cho
                // người dùng, không được ném đi cả lượt chat 200 OK.
                try {
                    saveExchange(message, data);
                } catch (error) {
                    /* Không làm hỏng câu trả lời đã nhận. */
                }
                return data;
            });
        }).finally(function () {
            window.clearTimeout(timeout);
        });
    }

    function reducedMotion() {
        if (window.UIKit && typeof window.UIKit.prefersReducedMotion === "function") {
            return window.UIKit.prefersReducedMotion();
        }
        return !!(window.matchMedia
            && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    }

    /* ==========================================================
       1. Markdown-lite — dựng bằng DOM, không parse HTML
       ==========================================================
       Hỗ trợ đúng 4 thứ mà câu trả lời grounded thực sự dùng:
       - đoạn cách nhau bằng dòng trắng      -> <p>
       - dòng "- " / "* " / "• "             -> <ul><li>
       - dòng "1. " / "1) "                  -> <ol><li>
       - `code` inline và ```fenced```       -> <code> / <pre><code>
       Mọi ký tự khác giữ nguyên dạng text. */

    /** Ghép text + `code` inline vào target. */
    function appendInline(target, raw) {
        var parts = asText(raw).split("`");
        if (parts.length % 2 === 0) {
            // Số backtick lẻ -> cặp cuối không đóng. Gộp lại thành text thuần
            // để không biến phần đuôi câu trả lời thành khối code.
            var tail = parts.pop();
            parts[parts.length - 1] = parts[parts.length - 1] + "`" + tail;
        }
        for (var i = 0; i < parts.length; i += 1) {
            if (parts[i] === "") {
                continue;
            }
            if (i % 2 === 1) {
                var code = document.createElement("code");
                code.className = "chat-code";
                code.textContent = parts[i];
                target.appendChild(code);
            } else {
                target.appendChild(document.createTextNode(parts[i]));
            }
        }
    }

    /** Nhiều dòng liền nhau (không có dòng trắng) -> một <p>, ngắt bằng <br>. */
    function buildParagraph(lines) {
        var p = document.createElement("p");
        p.className = "chat-p";
        for (var i = 0; i < lines.length; i += 1) {
            if (i > 0) {
                p.appendChild(document.createElement("br"));
            }
            appendInline(p, lines[i]);
        }
        return p;
    }

    function buildList(lines, tagName) {
        var list = document.createElement(tagName);
        list.className = "chat-list";
        for (var i = 0; i < lines.length; i += 1) {
            var item = document.createElement("li");
            var stripped = tagName === "ol"
                ? lines[i].replace(ORDERED_RE, "")
                : lines[i].replace(BULLET_RE, "");
            appendInline(item, stripped);
            list.appendChild(item);
        }
        return list;
    }

    function buildCodeBlock(body, lang) {
        var pre = document.createElement("pre");
        pre.className = "chat-pre";
        pre.tabIndex = 0;
        // Khối code có thể tràn ngang -> cần focus được để cuộn bằng bàn phím.
        pre.setAttribute("role", "group");
        pre.setAttribute(
            "aria-label",
            lang ? "Khối mã " + lang : "Khối mã"
        );
        var code = document.createElement("code");
        code.textContent = body;
        pre.appendChild(code);
        return pre;
    }

    /**
     * Một "block" = các dòng liên tiếp không có dòng trắng. Trong block vẫn có
     * thể trộn câu dẫn + gạch đầu dòng ("Tóm tắt:\n- a\n- b") nên phải cắt tiếp
     * theo từng loại dòng.
     */
    function buildBlock(lines) {
        var frag = document.createDocumentFragment();
        var buffer = [];
        var mode = null;

        function flush() {
            if (buffer.length === 0) {
                return;
            }
            if (mode === "ul" || mode === "ol") {
                frag.appendChild(buildList(buffer, mode));
            } else {
                frag.appendChild(buildParagraph(buffer));
            }
            buffer = [];
        }

        for (var i = 0; i < lines.length; i += 1) {
            var next = "p";
            if (BULLET_RE.test(lines[i])) {
                next = "ul";
            } else if (ORDERED_RE.test(lines[i])) {
                next = "ol";
            }
            if (mode !== null && next !== mode) {
                flush();
            }
            mode = next;
            buffer.push(lines[i]);
        }
        flush();
        return frag;
    }

    /**
     * Render markdown-lite vào container (container bị xoá sạch trước).
     * @param {Element} container phần tử chứa (nên là div, vì có <ul>/<pre>)
     * @param {string} raw văn bản thô từ /api/chat
     */
    function renderRichText(container, raw) {
        if (!container) {
            return;
        }
        clearNode(container);
        // Bật .is-rich: nội dung giờ là <p>/<ul>/<pre> thật nên phải tắt
        // white-space: pre-wrap của .chat-text (app.css) kẻo bị double-spacing.
        container.classList.add("is-rich");
        var text = asText(raw).replace(/\r\n?/g, "\n");
        var lines = text.split("\n");
        var index = 0;
        var rendered = false;

        while (index < lines.length) {
            if (FENCE_RE.test(lines[index])) {
                var lang = lines[index].trim().slice(3).trim();
                var body = [];
                index += 1;
                while (index < lines.length && !FENCE_RE.test(lines[index])) {
                    body.push(lines[index]);
                    index += 1;
                }
                if (index < lines.length) {
                    index += 1; // bỏ dòng ``` đóng
                }
                container.appendChild(buildCodeBlock(body.join("\n"), lang));
                rendered = true;
                continue;
            }
            if (lines[index].trim() === "") {
                index += 1;
                continue;
            }
            var block = [];
            while (index < lines.length
                && lines[index].trim() !== ""
                && !FENCE_RE.test(lines[index])) {
                block.push(lines[index]);
                index += 1;
            }
            container.appendChild(buildBlock(block));
            rendered = true;
        }

        if (!rendered) {
            // Chuỗi rỗng / chỉ toàn khoảng trắng: vẫn giữ một <p> để layout ổn.
            var fallback = document.createElement("p");
            fallback.className = "chat-p";
            fallback.textContent = text.trim();
            container.appendChild(fallback);
        }
    }

    /* ==========================================================
       2. Chỉ báo đang trả lời (ba chấm)
       ========================================================== */

    /**
     * Ba chấm động. Chấm là trang trí (aria-hidden), phần chữ ẩn thị giác mới
     * là thứ screen reader đọc -> khi prefers-reduced-motion tắt animation thì
     * vẫn còn nghĩa "đang chờ" cả về hình (chấm mờ dần tĩnh, xem chat-ui.css)
     * lẫn về ngữ nghĩa.
     */
    function buildTypingIndicator() {
        var wrap = document.createElement("span");
        wrap.className = "chat-typing";

        var label = document.createElement("span");
        label.className = "visually-hidden";
        label.textContent = "Trợ lý đang trả lời…";
        wrap.appendChild(label);

        var dots = document.createElement("span");
        dots.className = "chat-typing-dots";
        dots.setAttribute("aria-hidden", "true");
        for (var i = 0; i < 3; i += 1) {
            dots.appendChild(document.createElement("span"));
        }
        wrap.appendChild(dots);
        return wrap;
    }

    /* ==========================================================
       3. Provenance theo từng câu trả lời (<details> "Nguồn dữ liệu")
       ==========================================================
       Đây là phần thay thế cái footer dùng chung bị ghi đè mỗi lượt: mỗi bubble
       của trợ lý tự mang nguồn của chính nó nên cả hội thoại vẫn truy vết được. */

    function metaRow(label, value) {
        var row = document.createElement("p");
        row.className = "chat-cite-row";
        var strong = document.createElement("strong");
        strong.textContent = label + ": ";
        row.appendChild(strong);
        var span = document.createElement("span");
        span.textContent = value;
        row.appendChild(span);
        return row;
    }

    function metaList(label, values, extraClass) {
        var wrap = document.createElement("div");
        wrap.className = "chat-cite-block" + (extraClass ? " " + extraClass : "");
        var strong = document.createElement("strong");
        strong.textContent = label + ":";
        wrap.appendChild(strong);
        var list = document.createElement("ul");
        list.className = "chat-cite-list";
        for (var i = 0; i < values.length; i += 1) {
            var item = document.createElement("li");
            item.textContent = values[i];
            list.appendChild(item);
        }
        wrap.appendChild(list);
        return wrap;
    }

    /** Chuẩn hoá một source của /api/chat thành một dòng đọc được. */
    function formatSource(source, sourceLabels) {
        var kind = (source && source.kind) || "artifact";
        var labels = sourceLabels || {};
        var text = labels[kind] || kind;
        if (source && source.symbols && source.symbols.length) {
            text += " " + source.symbols.join(", ");
        }
        if (source && source.as_of) {
            text += " @ " + source.as_of;
        }
        return text;
    }

    /**
     * Dựng <details> provenance cho một lượt trả lời.
     * @returns {Element|null} null khi không có gì để hiện.
     */
    function buildSourceDetails(data, labels) {
        var payload = data || {};
        var maps = labels || {};
        var releaseLabels = maps.releaseLabels || {};
        var sources = payload.sources || [];
        var warnings = payload.warnings || [];
        var releaseStatus = payload.release_status;

        if (!releaseStatus && sources.length === 0 && warnings.length === 0) {
            return null;
        }

        var details = document.createElement("details");
        details.className = "chat-cite";

        var summary = document.createElement("summary");
        summary.className = "chat-cite-summary";
        var summaryText = document.createElement("span");
        summaryText.textContent = "Nguồn dữ liệu";
        summary.appendChild(summaryText);

        var count = document.createElement("span");
        count.className = "chat-cite-count";
        if (sources.length) {
            count.textContent = String(sources.length);
            count.setAttribute("aria-label", sources.length + " nguồn");
        }
        summary.appendChild(count);

        if (warnings.length) {
            var flag = document.createElement("span");
            flag.className = "chat-cite-flag";
            flag.textContent = "cảnh báo";
            summary.appendChild(flag);
        }
        details.appendChild(summary);

        var body = document.createElement("div");
        body.className = "chat-cite-body";

        if (releaseStatus) {
            body.appendChild(metaRow(
                "Phiên bản model",
                releaseLabels[releaseStatus] || releaseStatus
            ));
        }
        if (sources.length) {
            var rows = [];
            for (var i = 0; i < sources.length; i += 1) {
                rows.push(formatSource(sources[i], maps.sourceLabels));
            }
            body.appendChild(metaList("Nguồn", rows));
        }
        if (warnings.length) {
            var messages = [];
            for (var j = 0; j < warnings.length; j += 1) {
                var warning = warnings[j];
                messages.push(
                    (warning && warning.message) ? warning.message : asText(warning)
                );
            }
            body.appendChild(metaList("Cảnh báo", messages, "chat-cite-warn"));
        }

        details.appendChild(body);
        return details;
    }

    /* ==========================================================
       4. Nút copy câu trả lời
       ==========================================================
       UIKit.copy() TỰ toast ("Đã copy vào clipboard." / thông báo lỗi) nên ở đây
       không toast thêm lần nữa để tránh hai toast trùng. Nhánh fallback chỉ chạy
       khi ui-kit.js vắng mặt. */

    function fallbackCopy(text) {
        if (!document.body) {
            return false;
        }
        var area = document.createElement("textarea");
        area.value = text;
        area.setAttribute("readonly", "readonly");
        area.className = "visually-hidden";
        document.body.appendChild(area);
        var ok = false;
        try {
            area.select();
            ok = document.execCommand("copy");
        } catch (error) {
            ok = false;
        }
        document.body.removeChild(area);
        return ok;
    }

    /**
     * @param {function(): string} getText trả về nội dung cần copy tại lúc bấm.
     */
    function buildCopyButton(getText) {
        var button = document.createElement("button");
        button.type = "button";
        button.className = "chat-copy";
        button.textContent = "Copy";
        button.setAttribute("aria-label", "Copy câu trả lời của trợ lý");
        button.addEventListener("click", function () {
            var text = "";
            try {
                text = asText(getText());
            } catch (error) {
                text = "";
            }
            if (window.UIKit && typeof window.UIKit.copy === "function") {
                window.UIKit.copy(text);
                return;
            }
            var ok = fallbackCopy(text);
            if (window.UIKit && typeof window.UIKit.toast === "function") {
                window.UIKit.toast(
                    ok ? "Đã copy câu trả lời." : "Không copy được. Vui lòng copy thủ công.",
                    {type: ok ? "success" : "error"}
                );
            }
        });
        return button;
    }

    /* ==========================================================
       5. Hành vi ô nhập
       ========================================================== */

    /** Enter gửi, Shift+Enter xuống dòng (khớp hành vi của chat dock). */
    function bindEnterToSend(input, form) {
        if (!input || !form) {
            return;
        }
        input.addEventListener("keydown", function (event) {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                if (typeof form.requestSubmit === "function") {
                    form.requestSubmit();
                } else {
                    // Safari cũ: không có requestSubmit -> phát submit thủ công.
                    form.dispatchEvent(new Event("submit", {
                        bubbles: true,
                        cancelable: true
                    }));
                }
            }
        });
    }

    /**
     * Bộ đếm ký tự cho textarea có maxlength.
     *
     * Chiến lược a11y (chủ ý, đừng "đơn giản hoá" thành một aria-live duy nhất):
     * - `counter` cập nhật MỖI lần gõ nhưng KHÔNG phải live region. Nó được nối
     *   vào textarea bằng aria-describedby -> screen reader đọc theo yêu cầu,
     *   không bị đọc chen ngang từng ký tự.
     * - `liveRegion` (ẩn thị giác, aria-live="polite") chỉ phát khi VƯỢT NGƯỠNG
     *   (vào/ra vùng còn <=100, và lúc đầy) + throttle 1.2s. Tối đa vài lần
     *   thông báo cho cả một lần soạn tin, thay vì 1000 lần.
     */
    function bindCharCounter(input, options) {
        var opts = options || {};
        var counter = opts.counter;
        var live = opts.liveRegion || null;
        var max = typeof opts.max === "number" ? opts.max : MAX_MESSAGE;
        var warnAt = typeof opts.warnAt === "number" ? opts.warnAt : WARN_REMAINING;
        if (!input || !counter) {
            return function () {};
        }

        var wasWarning = false;
        var wasFull = false;
        var lastAnnounce = 0;

        function announce(text) {
            if (!live) {
                return;
            }
            var now = Date.now();
            if (now - lastAnnounce < LIVE_THROTTLE_MS) {
                return;
            }
            lastAnnounce = now;
            live.textContent = text;
        }

        function update() {
            var used = input.value.length;
            var remaining = max - used;
            counter.textContent = used + "/" + max;
            var isWarning = remaining <= warnAt;
            var isFull = remaining <= 0;
            counter.classList.toggle("is-warn", isWarning && !isFull);
            counter.classList.toggle("is-full", isFull);

            if (isFull && !wasFull) {
                announce("Đã đạt giới hạn " + max + " ký tự.");
            } else if (isWarning && !wasWarning) {
                announce("Còn " + remaining + " ký tự.");
            } else if (!isWarning && wasWarning) {
                announce("");
            }
            wasWarning = isWarning;
            wasFull = isFull;
        }

        input.addEventListener("input", update);
        update();
        return update;
    }

    /* ==========================================================
       6. Empty state + stagger lần đầu
       ========================================================== */

    /**
     * Ẩn khối onboarding ngay khi có tin nhắn thật. Khối empty state phải nằm
     * TRONG transcript và mang class được truyền vào (mặc định .chat-empty) để
     * hàm này phân biệt được với bubble thật.
     */
    function createEmptyStateToggle(transcript, emptyElement) {
        if (!transcript || !emptyElement) {
            return function () {};
        }
        return function () {
            var hasMessage = !!transcript.querySelector(".chat-entry");
            emptyElement.hidden = hasMessage;
            // Modifier để CSS căn giữa phần onboarding khi transcript còn rỗng.
            transcript.classList.toggle("is-empty", !hasMessage);
        };
    }

    /**
     * Stagger CHỈ cho lượt render đầu tiên. Gọi lại nhiều lần cũng chỉ tác dụng
     * một lần: nếu set --i mỗi lần append thì bubble thứ 8 sẽ bị delay ~224ms,
     * đọc ra thành "lag" chứ không phải "mượt".
     */
    function createStaggerOnce(transcript, selector) {
        var done = false;
        return function () {
            if (done || !transcript || reducedMotion()) {
                return;
            }
            done = true;
            if (window.UIKit && typeof window.UIKit.stagger === "function") {
                window.UIKit.stagger(transcript, selector || ":scope > .chat-entry", 8);
            } else {
                transcript.classList.add("stagger-in");
            }
        };
    }

    /* ==========================================================
       7. Gói tiện lợi — điểm vào một-lần-gọi cho dock sau này
       ========================================================== */

    /**
     * Nối các hành vi ô nhập trong một lần gọi.
     * @param {Object} options {form, input, counter, liveRegion, max, warnAt}
     * @returns {Object} {updateCounter}
     */
    function enhanceComposer(options) {
        var opts = options || {};
        bindEnterToSend(opts.input, opts.form);
        var updateCounter = bindCharCounter(opts.input, opts);
        if (opts.input && opts.counter && opts.counter.id) {
            var described = opts.input.getAttribute("aria-describedby") || "";
            if (described.indexOf(opts.counter.id) === -1) {
                opts.input.setAttribute(
                    "aria-describedby",
                    (described + " " + opts.counter.id).trim()
                );
            }
        }
        return {updateCounter: updateCounter};
    }

    window.ChatClientKit = {
        renderRichText: renderRichText,
        buildTypingIndicator: buildTypingIndicator,
        buildSourceDetails: buildSourceDetails,
        buildCopyButton: buildCopyButton,
        formatSource: formatSource,
        bindEnterToSend: bindEnterToSend,
        bindCharCounter: bindCharCounter,
        enhanceComposer: enhanceComposer,
        createEmptyStateToggle: createEmptyStateToggle,
        createStaggerOnce: createStaggerOnce,
        loadSession: loadSession,
        restoreSession: restoreSession,
        clearSession: clearSession,
        sendMessage: sendMessage,
        apiErrorMessage: apiErrorMessage,
        MAX_MESSAGE: MAX_MESSAGE,
        MAX_TRANSCRIPT: MAX_TRANSCRIPT,
        SESSION_KEY: SESSION_KEY
    };
})();
