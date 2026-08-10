/* ui-kit.js — lớp primitive dùng chung cho mọi trang (toast, busy, elapsed,
   countUp, stagger, copy). Nạp SAU app.css/ui-kit.css nhưng an toàn ở cả <head>
   lẫn cuối <body>: mọi truy cập DOM đều đi qua onReady().

   Không phụ thuộc thư viện, không build step, không ES module. Giữ đúng style
   của theme.js / table-sort.js: var + function, thụt 4 space.

   KHÔNG dùng innerHTML cho nội dung động — mọi text đi qua textContent để
   không mở đường cho HTML injection (cùng kỷ luật với chat dock).

   API công khai: window.UIKit — các trang khác code theo đúng chữ ký này. */
(function () {
    var TOAST_STACK_ID = "toast-stack";
    var TOAST_MAX = 4;
    var TOAST_DURATION = 4200;
    var STAGGER_MAX = 14;

    /* ---------- tiện ích chung ---------- */

    function prefersReducedMotion() {
        return !!(window.matchMedia
            && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    }

    function onReady(fn) {
        if (typeof fn !== "function") { return; }
        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", fn);
        } else {
            fn();
        }
    }

    function noop() {}

    // Gọi fn một lần duy nhất: dùng cho dọn dẹp (animationend + timeout đua nhau).
    function once(fn) {
        var done = false;
        return function () {
            if (done) { return; }
            done = true;
            fn();
        };
    }

    /* ---------- toast ---------- */

    function toastStack() {
        var stack = document.getElementById(TOAST_STACK_ID);
        if (stack) { return stack; }
        if (!document.body) { return null; }
        // base.html render sẵn khung này; nhánh dưới chỉ là lưới an toàn.
        stack = document.createElement("div");
        stack.id = TOAST_STACK_ID;
        stack.className = "toast-stack";
        stack.setAttribute("role", "status");
        stack.setAttribute("aria-live", "polite");
        stack.setAttribute("aria-atomic", "false");
        document.body.appendChild(stack);
        return stack;
    }

    var TOAST_TYPES = {
        success: "toast-success",
        warn: "toast-warn",
        error: "toast-error",
        info: "toast-info"
    };

    function toast(message, options) {
        var opts = options || {};
        var stack = toastStack();
        if (!stack) { return noop; }

        var type = TOAST_TYPES[opts.type] ? opts.type : "info";
        var duration = typeof opts.duration === "number" ? opts.duration : TOAST_DURATION;

        var item = document.createElement("div");
        item.className = "toast " + TOAST_TYPES[type];
        item.tabIndex = -1;
        if (type === "error") {
            // Lỗi cần được đọc ngay, không đợi hết câu đang đọc.
            item.setAttribute("role", "alert");
        }

        var accent = document.createElement("span");
        accent.className = "toast-accent";
        accent.setAttribute("aria-hidden", "true");

        var body = document.createElement("div");
        body.className = "toast-body";
        if (opts.title) {
            var title = document.createElement("p");
            title.className = "toast-title";
            title.textContent = String(opts.title);
            body.appendChild(title);
        }
        var msg = document.createElement("p");
        msg.className = "toast-msg";
        msg.textContent = message === null || message === undefined ? "" : String(message);
        body.appendChild(msg);

        var close = document.createElement("button");
        close.type = "button";
        close.className = "toast-close";
        close.setAttribute("aria-label", "Đóng thông báo");
        close.textContent = "✕";

        item.appendChild(accent);
        item.appendChild(body);
        item.appendChild(close);

        var timer = null;
        var remove = once(function () {
            if (timer) { window.clearTimeout(timer); timer = null; }
            if (item.parentNode) { item.parentNode.removeChild(item); }
        });

        function dismiss() {
            if (!item.parentNode || item.classList.contains("is-leaving")) { return; }
            if (timer) { window.clearTimeout(timer); timer = null; }
            item.classList.add("is-leaving");
            var finish = once(remove);
            item.addEventListener("animationend", finish);
            // Dự phòng khi animation bị tắt (reduced motion) → animationend không nổ.
            window.setTimeout(finish, 400);
        }

        close.addEventListener("click", dismiss);
        item.addEventListener("keydown", function (e) {
            if (e.key === "Escape") {
                e.stopPropagation();
                dismiss();
            }
        });

        stack.appendChild(item);

        // Quá 4 toast thì bỏ cái cũ nhất để góc màn hình không bị phủ kín.
        var live = stack.querySelectorAll(".toast:not(.is-leaving)");
        for (var i = 0; i < live.length - TOAST_MAX; i += 1) {
            if (live[i].parentNode) { live[i].parentNode.removeChild(live[i]); }
        }

        if (duration > 0) {
            timer = window.setTimeout(dismiss, duration);
        }
        return dismiss;
    }

    /* ---------- busy / lockForm ---------- */

    function busy(el, label) {
        if (!el || el.dataset.uiBusy === "1") { return noop; }
        el.dataset.uiBusy = "1";
        // Giữ nguyên các node con thật (không đi qua innerHTML): khôi phục bằng
        // cách gắn lại chính node cũ nên listener bên trong nút không bị mất.
        var original = Array.prototype.slice.call(el.childNodes);
        var wasDisabled = el.disabled;

        el.classList.add("is-loading");
        el.setAttribute("aria-busy", "true");

        if (label) {
            // Thay nhãn: dựng lại bằng textContent, không nhét chuỗi vào innerHTML.
            while (el.firstChild) { el.removeChild(el.firstChild); }
            var text = document.createElement("span");
            text.className = "ui-busy-label";
            text.textContent = String(label);
            el.appendChild(text);
        }

        var spinner = document.createElement("span");
        spinner.className = "spinner";
        spinner.setAttribute("aria-hidden", "true");
        el.insertBefore(spinner, el.firstChild);

        el.disabled = true;

        return once(function () {
            el.classList.remove("is-loading");
            el.removeAttribute("aria-busy");
            el.disabled = wasDisabled;
            // Gắn lại chính các node cũ (append node đã tồn tại là "move"), nên
            // nội dung và listener con trở lại nguyên trạng. Không dùng innerHTML.
            while (el.firstChild) { el.removeChild(el.firstChild); }
            original.forEach(function (node) { el.appendChild(node); });
            delete el.dataset.uiBusy;
        });
    }

    function submitButton(form, target) {
        if (!target) {
            return form.querySelector('[type="submit"]')
                || form.querySelector("button:not([type])");
        }
        if (typeof target === "string") { return form.querySelector(target); }
        return target;
    }

    function lockForm(form, opts) {
        if (!form || form.dataset.uiLockBound === "1") { return; }
        form.dataset.uiLockBound = "1";
        var options = opts || {};
        var release = noop;

        form.addEventListener("submit", function () {
            // Form không hợp lệ: trình duyệt chặn submit → không được khoá nút.
            if (typeof form.checkValidity === "function" && !form.checkValidity()) {
                return;
            }
            var btn = submitButton(form, options.button);
            if (!btn) { return; }
            release = busy(btn, options.label);
            form.setAttribute("aria-busy", "true");
        });

        // Bấm Back (bfcache): trang trở lại y nguyên, nút vẫn đang disabled →
        // mở lại để người dùng không mắc kẹt với nút chết.
        window.addEventListener("pageshow", function () {
            release();
            release = noop;
            form.removeAttribute("aria-busy");
        });
    }

    /* ---------- elapsed ---------- */

    function pad2(n) {
        return (n < 10 ? "0" : "") + n;
    }

    function formatElapsed(totalSeconds) {
        var s = Math.max(0, Math.floor(totalSeconds));
        var hours = Math.floor(s / 3600);
        var minutes = Math.floor((s % 3600) / 60);
        var seconds = s % 60;
        if (hours > 0) {
            return hours + ":" + pad2(minutes) + ":" + pad2(seconds);
        }
        return minutes + ":" + pad2(seconds);
    }

    function elapsed(el, isoStart) {
        if (!el || !isoStart) { return noop; }
        var start = Date.parse(isoStart);
        if (isNaN(start)) { return noop; }

        function tick() {
            el.textContent = formatElapsed((Date.now() - start) / 1000);
        }
        tick();
        var id = window.setInterval(tick, 1000);
        return once(function () { window.clearInterval(id); });
    }

    /* ---------- countUp ---------- */

    function currentNumber(el) {
        var raw = (el.textContent || "").replace(/[^0-9.+-]/g, "");
        var n = parseFloat(raw);
        return isFinite(n) ? n : 0;
    }

    function markUpdated(el) {
        el.classList.remove("just-updated");
        // Đọc offsetWidth để buộc reflow, nếu không class thêm lại ngay sẽ không
        // khởi động lại animation.
        void el.offsetWidth;
        el.classList.add("just-updated");
        window.setTimeout(function () { el.classList.remove("just-updated"); }, 600);
    }

    function countUp(el, to, opts) {
        if (!el || typeof to !== "number" || !isFinite(to)) { return; }
        var options = opts || {};
        var duration = typeof options.duration === "number" ? options.duration : 650;
        var decimals = typeof options.decimals === "number" ? options.decimals : 2;
        var suffix = options.suffix || "";
        var prefix = options.prefix || "";

        function write(value) {
            el.textContent = prefix + value.toFixed(decimals) + suffix;
        }

        if (prefersReducedMotion() || duration <= 0 || !window.requestAnimationFrame) {
            write(to);
            markUpdated(el);
            return;
        }

        var from = currentNumber(el);
        var startedAt = null;

        function frame(now) {
            if (startedAt === null) { startedAt = now; }
            var t = Math.min(1, (now - startedAt) / duration);
            var eased = 1 - Math.pow(1 - t, 3); // ease-out cubic
            write(from + (to - from) * eased);
            if (t < 1) {
                window.requestAnimationFrame(frame);
            } else {
                write(to);
                markUpdated(el);
            }
        }
        window.requestAnimationFrame(frame);
    }

    /* ---------- stagger ---------- */

    function stagger(container, selector, max) {
        if (!container || prefersReducedMotion()) { return; }
        var limit = typeof max === "number" && max > 0 ? max : STAGGER_MAX;
        var query = selector || ":scope > *";
        var nodes;
        try {
            nodes = container.querySelectorAll(query);
        } catch (e) {
            // :scope không được hỗ trợ (hoặc selector sai) → lùi về children.
            nodes = container.children;
        }
        var count = Math.min(nodes.length, limit);
        for (var i = 0; i < count; i += 1) {
            nodes[i].style.setProperty("--i", String(i));
        }
        container.classList.add("stagger-in");
    }

    /* ---------- copy ---------- */

    function legacyCopy(text) {
        if (!document.body) { return false; }
        var area = document.createElement("textarea");
        area.value = text;
        area.setAttribute("readonly", "readonly");
        area.style.position = "fixed";
        area.style.top = "-1000px";
        area.style.opacity = "0";
        document.body.appendChild(area);
        var ok = false;
        try {
            area.select();
            ok = document.execCommand("copy");
        } catch (e) {
            ok = false;
        }
        document.body.removeChild(area);
        return ok;
    }

    function copy(text) {
        var value = text === null || text === undefined ? "" : String(text);

        function done(ok) {
            if (ok) {
                toast("Đã copy vào clipboard.", {type: "success", duration: 2200});
            } else {
                toast("Không copy được. Vui lòng chọn và copy thủ công.", {type: "error"});
            }
            return ok;
        }

        if (navigator.clipboard && navigator.clipboard.writeText) {
            return navigator.clipboard.writeText(value).then(function () {
                return done(true);
            }, function () {
                return done(legacyCopy(value));
            });
        }
        return Promise.resolve(done(legacyCopy(value)));
    }

    /* ---------- auto-wiring ---------- */

    // Mỗi tính năng bọc try/catch riêng: một data-attribute sai không được phép
    // làm chết cả phần còn lại.
    function guard(fn) {
        try {
            fn();
        } catch (e) {
            if (window.console && window.console.warn) {
                window.console.warn("UIKit auto-wire lỗi:", e);
            }
        }
    }

    function each(selector, fn) {
        var nodes = document.querySelectorAll(selector);
        for (var i = 0; i < nodes.length; i += 1) {
            fn(nodes[i]);
        }
    }

    function wireLockForms() {
        each("form[data-ui-lock]", function (form) {
            lockForm(form, {label: form.dataset.uiLock || undefined});
        });
    }

    // Flash message render từ server: chỉ biến thành toast khi phần tử vốn đã bị
    // ẩn, tránh nội dung hiện hai lần (một trong luồng trang, một trong toast).
    function wireServerToasts() {
        each("[data-toast][data-toast-type]", function (el) {
            if (el.dataset.toastFired === "1") { return; }
            var hidden = el.hasAttribute("hidden")
                || el.classList.contains("visually-hidden");
            if (!hidden) { return; }
            el.dataset.toastFired = "1";
            toast(el.dataset.toast, {
                type: el.dataset.toastType,
                title: el.dataset.toastTitle || undefined
            });
        });
    }

    function wireElapsed() {
        each("[data-elapsed]", function (el) {
            if (el.dataset.elapsedBound === "1") { return; }
            el.dataset.elapsedBound = "1";
            elapsed(el, el.dataset.elapsed);
        });
    }

    function wireCountUp() {
        each("[data-count-up]", function (el) {
            if (el.dataset.countUpBound === "1") { return; }
            el.dataset.countUpBound = "1";
            var decimals = el.dataset.countUpDecimals;
            countUp(el, parseFloat(el.dataset.countUp), {
                decimals: decimals === undefined || decimals === "" ? 2 : +decimals,
                suffix: el.dataset.countUpSuffix || ""
            });
        });
    }

    function wireStagger() {
        each("[data-stagger]", function (el) {
            if (el.dataset.staggerBound === "1") { return; }
            el.dataset.staggerBound = "1";
            stagger(el, el.dataset.stagger || ":scope > *", +(el.dataset.staggerMax || STAGGER_MAX));
        });
    }

    /* Bảng rộng min-width:980px nằm trong .table-wrap overflow-x:auto — chuột kéo
       được, bàn phím thì không (vùng cuộn không focus được). Gắn tabindex+role khi
       thực sự tràn để mũi tên cuộn được ngang. */
    function scrollLabel(el) {
        var caption = el.querySelector("caption");
        if (caption && caption.textContent.trim()) {
            return caption.textContent.trim();
        }
        var prev = el.previousElementSibling;
        while (prev) {
            if (/^H[1-6]$/.test(prev.tagName) && prev.textContent.trim()) {
                return prev.textContent.trim();
            }
            prev = prev.previousElementSibling;
        }
        return "Bảng dữ liệu, dùng mũi tên để cuộn ngang";
    }

    function syncTableScroll() {
        each('.table-wrap, [data-ui~="table-scroll"]', function (el) {
            if (el.scrollWidth > el.clientWidth) {
                if (el.dataset.uiScrollable === "1") { return; }
                el.dataset.uiScrollable = "1";
                el.setAttribute("tabindex", "0");
                el.setAttribute("role", "region");
                el.setAttribute("aria-label", scrollLabel(el));
            } else if (el.dataset.uiScrollable === "1") {
                // Không còn tràn → bỏ điểm dừng tab rỗng khỏi thứ tự bàn phím.
                delete el.dataset.uiScrollable;
                el.removeAttribute("tabindex");
                el.removeAttribute("role");
                el.removeAttribute("aria-label");
            }
        });
    }

    function wireTableScroll() {
        syncTableScroll();
        var timer = null;
        window.addEventListener("resize", function () {
            if (timer) { window.clearTimeout(timer); }
            timer = window.setTimeout(function () {
                guard(syncTableScroll);
            }, 150);
        });
    }

    function autoWire() {
        guard(wireLockForms);
        guard(wireServerToasts);
        guard(wireElapsed);
        guard(wireCountUp);
        guard(wireStagger);
        guard(wireTableScroll);
    }

    window.UIKit = {
        toast: toast,
        busy: busy,
        lockForm: lockForm,
        elapsed: elapsed,
        countUp: countUp,
        stagger: stagger,
        copy: copy,
        prefersReducedMotion: prefersReducedMotion,
        onReady: onReady,
        // Cho trang nào thay DOM bằng AJAX (Tuning Lab) gắn lại sau khi swap.
        autoWire: autoWire
    };

    onReady(autoWire);
})();
