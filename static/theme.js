/* theme.js — chế độ sáng/tối dùng chung cho mọi trang.
   Dùng 2 phần:
   1) applyTheme() gọi NGAY trong <head> (trước khi body render) để chống nhấp nháy.
   2) Phần cuối tự chèn nút bật/tắt vào .nav-links sau khi DOM sẵn sàng.
   Lựa chọn lưu ở localStorage; nếu chưa chọn thì theo cài đặt hệ điều hành. */
(function () {
    var KEY = "hose-theme";

    function stored() {
        try { return localStorage.getItem(KEY); } catch (e) { return null; }
    }

    function systemPref() {
        return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
            ? "dark" : "light";
    }

    function resolved() {
        return stored() || systemPref();
    }

    // Áp theme lên <html data-theme>. Gọi sớm nhất có thể.
    function applyTheme(mode) {
        document.documentElement.setAttribute("data-theme", mode);
    }

    // Áp ngay lập tức (chạy trong <head>, trước khi body vẽ) → không nhấp nháy.
    applyTheme(resolved());

    // Nếu người dùng chưa chọn thủ công, đổi theo hệ điều hành khi họ đổi.
    if (window.matchMedia) {
        var mq = window.matchMedia("(prefers-color-scheme: dark)");
        var onChange = function () { if (!stored()) { applyTheme(systemPref()); syncButton(); } };
        if (mq.addEventListener) { mq.addEventListener("change", onChange); }
        else if (mq.addListener) { mq.addListener(onChange); }
    }

    var SUN = "☀";
    var MOON = "☾";

    function syncButton() {
        var btn = document.getElementById("theme-toggle");
        if (!btn) { return; }
        var dark = document.documentElement.getAttribute("data-theme") === "dark";
        btn.textContent = dark ? SUN : MOON;
        btn.setAttribute("aria-label", dark ? "Chuyển sang chế độ sáng" : "Chuyển sang chế độ tối");
        btn.setAttribute("title", dark ? "Chế độ sáng" : "Chế độ tối");
        btn.setAttribute("aria-pressed", dark ? "true" : "false");
    }

    function toggle() {
        var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
        applyTheme(next);
        try { localStorage.setItem(KEY, next); } catch (e) {}
        syncButton();
    }

    function mountButton() {
        var nav = document.querySelector(".nav-links");
        if (!nav || document.getElementById("theme-toggle")) { return; }
        var btn = document.createElement("button");
        btn.type = "button";
        btn.id = "theme-toggle";
        btn.className = "theme-toggle";
        btn.addEventListener("click", toggle);
        nav.appendChild(btn);
        syncButton();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", mountButton);
    } else {
        mountButton();
    }

    // Cho trang khác (vd Tuning Lab thay <main> bằng AJAX) gắn lại nút sau khi
    // DOM đổi — nút nằm trong nav thuộc <main> nên bị mất sau mỗi lần swap.
    window.mountThemeToggle = mountButton;
})();
