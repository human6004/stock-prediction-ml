/* theme.js — chế độ sáng/tối + menu điều hướng mobile, dùng chung mọi trang.
   Dùng 2 phần:
   1) applyTheme() gọi NGAY trong <head> (trước khi body render) để chống nhấp nháy.
   2) Sau khi DOM sẵn sàng: gắn listener cho nút #theme-toggle (render sẵn từ
      base.html) và nút #nav-toggle của sidebar mobile.
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

    // Nút #theme-toggle nay render sẵn từ base.html (không còn tạo bằng JS).
    // Ở đây chỉ tìm nút có sẵn rồi gắn listener một lần (dataset.bound chống
    // gắn trùng khi gọi lại sau AJAX swap) và đồng bộ icon.
    function mountButton() {
        var btn = document.getElementById("theme-toggle");
        if (!btn) { return; }
        if (!btn.dataset.bound) {
            btn.addEventListener("click", toggle);
            btn.dataset.bound = "1";
        }
        syncButton();
    }

    // Menu điều hướng dạng đóng/mở trên mobile. Sidebar luôn trong DOM; nút
    // #nav-toggle chỉ bật/tắt trạng thái mở và hỗ trợ Escape trả focus.
    function mountNav() {
        var toggle = document.getElementById("nav-toggle");
        var shell = document.getElementById("app-shell");
        if (!toggle || !shell || toggle.dataset.bound) { return; }
        toggle.dataset.bound = "1";

        function setOpen(open) {
            shell.classList.toggle("nav-open", open);
            toggle.setAttribute("aria-expanded", open ? "true" : "false");
            toggle.setAttribute(
                "aria-label", open ? "Đóng menu điều hướng" : "Mở menu điều hướng"
            );
        }

        toggle.addEventListener("click", function () {
            setOpen(!shell.classList.contains("nav-open"));
        });
        document.addEventListener("keydown", function (e) {
            if (e.key === "Escape" && shell.classList.contains("nav-open")) {
                setOpen(false);
                toggle.focus();
            }
        });
    }

    function mount() {
        mountButton();
        mountNav();
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", mount);
    } else {
        mount();
    }

    // Cho Tuning Lab gắn lại nút theme sau mỗi lần swap #main-content (nút theme
    // nằm trong shell, ngoài vùng swap, nên chỉ cần đồng bộ icon là đủ).
    window.mountThemeToggle = mountButton;
})();
