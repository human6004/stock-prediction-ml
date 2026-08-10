/* page-signal.js — hành vi dùng chung cho 2 trang tín hiệu: "Dự báo một mã"
   (index.html) và "So sánh hai mã" (compare.html).

   Bổ sung cho window.UIKit, không thay thế: UIKit lo primitive (toast, busy,
   lockForm, stagger), file này lo phần riêng của 2 trang — chọn nhanh mã, đọc
   token màu cho Chart.js, và vẽ lại chart khi đổi theme.

   Nạp SAU ui-kit.js. Mọi node dựng bằng createElement + textContent, không gán
   chuỗi HTML thô. Style theo repo: var + function, thụt 4 space.

   API công khai: window.PageSignal */
(function () {
    /* ---------- đọc token màu ----------
       Chart.js vẽ trên canvas nên không "thấy" CSS var → phải resolve token ra
       chuỗi màu tại thời điểm dựng chart. Nhờ vậy dark mode vẫn đúng mà template
       không cần hardcode hex nào. */

    function cssVar(name, fallback) {
        var root = document.documentElement;
        var value = "";
        if (window.getComputedStyle) {
            value = window.getComputedStyle(root).getPropertyValue(name);
        }
        value = (value || "").trim();
        return value || fallback || "";
    }

    /* --up-rgb… là 3 kênh rời ("11 107 67") → dựng rgba() để canvas nhận được
       alpha ở mọi trình duyệt, không phụ thuộc cú pháp rgb(x y z / a). */
    function alphaFrom(varName, alpha, fallback) {
        var raw = cssVar(varName, "");
        var parts = raw.split(/[\s,]+/).filter(function (p) { return p !== ""; });
        if (parts.length < 3) { return fallback || "transparent"; }
        return "rgba(" + parts[0] + ", " + parts[1] + ", " + parts[2] + ", " + alpha + ")";
    }

    /* Fallback dùng system color (canvastext/canvas/graytext) chứ KHÔNG hardcode
       hex: nếu app.css chưa nạp thì trình duyệt vẫn tự chọn màu đúng theo chế độ
       sáng/tối, không có nguy cơ mực đen trên nền đen. Mất phân biệt UP/ngưỡng ở
       nhánh này không sao vì hai đường còn khác nhau bằng nét liền / nét đứt. */
    function palette() {
        return {
            ink: cssVar("--ink", "canvastext"),
            inkSoft: cssVar("--ink-soft", "canvastext"),
            muted: cssVar("--muted", "graytext"),
            faint: cssVar("--faint", "graytext"),
            line: cssVar("--line", "graytext"),
            lineSoft: cssVar("--line-soft", "graytext"),
            paper2: cssVar("--paper-2", "canvas"),
            panel: cssVar("--panel", "canvas"),
            up: cssVar("--up", "canvastext"),
            warn: cssVar("--warn", "canvastext"),
            down: cssVar("--down", "canvastext"),
            upSoft: alphaFrom("--up-rgb", 0.22, "transparent"),
            upFaint: alphaFrom("--up-rgb", 0.02, "transparent"),
            inkFaint: alphaFrom("--up-rgb", 0.05, "transparent")
        };
    }

    /* Gradient dọc dưới đường giá: đậm ở đỉnh, tan dần xuống đáy vùng vẽ.
       chartArea chưa có ở lần dựng đầu (layout chưa đo) → trả màu phẳng, Chart.js
       sẽ gọi lại hàm này sau khi có area. */
    function areaGradient(chart, varName) {
        var area = chart.chartArea;
        if (!area || !chart.ctx) { return alphaFrom(varName, 0.12, "transparent"); }
        var grad = chart.ctx.createLinearGradient(0, area.top, 0, area.bottom);
        grad.addColorStop(0, alphaFrom(varName, 0.26, "transparent"));
        grad.addColorStop(1, alphaFrom(varName, 0.01, "transparent"));
        return grad;
    }

    /* Phần "khung" của chart: tick, grid, tiêu đề trục, viền trục, legend,
       tooltip. Không đặt thì Chart.js dùng mặc định THƯ VIỆN (#666 cho chữ,
       rgba(0,0,0,0.1) cho grid) — hằng số, không theo theme. Ở dark mode grid
       10% đen gần như vô hình trên nền --paper #1a1713 và tick #666 tương phản
       rất thấp.

       Một hàm dùng chung cho index.html + compare.html thay vì lặp literal ở
       từng template. Phải gọi HAI lần: một khi dựng, một trong onThemeChange —
       Chart.js đã copy các giá trị này vào options nên đổi token CSS không tự
       lan vào. Lặp qua Object.keys(scales) nên không cần biết chart có mấy trục.

       Không chạm dataset: màu dataset mang nghĩa tài chính (UP / ngưỡng), do
       từng trang tự quyết.

       Ghi thẳng vào ticks/grid/title nên phải tự bảo đảm object tồn tại:
       template có thể chỉ khai báo `x: {grid: {display: false}}`, không có
       `ticks`. */
    function chromeTarget(host, key) {
        if (!host[key]) { host[key] = {}; }
        return host[key];
    }

    function applyChartTheme(chart, colors) {
        if (!chart || !chart.options) { return; }
        var c = colors || palette();
        var scales = chart.options.scales || {};
        Object.keys(scales).forEach(function (name) {
            var scale = scales[name];
            if (!scale || typeof scale !== "object") { return; }
            chromeTarget(scale, "ticks").color = c.muted;
            // grid.display: false vẫn giữ nguyên — chỉ đặt màu, không bật lại.
            chromeTarget(scale, "grid").color = c.lineSoft;
            chromeTarget(scale, "title").color = c.muted;
            chromeTarget(scale, "border").color = c.line;
        });

        var plugins = chart.options.plugins || {};
        if (plugins.legend) {
            chromeTarget(plugins.legend, "labels").color = c.inkSoft;
        }
        if (plugins.tooltip) {
            plugins.tooltip.backgroundColor = c.panel;
            plugins.tooltip.titleColor = c.ink;
            plugins.tooltip.bodyColor = c.inkSoft;
            plugins.tooltip.borderColor = c.line;
            plugins.tooltip.borderWidth = 1;
        }
    }

    /* ---------- theme ----------
       theme.js đổi thuộc tính data-theme trên <html>. Chart đã vẽ giữ nguyên màu
       cũ → theo dõi thuộc tính đó rồi cho trang tự nạp lại màu. */
    function onThemeChange(fn) {
        if (typeof fn !== "function" || !window.MutationObserver) { return; }
        var observer = new MutationObserver(function (records) {
            for (var i = 0; i < records.length; i += 1) {
                if (records[i].attributeName === "data-theme") {
                    fn(palette());
                    return;
                }
            }
        });
        observer.observe(document.documentElement, {attributes: true});
    }

    /* ---------- chọn nhanh ----------
       Dùng cho cả .quick-pick một mã (index) và .pair-pick hai mã (compare):
       template khai báo dữ liệu qua data-*, hàm này chỉ điền input rồi submit. */
    function wireQuickPicks(container, apply) {
        if (!container || typeof apply !== "function") { return; }
        container.addEventListener("click", function (event) {
            var pick = event.target.closest("[data-symbol], [data-symbol-a]");
            if (!pick || !container.contains(pick)) { return; }
            apply(pick);
        });
    }

    /* % thay đổi so với mốc đầu — trục chung để so 2 mã có mức giá khác nhau. */
    function pctChange(base, value) {
        if (!isFinite(base) || base === 0 || !isFinite(value)) { return null; }
        return ((value - base) / base) * 100;
    }

    function signed(value, decimals) {
        var d = typeof decimals === "number" ? decimals : 2;
        if (!isFinite(value)) { return "—"; }
        return (value > 0 ? "+" : "") + value.toFixed(d);
    }

    function formatDateVi(iso) {
        var parts = String(iso).slice(0, 10).split("-");
        if (parts.length < 3) { return String(iso); }
        return parts[2] + "/" + parts[1] + "/" + parts[0];
    }

    window.PageSignal = {
        cssVar: cssVar,
        alphaFrom: alphaFrom,
        palette: palette,
        areaGradient: areaGradient,
        applyChartTheme: applyChartTheme,
        onThemeChange: onThemeChange,
        wireQuickPicks: wireQuickPicks,
        pctChange: pctChange,
        signed: signed,
        formatDateVi: formatDateVi
    };
})();
