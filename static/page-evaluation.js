/* page-evaluation.js — hành vi riêng của trang Đánh giá model (evaluation.html).

   Ba việc:
   1. Vẽ biểu đồ so sánh model bằng Chart.js (grouped bar), màu resolve từ CSS
      token nên dark mode đúng, vẽ lại khi người dùng đổi sáng/tối.
   2. Đếm số lên cho thẻ metric của Final Model trên TEST (UIKit tự tắt khi
      prefers-reduced-motion).
   3. Guard ảnh ma trận nhầm lẫn: route trả plain-text 404 khi chưa có PNG, nên
      phải bắt lỗi tải ảnh rồi hiện thông báo tiếng Việt thay thế.

   Nạp SAU ui-kit.js và SAU vendor/chart.umd.min.js. Mọi node dựng bằng
   createElement + textContent — không gán chuỗi HTML thô (kỷ luật toàn dự án).

   Style theo repo: var + function, thụt 4 space, không ES module, không build.

   KHÔNG import chéo page-signal.js: file đó chỉ nạp trên index/compare, trang này
   không có nó. Phần helper màu được copy sang đây (khoảng 20 dòng) — cố ý trùng
   lặp để trang không phụ thuộc thứ tự nạp của trang khác. */
(function () {
    /* ---------- đọc token màu ----------
       Chart.js vẽ trên canvas nên không "thấy" CSS var → phải resolve token ra
       chuỗi màu thật tại thời điểm dựng chart. */

    function cssVar(name, fallback) {
        var root = document.documentElement;
        var value = "";
        if (window.getComputedStyle) {
            value = window.getComputedStyle(root).getPropertyValue(name);
        }
        value = (value || "").trim();
        return value === "" ? fallback : value;
    }

    /* Token *-rgb là 3 kênh rời ("11 107 67") → ghép thành rgba() để chèn alpha
       ở mọi trình duyệt, không phụ thuộc cú pháp rgb(x y z / a). */
    function alphaFrom(varName, alpha, fallback) {
        var raw = cssVar(varName, "");
        var parts = raw.split(/[\s,]+/).filter(function (p) { return p !== ""; });
        if (parts.length < 3) { return fallback || "transparent"; }
        return "rgba(" + parts[0] + ", " + parts[1] + ", " + parts[2] + ", " + alpha + ")";
    }

    /* Fallback dùng system color (canvastext/canvas/graytext) chứ KHÔNG hex trần:
       nếu app.css chưa nạp thì trình duyệt vẫn chọn màu đúng theo chế độ sáng/tối,
       không có nguy cơ mực đen trên nền đen. */
    function palette() {
        return {
            ink: cssVar("--ink", "canvastext"),
            inkSoft: cssVar("--ink-soft", "canvastext"),
            muted: cssVar("--muted", "graytext"),
            faint: cssVar("--faint", "graytext"),
            line: cssVar("--line", "graytext"),
            lineSoft: cssVar("--line-soft", "graytext"),
            panel: cssVar("--panel", "canvas"),
            paper2: cssVar("--paper-2", "canvas")
        };
    }

    /* theme.js đổi thuộc tính data-theme trên <html>. Chart đã vẽ giữ nguyên màu
       cũ → theo dõi thuộc tính đó rồi vẽ lại. */
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

    /* ---------- 1. biểu đồ so sánh model ---------- */

    /* Ba mức ĐẬM của cùng một mực trung tính, không phải ba hue.
       Lý do không dùng --up/--down: trong app này --up nghĩa là "dự báo giá tăng".
       Tô cột model bằng nó sẽ đọc thành khuyến nghị mua, trái với disclaimer.

       Ba token mực sẵn có vốn đã cách nhau rõ về độ sáng ở CẢ hai theme:
         ink (đậm nhất) → muted (vừa) → line-strong (nhạt nhất).
       Kèm viền cùng màu ink cho mọi cột nên cột nhạt vẫn có đường bao rõ. Cùng với
       thứ tự legend cố định, đây là đủ để phân biệt kể cả với người mù màu hoặc
       khi in trắng đen. */
    function seriesColors(colors) {
        return [colors.ink, colors.muted, cssVar("--line-strong", "graytext")];
    }

    function readJson(id) {
        var node = document.getElementById(id);
        if (!node) { return null; }
        try {
            return JSON.parse(node.textContent || "null");
        } catch (e) {
            if (window.console && window.console.warn) {
                window.console.warn("page-evaluation: JSON dữ liệu biểu đồ lỗi:", e);
            }
            return null;
        }
    }

    function validChartData(data) {
        return !!(data
            && Array.isArray(data.metrics) && data.metrics.length
            && Array.isArray(data.models) && data.models.length
            && Array.isArray(data.values) && data.values.length === data.models.length);
    }

    function buildChart() {
        var canvas = document.getElementById("model-compare-chart");
        var card = document.getElementById("eval-chart-card");
        if (!canvas || !window.Chart) { return; }

        var data = readJson("model-compare-data");
        if (!validChartData(data)) {
            // Template đã guard, nhánh này chỉ để không bao giờ để lại canvas trống.
            if (card) { card.hidden = true; }
            return;
        }

        var colors = palette();
        var fills = seriesColors(colors);

        var datasets = data.models.map(function (name, i) {
            return {
                label: name,
                data: data.values[i],
                backgroundColor: fills[i % fills.length],
                borderColor: colors.ink,
                borderWidth: 1,
                borderRadius: 3,
                // Cột nhạt nhất vẫn phải thấy rõ mép trên khi giá trị nhỏ.
                borderSkipped: false,
                maxBarThickness: 46
            };
        });

        var chart = new window.Chart(canvas, {
            type: "bar",
            data: {labels: data.metrics, datasets: datasets},
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: window.UIKit && window.UIKit.prefersReducedMotion()
                    ? false
                    : {duration: 600},
                interaction: {mode: "index", intersect: false},
                scales: {
                    x: {
                        grid: {display: false},
                        ticks: {color: colors.inkSoft, maxRotation: 0, autoSkip: false},
                        border: {color: colors.line}
                    },
                    y: {
                        beginAtZero: true,
                        max: 1,
                        title: {display: true, text: "Giá trị metric (0 tới 1)", color: colors.muted},
                        grid: {color: colors.lineSoft},
                        ticks: {color: colors.muted},
                        border: {color: colors.line}
                    }
                },
                plugins: {
                    legend: {
                        position: "bottom",
                        labels: {color: colors.inkSoft, usePointStyle: true, boxWidth: 10}
                    },
                    tooltip: {
                        callbacks: {
                            label: function (item) {
                                var value = Number(item.raw);
                                if (!isFinite(value)) { return item.dataset.label + ": —"; }
                                return item.dataset.label + ": " + value.toFixed(4);
                            }
                        }
                    }
                }
            }
        });

        // Đổi sáng/tối: nạp lại token rồi vẽ lại, nếu không chart giữ màu cũ.
        onThemeChange(function (next) {
            var nextFills = seriesColors(next);
            chart.data.datasets.forEach(function (set, i) {
                set.backgroundColor = nextFills[i % nextFills.length];
                set.borderColor = next.ink;
            });
            var scales = chart.options.scales;
            scales.x.ticks.color = next.inkSoft;
            scales.x.border.color = next.line;
            scales.y.ticks.color = next.muted;
            scales.y.title.color = next.muted;
            scales.y.grid.color = next.lineSoft;
            scales.y.border.color = next.line;
            chart.options.plugins.legend.labels.color = next.inkSoft;
            chart.update("none");
        });
    }

    /* ---------- 2. đếm số cho thẻ Final Model TEST ----------
       Hiệu ứng hiện lần lượt của grid do `data-stagger` trong template lo (UIKit
       autoWire, tự tắt khi reduced motion).

       Phần đếm số KHÔNG dùng hook `data-count-up` của UIKit, vì hook đó lấy mốc
       bắt đầu từ chính textContent đang có. Server render sẵn giá trị CUỐI (để
       nhánh không-JS và trình đọc màn hình luôn thấy số thật), nên from == to và
       số sẽ đứng im. Ở đây tự đảo thứ tự: chỉ khi JS chạy và người dùng không tắt
       animation mới hạ ô về 0 rồi đếm lên. Reduced motion → không chạm vào DOM,
       số thật giữ nguyên. */
    function wireFinalStats() {
        if (!window.UIKit) { return; }
        if (window.UIKit.prefersReducedMotion()) { return; }
        var cells = document.querySelectorAll(".eval-final-stats [data-eval-count]");
        for (var i = 0; i < cells.length; i += 1) {
            var cell = cells[i];
            var to = parseFloat(cell.getAttribute("data-eval-count"));
            if (!isFinite(to)) { continue; }
            cell.textContent = "0.0000";
            window.UIKit.countUp(cell, to, {decimals: 4, duration: 700});
        }
    }

    /* ---------- 3. guard ảnh ma trận nhầm lẫn ----------
       app.py:591-595 trả về chuỗi plain-text kèm 404 khi reports/confusion_matrix.png
       chưa tồn tại. Trình duyệt coi đó là ảnh lỗi → onerror nổ. Khi đó ẩn figure và
       hiện khối thông báo tiếng Việt đã render sẵn trong template.

       Wire ở đây thay vì thuộc tính onerror inline: inline handler là JS trong HTML,
       vướng CSP và trái quy ước dự án. */
    function wireConfusionMatrix() {
        var img = document.getElementById("cm-image");
        var figure = document.getElementById("cm-figure");
        var missing = document.getElementById("cm-missing");
        if (!img) { return; }

        function fail() {
            if (figure) { figure.hidden = true; }
            if (missing) { missing.hidden = false; }
        }

        img.addEventListener("error", fail);
        // Ảnh có thể đã lỗi TRƯỚC khi script này chạy (cache, kết nối nhanh):
        // complete=true + naturalWidth=0 là dấu hiệu tải thất bại.
        if (img.complete && img.naturalWidth === 0) { fail(); }
    }

    function guard(fn) {
        try {
            fn();
        } catch (e) {
            if (window.console && window.console.warn) {
                window.console.warn("page-evaluation lỗi:", e);
            }
        }
    }

    function init() {
        guard(wireConfusionMatrix);
        guard(wireFinalStats);
        guard(buildChart);
    }

    if (window.UIKit && window.UIKit.onReady) {
        window.UIKit.onReady(init);
    } else if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", init);
    } else {
        init();
    }
})();
