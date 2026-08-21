/* tuning-lab.js — hành vi động của Tuning Lab (/tuning) và trang tiến độ fetch
   (/tuning/fetch-status).

   Lý do file này tồn tại: trước đây cả hai trang tự làm mới bằng
   <meta http-equiv="refresh" content="5">. Mỗi 5 giây trang tải lại toàn bộ,
   kéo theo ba tác hại thật:
     - vị trí cuộn nhảy về đầu trang (đang đọc bảng lịch sử thì bị giật lên),
     - mọi ô input đang gõ ở hai model còn lại bị xoá (nút submit bị disable
       nhưng input thì không),
     - cả shell + chat dock nháy theo.

   Nay chỉ đúng vùng trạng thái bị thay bằng fetch + DOMParser.

   Phụ thuộc: static/ui-kit.js (window.UIKit) cho toast/elapsed. Nếu ui-kit
   chưa nạp thì mọi thứ ở đây vẫn chạy, chỉ mất phần thông báo/đồng hồ.
   Nạp sau ui-kit.js qua block page_scripts.

   Không dùng innerHTML ở bất cứ đâu (giữ đúng quy ước của repo). */

(function () {
    "use strict";

    var POLL_MS = 5000;
    var JOB_REGION_ID = "tuning-job-status";
    var ELAPSED_ID = "tuning-job-elapsed";

    function kit() {
        return window.UIKit || null;
    }

    /* Bọc mọi callback: một lỗi lẻ ở đây không được phép làm chết cả trang. */
    function guard(fn) {
        try {
            return fn();
        } catch (err) {
            if (window.console && window.console.warn) {
                window.console.warn("[tuning-lab]", err);
            }
            return undefined;
        }
    }

    function formatClock(totalSeconds) {
        var s = Math.max(0, Math.floor(totalSeconds));
        var h = Math.floor(s / 3600);
        var m = Math.floor((s % 3600) / 60);
        var sec = s % 60;
        function pad(n) {
            return n < 10 ? "0" + n : String(n);
        }
        if (h > 0) {
            return h + ":" + pad(m) + ":" + pad(sec);
        }
        return m + ":" + pad(sec);
    }

    /* Đồng hồ tự chạy — dùng UIKit.elapsed khi có (một nguồn định dạng duy
       nhất), tự đếm khi không có để trang vẫn hiện được thời gian. */
    function startClock(el, isoStart) {
        if (!el || !isoStart) { return function () {}; }
        var ui = kit();
        if (ui && typeof ui.elapsed === "function") {
            return ui.elapsed(el, isoStart);
        }
        var start = Date.parse(isoStart);
        if (isNaN(start)) { return function () {}; }
        function tick() {
            el.textContent = formatClock((Date.now() - start) / 1000);
        }
        tick();
        var id = window.setInterval(tick, 1000);
        return function () { window.clearInterval(id); };
    }

    function notify(message, type, title) {
        var ui = kit();
        if (ui && typeof ui.toast === "function") {
            ui.toast(message, {type: type, title: title});
        }
    }

    /* ============================================================
       1. Vùng trạng thái job CV — poll rồi thay đúng một element
       ============================================================ */

    function mountJobStatus() {
        var region = document.getElementById(JOB_REGION_ID);
        if (!region) { return; }
        if (region.dataset.jobBound === "1") { return; }
        region.dataset.jobBound = "1";

        var stopClock = startClock(
            document.getElementById(ELAPSED_ID),
            region.dataset.jobStarted
        );

        // data-job-running là cờ DUY NHẤT quyết định có poll hay không.
        // Đừng suy từ sự tồn tại của panel: panel "completed" cũng nằm trong
        // cùng element này.
        if (region.dataset.jobRunning !== "1") { return; }

        var wasRunning = true;
        var timer = null;
        var stopped = false;

        function stop() {
            stopped = true;
            if (timer) { window.clearTimeout(timer); timer = null; }
            stopClock();
        }

        function schedule() {
            if (stopped) { return; }
            timer = window.setTimeout(poll, POLL_MS);
        }

        function announce(status, model) {
            if (status === "completed") {
                notify(
                    "CV đã xong" + (model ? " cho " + model : "") +
                    ". Kết quả và bảng lịch sử đã được cập nhật.",
                    "success",
                    "Tuning Lab"
                );
            } else if (status === "error") {
                notify(
                    "Job CV thất bại. Xem chi tiết lỗi trong phần trạng thái.",
                    "error",
                    "Tuning Lab"
                );
            }
        }

        function poll() {
            if (stopped) { return; }
            // cache: "no-store" — không thì một số proxy/trình duyệt trả lại
            // đúng bản HTML đã cache và trạng thái sẽ đứng im mãi.
            window.fetch(window.location.href, {
                cache: "no-store",
                headers: {"X-Requested-With": "tuning-poll"}
            })
                .then(function (r) {
                    if (!r.ok) { throw new Error("HTTP " + r.status); }
                    return r.text();
                })
                .then(function (html) {
                    if (stopped) { return; }
                    var doc = new DOMParser().parseFromString(html, "text/html");
                    var next = doc.getElementById(JOB_REGION_ID);
                    if (!next) { schedule(); return; }

                    var nextStatus = next.dataset.jobStatus || "idle";
                    var nextRunning = next.dataset.jobRunning === "1";

                    var current = document.getElementById(JOB_REGION_ID);
                    if (!current) { stop(); return; }

                    if (nextRunning) {
                        // Vẫn đang chạy: KHÔNG thay DOM. Thay sẽ giết đồng hồ
                        // đang chạy và làm thanh tiến trình khởi động lại
                        // animation mỗi 5 giây (nhìn như bị kẹt).
                        schedule();
                        return;
                    }

                    stop();
                    if (!wasRunning) { return; }
                    wasRunning = false;

                    var modelLabel = "";
                    var title = next.querySelector(".section-label");
                    if (title && title.textContent) {
                        modelLabel = title.textContent.replace(/^CV hoàn tất\s*—\s*/, "").trim();
                    }

                    current.replaceWith(next);
                    guard(function () {
                        var ui = kit();
                        if (ui && typeof ui.autoWire === "function") { ui.autoWire(); }
                    });
                    announce(nextStatus, modelLabel);

                    // Job xong thì không chỉ vùng trạng thái đổi: thẻ model,
                    // số lần CV, bảng lịch sử đều có dữ liệu mới. Tải lại đúng
                    // MỘT lần ở đây thay vì tải lại mỗi 5 giây như trước.
                    window.setTimeout(function () {
                        window.location.reload();
                    }, 1200);
                })
                .catch(function () {
                    // Lỗi mạng lẻ (server đang bận vì chính job CV) không nên
                    // dừng poll — thử lại lượt sau.
                    schedule();
                });
        }

        schedule();
        window.addEventListener("beforeunload", stop);
    }

    /* ============================================================
       2. Overlay khi bấm "Chạy pipeline chính thức"
       app.py chạy pipeline bằng subprocess.run đồng bộ ngay trong request
       handler, nên tab thật sự bị treo vài phút và không có cách nào tránh ở
       phía client. Overlay không sửa được điều đó — nó biến một cú treo im
       lặng thành một khoảng chờ đọc được.
       ============================================================ */

    var PIPELINE_STEPS = [
        ["1", "Tạo đặc trưng và chia dữ liệu theo thời gian"],
        ["2", "Chọn model trên tập VALIDATION"],
        ["3", "Đánh giá một lần trên tập TEST rồi phát hành"]
    ];

    function buildOverlay() {
        var overlay = document.createElement("div");
        overlay.className = "pipeline-overlay";
        overlay.id = "pipeline-overlay";
        overlay.setAttribute("role", "dialog");
        overlay.setAttribute("aria-modal", "true");
        overlay.setAttribute("aria-labelledby", "pipeline-overlay-title");

        var panel = document.createElement("div");
        panel.className = "pipeline-overlay-panel";
        panel.tabIndex = -1;

        var head = document.createElement("div");
        head.className = "pipeline-overlay-head";
        var spinner = document.createElement("span");
        spinner.className = "spinner";
        spinner.setAttribute("aria-hidden", "true");
        var title = document.createElement("h2");
        title.className = "pipeline-overlay-title";
        title.id = "pipeline-overlay-title";
        title.textContent = "Đang chạy pipeline chính thức";
        head.appendChild(spinner);
        head.appendChild(title);

        var desc = document.createElement("p");
        desc.className = "pipeline-overlay-desc";
        desc.textContent =
            "Pipeline chạy đồng bộ trong request nên trang sẽ đứng yên cho tới khi " +
            "xong. Đây là hành vi bình thường, không phải lỗi.";

        var warn = document.createElement("p");
        warn.className = "pipeline-overlay-warn";
        warn.textContent =
            "Giữ tab này mở. Đóng tab hoặc bấm tải lại sẽ làm mất kết quả của lượt chạy.";

        var clock = document.createElement("div");
        clock.className = "pipeline-overlay-clock";
        var clockLabel = document.createElement("span");
        clockLabel.className = "pipeline-overlay-clock-label";
        clockLabel.textContent = "Đã chờ";
        var clockValue = document.createElement("span");
        clockValue.className = "pipeline-overlay-clock-value";
        clockValue.setAttribute("role", "timer");
        clockValue.setAttribute("aria-label", "Thời gian đã chờ pipeline");
        clockValue.textContent = "0:00";
        clock.appendChild(clockLabel);
        clock.appendChild(clockValue);

        var bar = document.createElement("div");
        bar.className = "progress-indeterminate";
        bar.setAttribute("aria-hidden", "true");
        bar.appendChild(document.createElement("span"));

        var steps = document.createElement("ol");
        steps.className = "pipeline-overlay-steps";
        PIPELINE_STEPS.forEach(function (pair) {
            var li = document.createElement("li");
            li.className = "pipeline-overlay-step";
            var num = document.createElement("span");
            num.className = "pipeline-overlay-step-num";
            num.textContent = pair[0];
            var text = document.createElement("span");
            text.className = "pipeline-overlay-step-text";
            text.textContent = pair[1];
            li.appendChild(num);
            li.appendChild(text);
            steps.appendChild(li);
        });

        panel.appendChild(head);
        panel.appendChild(desc);
        panel.appendChild(warn);
        panel.appendChild(clock);
        panel.appendChild(bar);
        panel.appendChild(steps);
        overlay.appendChild(panel);

        return {overlay: overlay, panel: panel, clock: clockValue};
    }

    function showPipelineOverlay() {
        if (document.getElementById("pipeline-overlay")) { return; }
        var built = buildOverlay();
        document.body.appendChild(built.overlay);
        built.panel.focus();

        var start = Date.now();
        var id = window.setInterval(function () {
            built.clock.textContent = formatClock((Date.now() - start) / 1000);
        }, 1000);

        // Overlay là modal: chặn Tab thoát ra sau nền, và chặn Escape vì đóng
        // overlay không huỷ được pipeline đang chạy ở server.
        document.addEventListener("keydown", function (e) {
            if (!document.getElementById("pipeline-overlay")) { return; }
            if (e.key === "Tab") {
                e.preventDefault();
                built.panel.focus();
            } else if (e.key === "Escape") {
                e.preventDefault();
            }
        }, true);

        window.addEventListener("beforeunload", function () {
            window.clearInterval(id);
        });
    }

    function wirePipelineForm() {
        var form = document.getElementById("run-pipeline-form");
        if (!form || form.dataset.overlayBound === "1") { return; }
        form.dataset.overlayBound = "1";
        form.addEventListener("submit", function () {
            var btn = form.querySelector("button[type=submit]");
            if (btn && btn.disabled) { return; }
            guard(showPipelineOverlay);
        });
    }

    /* ============================================================
       3. Trang tiến độ fetch — tự poll thay cho <meta refresh>
       Fetch chạy ~20-25 phút. Tải lại cả trang mỗi 5 giây suốt 25 phút làm log
       nhảy về đầu và nháy màn hình liên tục.
       ============================================================ */

    var FETCH_TARGETS = [
        "#fetch-hero",
        "#fetch-steps",
        "#fetch-stats",
        "#fetch-log-panel"
    ];

    function scrollLogToEnd() {
        var log = document.getElementById("fetch-log");
        if (log) { log.scrollTop = log.scrollHeight; }
    }

    function mountFetchStatus() {
        var root = document.getElementById("fetch-status-root");
        if (!root || root.dataset.pollBound === "1") { return; }
        root.dataset.pollBound = "1";

        scrollLogToEnd();

        if (root.dataset.fetchRunning !== "1") { return; }

        var stopped = false;
        var timer = null;

        function stop() {
            stopped = true;
            if (timer) { window.clearTimeout(timer); timer = null; }
        }

        function schedule() {
            if (stopped) { return; }
            timer = window.setTimeout(poll, POLL_MS);
        }

        function swapSection(doc, selector) {
            var next = doc.querySelector(selector);
            var current = document.querySelector(selector);
            if (!next || !current) { return; }
            // Log đang được người dùng cuộn lên đọc thì đừng kéo xuống cuối.
            var log = current.querySelector("#fetch-log");
            var pinned = true;
            if (log) {
                pinned = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
            }
            current.replaceWith(next);
            if (log && pinned) { scrollLogToEnd(); }
        }

        function poll() {
            if (stopped) { return; }
            window.fetch(window.location.href, {
                cache: "no-store",
                headers: {"X-Requested-With": "fetch-poll"}
            })
                .then(function (r) {
                    if (!r.ok) { throw new Error("HTTP " + r.status); }
                    return r.text();
                })
                .then(function (html) {
                    if (stopped) { return; }
                    var doc = new DOMParser().parseFromString(html, "text/html");
                    var nextRoot = doc.getElementById("fetch-status-root");
                    FETCH_TARGETS.forEach(function (sel) {
                        guard(function () { swapSection(doc, sel); });
                    });
                    guard(function () {
                        var ui = kit();
                        if (ui && typeof ui.autoWire === "function") { ui.autoWire(); }
                    });

                    var stillRunning = nextRoot && nextRoot.dataset.fetchRunning === "1";
                    if (stillRunning) { schedule(); return; }

                    stop();
                    root.dataset.fetchRunning = "0";
                    var status = nextRoot ? (nextRoot.dataset.fetchStatus || "") : "";
                    if (status === "failed") {
                        notify(
                            "Làm mới dữ liệu thất bại. Xem log bên dưới để biết chi tiết.",
                            "error",
                            "Làm mới dữ liệu"
                        );
                    } else {
                        notify(
                            "Đã làm mới dữ liệu xong. Có thể train lại trên dataset mới.",
                            "success",
                            "Làm mới dữ liệu"
                        );
                    }
                    // Tải lại một lần để nút "Train lại trên dataset mới" và các
                    // trạng thái phụ thuộc lock cùng cập nhật.
                    window.setTimeout(function () {
                        window.location.reload();
                    }, 1500);
                })
                .catch(function () {
                    schedule();
                });
        }

        schedule();
        window.addEventListener("beforeunload", stop);
    }

    /* ============================================================
       4. Biểu đồ F1 theo số lần CV — "tuning đã hội tụ chưa?"
       Bảng lịch sử trả lời được "run nào tốt nhất" nhưng không trả lời được
       "còn thử nữa có hơn không". Đường F1 theo thứ tự thời gian trả lời câu
       đó: đi ngang mấy lượt cuối = đã tới hạn của không gian tham số này.
       Dữ liệu do server dựng (history_trend), đã sắp theo timestamp — client
       KHÔNG sắp lại, để một nguồn thứ tự duy nhất.
       ============================================================ */

    var TREND_CANVAS_ID = "cv-trend-chart";
    // Giữ instance ở tầng module: mountAll() chạy lại sau mỗi lần AJAX swap
    // #main-content, mà swap thì thay luôn node <canvas>. Chart.js cũ vẫn giữ
    // tham chiếu tới canvas đã bị bỏ + một listener resize → rò rỉ dần. Phải
    // destroy() bản trước khi dựng bản mới.
    var trendChart = null;
    // MutationObserver theo dõi data-theme cũng phải gỡ khi dựng lại: mỗi lần
    // swap mà bỏ quên một observer là thêm một closure giữ mảng points CŨ, rồi
    // lần đổi theme sau nó ghi màu theo dữ liệu cũ lên chart mới.
    var trendThemeWatcher = null;

    /* Chart.js vẽ lên canvas nên không "thấy" CSS var → phải resolve token
       thành chuỗi màu tại thời điểm dựng. page-signal.js làm đúng việc này cho
       2 trang tín hiệu, nhưng file đó KHÔNG được nạp ở trang tuning nên đọc
       tại chỗ, không phụ thuộc window.PageSignal.
       Fallback là system color (canvastext/graytext) chứ không hex: nếu app.css
       chưa nạp thì trình duyệt vẫn chọn màu theo chế độ sáng/tối, không có nguy
       cơ mực đen trên nền đen. */
    function cssVar(name, fallback) {
        var value = "";
        if (window.getComputedStyle) {
            value = window.getComputedStyle(document.documentElement)
                .getPropertyValue(name);
        }
        value = (value || "").trim();
        return value || fallback || "";
    }

    /* Chỉ token mực trung tính. KHÔNG dùng --up/--down: theo ngôn ngữ thiết kế
       của repo, xanh/đỏ dành riêng cho tín hiệu tăng/giảm của thị trường, còn
       điểm F1 cao không phải là "mã này sẽ tăng". */
    function trendPalette() {
        return {
            ink: cssVar("--ink", "canvastext"),
            inkSoft: cssVar("--ink-soft", "canvastext"),
            muted: cssVar("--muted", "graytext"),
            line: cssVar("--line", "graytext"),
            lineSoft: cssVar("--line-soft", "graytext"),
            panel: cssVar("--panel", "canvas")
        };
    }

    function reducedMotion() {
        var ui = kit();
        if (ui && typeof ui.prefersReducedMotion === "function") {
            return ui.prefersReducedMotion();
        }
        return !!(window.matchMedia
            && window.matchMedia("(prefers-reduced-motion: reduce)").matches);
    }

    /* Điểm best và điểm đang dùng phải phân biệt được bằng HÌNH, không chỉ bằng
       màu: hai token mực gần nhau khi in đen trắng hoặc với người khó phân biệt
       màu là như nhau. Best = hình thoi to, đang dùng = vòng tròn có viền dày. */
    function pointRadius(item) {
        if (item.is_best) { return 6; }
        if (item.is_selected) { return 5; }
        return 3;
    }

    function pointStyle(item) {
        if (item.is_best) { return "rectRot"; }
        if (item.is_selected) { return "circle"; }
        return "circle";
    }

    function pointBorderWidth(item) {
        return (item.is_best || item.is_selected) ? 2.5 : 1;
    }

    function pointColors(points, colors) {
        return {
            // Điểm thường tô rỗng (màu nền panel) để đường không bị "hạt" dày
            // đặc; hai điểm đặc biệt tô đầy nên bật ra ngay.
            background: points.map(function (item) {
                return (item.is_best || item.is_selected) ? colors.ink : colors.panel;
            }),
            border: points.map(function (item) {
                return item.is_best ? colors.ink : colors.inkSoft;
            }),
            radius: points.map(pointRadius),
            style: points.map(pointStyle),
            width: points.map(pointBorderWidth)
        };
    }

    function applyTrendColors(chart, points) {
        var colors = trendPalette();
        var marks = pointColors(points, colors);
        var dataset = chart.data.datasets[0];
        dataset.borderColor = colors.inkSoft;
        dataset.pointBackgroundColor = marks.background;
        dataset.pointBorderColor = marks.border;
        dataset.pointRadius = marks.radius;
        dataset.pointStyle = marks.style;
        dataset.pointBorderWidth = marks.width;

        var scales = chart.options.scales;
        scales.x.ticks.color = colors.muted;
        scales.x.grid.color = colors.lineSoft;
        scales.x.title.color = colors.muted;
        scales.y.ticks.color = colors.muted;
        scales.y.grid.color = colors.lineSoft;
        scales.y.title.color = colors.muted;

        var tooltip = chart.options.plugins.tooltip;
        tooltip.backgroundColor = colors.panel;
        tooltip.titleColor = colors.ink;
        tooltip.bodyColor = colors.inkSoft;
        tooltip.borderColor = colors.line;
    }

    /* theme.js đổi thuộc tính data-theme trên <html> (không phát custom event),
       nên bám vào chính mutation đó — cùng cách page-signal.onThemeChange làm. */
    function watchTheme(fn) {
        if (!window.MutationObserver) { return null; }
        var observer = new MutationObserver(function (records) {
            for (var i = 0; i < records.length; i += 1) {
                if (records[i].attributeName === "data-theme") {
                    guard(fn);
                    return;
                }
            }
        });
        observer.observe(document.documentElement, {attributes: true});
        return observer;
    }

    function readTrend(canvas) {
        var raw = canvas.getAttribute("data-trend");
        if (!raw) { return []; }
        var parsed;
        try {
            parsed = JSON.parse(raw);
        } catch (err) {
            // Dữ liệu méo thì mất biểu đồ, không được làm chết cả trang.
            return [];
        }
        if (!parsed || typeof parsed.length !== "number") { return []; }
        var points = [];
        for (var i = 0; i < parsed.length; i += 1) {
            var item = parsed[i] || {};
            var f1 = Number(item.f1);
            // Chốt chặn thứ hai sau _numeric() ở server: nếu vì lý do gì mà một
            // giá trị không hữu hạn lọt tới đây, Chart.js sẽ vẽ ra đường gãy
            // im lặng thay vì báo lỗi.
            if (!isFinite(f1)) { continue; }
            points.push({
                n: Number(item.n) || (points.length + 1),
                f1: f1,
                timestamp: typeof item.timestamp === "string" ? item.timestamp : "",
                is_best: !!item.is_best,
                is_selected: !!item.is_selected
            });
        }
        return points;
    }

    function mountCvTrend() {
        var canvas = document.getElementById(TREND_CANVAS_ID);
        // Canvas chỉ có ở /tuning; mọi trang khác dùng chung file này nên thoát
        // im lặng là đường đi bình thường, không phải lỗi.
        if (!canvas) { return; }
        if (!window.Chart) { return; }
        if (canvas.dataset.trendBound === "1") { return; }
        canvas.dataset.trendBound = "1";

        if (trendChart) {
            guard(function () { trendChart.destroy(); });
            trendChart = null;
        }
        if (trendThemeWatcher) {
            guard(function () { trendThemeWatcher.disconnect(); });
            trendThemeWatcher = null;
        }

        var points = readTrend(canvas);
        if (!points.length) { return; }

        var chart = new window.Chart(canvas, {
            type: "line",
            data: {
                labels: points.map(function (item) { return item.n; }),
                datasets: [{
                    label: "F1 (UP) trung bình CV",
                    data: points.map(function (item) { return item.f1; }),
                    borderWidth: 2,
                    tension: 0.25,
                    fill: false,
                    pointHitRadius: 12
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                // Tôn trọng prefers-reduced-motion: false tắt hẳn animation vào
                // của Chart.js, không chỉ làm nó ngắn hơn.
                animation: reducedMotion() ? false : {duration: 320},
                interaction: {mode: "nearest", intersect: false},
                scales: {
                    x: {
                        title: {display: true, text: "Lần chạy CV (theo thời gian)"},
                        ticks: {},
                        grid: {display: false}
                    },
                    y: {
                        title: {display: true, text: "F1 (UP)"},
                        ticks: {},
                        grid: {}
                    }
                },
                plugins: {
                    legend: {display: false},
                    tooltip: {
                        displayColors: false,
                        borderWidth: 1,
                        callbacks: {
                            title: function (items) {
                                if (!items.length) { return ""; }
                                var item = points[items[0].dataIndex];
                                return item && item.timestamp
                                    ? item.timestamp.replace("T", " ")
                                    : "Lần " + (items[0].label || "");
                            },
                            label: function (item) {
                                var point = points[item.dataIndex] || {};
                                var text = "F1 = " + Number(point.f1).toFixed(4);
                                if (point.is_best) { text += " · tốt nhất"; }
                                if (point.is_selected) { text += " · đang dùng"; }
                                return text;
                            }
                        }
                    }
                }
            }
        });

        applyTrendColors(chart, points);
        chart.update();
        trendChart = chart;

        trendThemeWatcher = watchTheme(function () {
            // Canvas có thể đã bị AJAX swap thay mất trong lúc chờ; khi đó
            // instance này không còn thuộc DOM nào, cập nhật là vô nghĩa.
            if (!trendChart || !document.body.contains(trendChart.canvas)) { return; }
            applyTrendColors(trendChart, points);
            trendChart.update();
        });
    }

    /* ============================================================
       5. Khởi động
       ============================================================ */

    function mountAll() {
        guard(mountJobStatus);
        guard(wirePipelineForm);
        guard(mountFetchStatus);
        guard(mountCvTrend);
    }

    // Cho phần AJAX swap của bảng lịch sử gắn lại sau khi thay #main-content.
    window.mountTuningLab = mountAll;

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", mountAll);
    } else {
        mountAll();
    }
})();
