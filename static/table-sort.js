/* table-sort.js — sắp xếp cột client-side cho các bảng render sẵn cả bộ dữ liệu
   (Xếp hạng tín hiệu, Đánh giá model). Ba trạng thái xoay vòng theo mỗi lần bấm:
   tăng dần → giảm dần → mặc định (thứ tự server trả về).

   KHÔNG dùng cho bảng lịch sử Tuning Lab: bảng đó phân trang 50 dòng/trang ở
   server, sort client sẽ chỉ sắp trong một trang nên ra kết quả sai. Bảng đó đi
   theo tham số ?sort= của backend.

   Cách bật: thêm data-ui="sortable" cho <table>, rồi data-sort="<kiểu>" cho mỗi
   <th> muốn sắp. Kiểu: text | number | percent | date | none.
   Ô trống ("—", "N/A") luôn bị đẩy xuống cuối, bất kể chiều sắp. */
(function () {
    var ASC = "ascending";
    var DESC = "descending";

    var ICON = {};
    ICON[""] = "⇅";      // ⇅ mặc định
    ICON[ASC] = "▲";     // ▲ tăng dần
    ICON[DESC] = "▼";    // ▼ giảm dần

    var LABEL = {};
    LABEL[""] = "thứ tự mặc định";
    LABEL[ASC] = "tăng dần";
    LABEL[DESC] = "giảm dần";

    var NEXT = {};
    NEXT[""] = ASC;
    NEXT[ASC] = DESC;
    NEXT[DESC] = "";

    // Ô coi như rỗng: giữ nguyên vị trí tương đối và luôn nằm cuối bảng.
    function isBlank(text) {
        var t = text.trim();
        return t === "" || t === "—" || t === "-" || t === "N/A" || t === "NaN";
    }

    // "72.3%" → 72.3 ; "1.234,5" và "1,234.5" đều về số ; "1.23x" → 1.23
    function toNumber(text) {
        var t = text.replace(/[\s%x]/gi, "");
        if (t.indexOf(",") !== -1 && t.indexOf(".") !== -1) {
            t = t.lastIndexOf(",") > t.lastIndexOf(".")
                ? t.replace(/\./g, "").replace(",", ".")
                : t.replace(/,/g, "");
        } else if (t.indexOf(",") !== -1) {
            t = t.replace(",", ".");
        }
        t = t.replace(/[^0-9.+-]/g, "");
        var n = parseFloat(t);
        return isFinite(n) ? n : null;
    }

    // dd/mm/yyyy (định dạng đang hiển thị) và ISO yyyy-mm-dd[Thh:mm:ss].
    function toDate(text) {
        var t = text.trim();
        var vi = t.match(/^(\d{1,2})\/(\d{1,2})\/(\d{4})$/);
        if (vi) {
            return Number(vi[3]) * 10000 + Number(vi[2]) * 100 + Number(vi[1]);
        }
        var iso = t.match(/^(\d{4})-(\d{2})-(\d{2})/);
        if (iso) {
            var rest = t.slice(10).replace(/\D/g, "");
            return Number(iso[1] + iso[2] + iso[3] + (rest + "000000").slice(0, 6));
        }
        return null;
    }

    function cellText(row, index) {
        var cell = row.children[index];
        return cell ? (cell.textContent || "") : "";
    }

    function valueOf(row, index, kind) {
        var text = cellText(row, index);
        if (isBlank(text)) { return null; }
        if (kind === "number" || kind === "percent") { return toNumber(text); }
        if (kind === "date") { return toDate(text); }
        return text.trim().toLocaleUpperCase("vi");
    }

    function compare(a, b) {
        if (typeof a === "number" && typeof b === "number") {
            return a < b ? -1 : (a > b ? 1 : 0);
        }
        return String(a).localeCompare(String(b), "vi");
    }

    // Dòng placeholder (colspan .empty) không phải dữ liệu, không đem sắp.
    function dataRows(tbody) {
        return Array.prototype.filter.call(tbody.rows, function (row) {
            return !row.querySelector("td.empty");
        });
    }

    function mountTable(table) {
        if (table.dataset.sortBound) { return; }
        var tbody = table.tBodies[0];
        var headRow = table.tHead && table.tHead.rows[0];
        if (!tbody || !headRow) { return; }

        var rows = dataRows(tbody);
        if (rows.length < 2) { return; }
        table.dataset.sortBound = "1";

        // Thứ tự gốc do server sắp — chính là trạng thái "mặc định" của vòng xoay.
        var original = rows.slice();
        var headers = [];
        var state = { th: null, dir: "" };

        function render() {
            headers.forEach(function (item) {
                var on = item.th === state.th && state.dir !== "";
                var dir = on ? state.dir : "";
                if (on) {
                    item.th.setAttribute("aria-sort", dir);
                } else {
                    item.th.removeAttribute("aria-sort");
                }
                item.th.classList.toggle("is-sorted", on);
                item.icon.textContent = ICON[dir];
                item.btn.setAttribute(
                    "title", "Sắp xếp theo " + item.name + " (" + LABEL[NEXT[dir]] + ")"
                );
                item.btn.setAttribute(
                    "aria-label", item.name + " — đang " + LABEL[dir]
                        + ", bấm để " + LABEL[NEXT[dir]]
                );
            });
        }

        function apply() {
            var ordered;
            if (state.dir === "") {
                ordered = original;
            } else {
                var index = state.th.cellIndex;
                var kind = state.th.dataset.sort;
                var factor = state.dir === DESC ? -1 : 1;
                var keyed = original.map(function (row, i) {
                    return { row: row, key: valueOf(row, index, kind), i: i };
                });
                var filled = keyed.filter(function (item) { return item.key !== null; });
                var blanks = keyed.filter(function (item) { return item.key === null; });
                // So bằng thì giữ thứ tự gốc (sort ổn định trên mọi engine cũ).
                filled.sort(function (x, y) {
                    return compare(x.key, y.key) * factor || x.i - y.i;
                });
                ordered = filled.concat(blanks).map(function (item) { return item.row; });
            }
            // Ghi lại cả khối một lượt: append phần tử đã có trong DOM là move,
            // nên không mất listener hay trạng thái hidden của bộ lọc.
            var frag = document.createDocumentFragment();
            ordered.forEach(function (row) { frag.appendChild(row); });
            tbody.appendChild(frag);
            render();
        }

        Array.prototype.forEach.call(headRow.cells, function (th) {
            var kind = th.dataset.sort;
            if (!kind || kind === "none") { return; }
            var name = (th.textContent || "").trim() || "cột này";
            var btn = document.createElement("button");
            btn.type = "button";
            btn.className = "sort-btn";
            var label = document.createElement("span");
            label.className = "sort-label";
            label.textContent = name;
            var icon = document.createElement("span");
            icon.className = "sort-icon";
            icon.setAttribute("aria-hidden", "true");
            btn.appendChild(label);
            btn.appendChild(icon);
            th.textContent = "";
            th.appendChild(btn);
            // Gắn class thay vì để CSS dò :has(> .sort-btn) — ô tiêu đề phải bỏ
            // padding cho nút chiếm hết, và class thì trình duyệt nào cũng hiểu.
            th.classList.add("th-sortable");

            var item = { th: th, btn: btn, icon: icon, name: name };
            headers.push(item);
            btn.addEventListener("click", function () {
                state.dir = state.th === th ? NEXT[state.dir] : ASC;
                state.th = state.dir === "" ? null : th;
                apply();
            });
        });

        if (headers.length) { render(); }
    }

    function mount() {
        // ~= để data-ui còn chứa hook khác cùng lúc (vd "data-table sortable").
        var tables = document.querySelectorAll('table[data-ui~="sortable"]');
        Array.prototype.forEach.call(tables, mountTable);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", mount);
    } else {
        mount();
    }

    // Cho trang nào thay DOM bằng AJAX gắn lại sau khi swap.
    window.mountTableSort = mount;
})();
