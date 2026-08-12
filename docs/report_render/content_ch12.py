"""Chuong 1 (Gioi thieu) va Chuong 2 (Co so ly thuyet)."""

from __future__ import annotations

from docx_builder import DocxBuilder, M


def _C(arg: str) -> str:
    """C(arg) — gia dong cua."""
    return M.r("C") + M.paren(M.r(arg))


def _fn(name: str, arg: str) -> str:
    """Ten dac trung dang chu dung kem doi so, vd close_vs_sma20(t)."""
    return M.t(name) + M.paren(M.r(arg))


def _sma(n: str, arg: str = "t") -> str:
    return M.sub(M.t("SMA"), M.r(n)) + M.paren(M.r(arg))


def build_chapter1(doc: DocxBuilder, assets) -> None:
    doc.heading(1, "CHƯƠNG 1: GIỚI THIỆU")

    doc.heading(2, "1.1. Đặt vấn đề")
    doc.paragraph(
        "Sàn giao dịch chứng khoán TP. Hồ Chí Minh (HOSE) hiện có khoảng 400 mã cổ phiếu "
        "giao dịch mỗi phiên. Một nhà đầu tư cá nhân theo dõi thủ công chỉ kịp đọc vài mã "
        "quen thuộc, trong khi tín hiệu kỹ thuật của toàn sàn thay đổi từng phiên. Việc "
        "tính lại các chỉ báo như trung bình động, RSI, độ biến động hay vị thế giá so với "
        "vùng đỉnh - đáy 20 phiên cho toàn bộ sàn là công việc lặp đi lặp lại, dễ sai sót "
        "và không thể làm kịp bằng bảng tính."
    )
    doc.paragraph(
        "Vấn đề thứ hai nghiêm trọng hơn về mặt khoa học dữ liệu. Các thử nghiệm dự báo "
        "chứng khoán rất dễ cho ra kết quả đẹp nhưng vô giá trị vì rò rỉ dữ liệu (data "
        "leakage). Ba lỗi thường gặp là: gán nhãn tương lai theo dòng thứ năm của riêng "
        "từng mã nên nhãn không nằm cùng một phiên thị trường; chia dữ liệu kiểm định chéo "
        "theo dòng dữ liệu gộp nên cùng một ngày giao dịch xuất hiện ở cả tập huấn luyện "
        "lẫn tập kiểm định; và dùng tập TEST để chọn mô hình rồi báo cáo lại chính tập đó "
        "như kết quả khách quan. Ba lỗi này khiến chỉ số báo cáo cao hơn năng lực thực tế."
    )
    doc.paragraph(
        "Vì vậy bài toán thực tế cần giải quyết không chỉ là huấn luyện một mô hình phân "
        "loại, mà là xây dựng một quy trình thực nghiệm có kỷ luật: nhãn được định nghĩa "
        "trên lịch phiên chung của thị trường, ranh giới dữ liệu theo thời gian có purge, "
        "tập TEST bị khóa và chỉ được đánh giá một lần, đồng thời so sánh bắt buộc với các "
        "baseline hằng số để biết mô hình có thực sự đóng góp gì hay không. Kết quả trung "
        "thực, kể cả khi mô hình không vượt baseline, có giá trị sử dụng cao hơn một con số "
        "đẹp nhưng bị rò rỉ."
    )

    doc.heading(2, "1.2. Mục tiêu")
    doc.paragraph(
        "Mục tiêu tổng quát của niên luận là xây dựng và đánh giá một hệ thống học máy dự "
        "báo xu hướng ngắn hạn của cổ phiếu HOSE, kèm giao diện web cho phép người dùng tra "
        "cứu và kiểm chứng lại kết quả. Bài toán được phát biểu cụ thể là phân loại nhị "
        "phân: giá đóng cửa tại đúng phiên giao dịch thứ năm sau ngày tham chiếu tăng hơn "
        "1% thì nhãn là UP, ngược lại là NOT_UP."
    )
    doc.paragraph("Các mục tiêu cụ thể, đo lường được và đã hoàn thành trong học kỳ:")
    doc.bullets(
        [
            "Xây dựng pipeline dữ liệu từ tệp OHLCV thô đến tập dữ liệu học máy, gồm kiểm "
            "tra chất lượng, làm sạch, lọc mã đủ điều kiện và sinh 20 đặc trưng kỹ thuật.",
            "Định nghĩa nhãn UP/NOT_UP theo đúng phiên thị trường thứ năm, loại bỏ các "
            "dòng không có giá tại phiên đích thay vì nhảy sang phiên kế tiếp.",
            "Chia dữ liệu TRAIN/VALIDATION/TEST theo thời gian, có purge các dòng có nhãn "
            "vắt qua ranh giới, và kiểm chứng ràng buộc chống rò rỉ bằng mã nguồn.",
            "Tinh chỉnh siêu tham số cho Logistic Regression, Random Forest và Gradient "
            "Boosting bằng kiểm định chéo chuỗi thời gian bốn fold, khoảng cách năm phiên.",
            "Chọn ngưỡng quyết định từ dự báo ngoài fold (OOF) trên TRAIN, có ràng buộc "
            "precision và tỷ lệ dự báo UP để mô hình không suy biến thành luôn dự báo UP.",
            "Chọn họ mô hình chỉ dựa trên VALIDATION, refit trên TRAIN+VALIDATION và đánh "
            "giá TEST đúng một lần, có so sánh với baseline luôn UP và luôn NOT_UP.",
            "Phát triển ứng dụng web Flask gồm trang dự báo, so sánh hai mã, xếp hạng toàn "
            "sàn, trang đánh giá mô hình, Tuning Lab và chatbot giới hạn trong dữ liệu nội bộ.",
            "Viết bộ kiểm thử tự động phủ giao thức dữ liệu, kiểm định chéo, chọn mô hình, "
            "luồng dự báo, Tuning Lab, giao diện và chatbot.",
        ]
    )
    doc.paragraph(
        "Phạm vi giới hạn: hệ thống chỉ dùng dữ liệu giá và khối lượng lịch sử, không dùng "
        "báo cáo tài chính, tin tức hay dữ liệu thời gian thực; kết quả phục vụ học tập và "
        "nghiên cứu, không phải khuyến nghị đầu tư."
    )

    doc.heading(2, "1.3. Phương pháp nghiên cứu")
    doc.paragraph(
        "Niên luận kết hợp hai phương pháp: nghiên cứu lý thuyết để chọn đúng công cụ và "
        "thực nghiệm có kiểm soát để đo năng lực thực tế của công cụ đó trên dữ liệu HOSE."
    )

    doc.heading(3, "1.3.1. Nghiên cứu lý thuyết")
    doc.paragraph(
        "Phần lý thuyết tổng hợp tài liệu về ba nhóm chủ đề. Nhóm thứ nhất là phân tích kỹ "
        "thuật, cung cấp định nghĩa của các chỉ báo được dùng làm đặc trưng: trung bình động, "
        "chỉ số sức mạnh tương đối RSI, độ biến động của lợi suất và vị thế giá trong vùng "
        "đỉnh - đáy. Nhóm thứ hai là học máy có giám sát cho bài toán phân loại nhị phân, gồm "
        "hồi quy logistic, rừng ngẫu nhiên và tăng cường gradient; với mỗi họ mô hình, tài "
        "liệu được đọc để xác định giả thiết, dạng hàm mất mát và các siêu tham số thực sự "
        "ảnh hưởng đến khả năng khái quát hóa. Nhóm thứ ba là phương pháp đánh giá cho dữ "
        "liệu chuỗi thời gian, đặc biệt là kiểm định chéo theo thời gian có purge và các cạm "
        "bẫy rò rỉ dữ liệu đã nêu ở mục 1.1. Kết quả của bước này là tập hợp các quyết định "
        "thiết kế có căn cứ, được trình bày trong chương 2 và trích dẫn theo chuẩn IEEE."
    )

    doc.heading(3, "1.3.2. Phương pháp thực nghiệm")
    doc.paragraph(
        "Phần thực nghiệm được tổ chức thành một pipeline có thứ tự cố định, mỗi bước ghi lại "
        "hiện vật ra thư mục để có thể kiểm chứng lại: kiểm tra chất lượng dữ liệu thô, làm "
        "sạch và lọc mã đủ điều kiện, sinh đặc trưng và gán nhãn, chia dữ liệu theo thời gian "
        "có purge, tinh chỉnh siêu tham số, chọn họ mô hình, rồi đánh giá trên tập TEST. "
        "Nguyên tắc thực nghiệm quan trọng nhất là tập TEST bị khóa: mọi so sánh và mọi lựa "
        "chọn đều thực hiện trên TRAIN và VALIDATION, tập TEST chỉ được mở đúng một lần cho "
        "cấu hình cuối cùng. Mỗi kết quả đều đi kèm hai baseline hằng số là luôn dự báo UP và "
        "luôn dự báo NOT_UP, vì một mô hình chỉ có ý nghĩa khi vượt được các baseline này."
    )
    doc.paragraph(
        "Toàn bộ thực nghiệm được lặp lại được nhờ ba biện pháp: cố định random_state cho mọi "
        "thành phần có yếu tố ngẫu nhiên, lưu cấu hình siêu tham số cùng lịch sử tinh chỉnh "
        "vào tệp, và tính dấu vân tay (fingerprint) của dữ liệu để phát hiện khi nào mô hình "
        "đã phát hành không còn khớp với dữ liệu hiện tại. Bên cạnh đó, các ràng buộc chống "
        "rò rỉ và các bất biến của pipeline được kiểm chứng bằng bộ kiểm thử tự động thay vì "
        "kiểm tra bằng mắt, nội dung trình bày ở mục 3.3."
    )

    doc.heading(2, "1.4. Kế hoạch thực hiện")
    doc.paragraph(
        "Công việc được chia thành mười một nhiệm vụ trải trong mười hai tuần của học kỳ, "
        "nhóm theo bốn giai đoạn: chuẩn bị và báo cáo, dữ liệu và đặc trưng, mô hình và đánh "
        "giá, ứng dụng và kiểm thử. Các nhiệm vụ có gối đầu nhau vì việc viết báo cáo và viết "
        "kiểm thử diễn ra song song với phát triển thay vì dồn vào cuối kỳ."
    )
    doc.image(
        assets / "gantt_plan.png",
        "Sơ đồ Gantt kế hoạch thực hiện niên luận",
        width_cm=15.5,
    )
    doc.paragraph(
        "Hình trên biểu diễn kế hoạch dưới dạng sơ đồ Gantt. Trục ngang là tuần thực hiện "
        "trong học kỳ, mỗi thanh ngang là một nhiệm vụ với nhãn ghi khoảng tuần tương ứng, "
        "màu của thanh cho biết nhiệm vụ thuộc giai đoạn nào. Ba tuần đầu dành cho khảo sát "
        "đề tài, thu thập dữ liệu OHLCV và kiểm tra chất lượng dữ liệu, vì chất lượng dữ liệu "
        "quyết định toàn bộ các bước sau. Giai đoạn giữa, từ tuần 4 đến tuần 8, tập trung vào "
        "sinh đặc trưng, gán nhãn, chia dữ liệu có purge và tinh chỉnh ba họ mô hình - đây là "
        "phần chiếm nhiều thời gian nhất do mỗi lần tinh chỉnh phải chạy kiểm định chéo bốn "
        "fold. Từ tuần 8 trở đi, công việc chuyển sang chọn mô hình cuối, phát triển ứng dụng "
        "Flask, viết kiểm thử tự động và hoàn thiện tài liệu; ba tuần cuối dành cho đánh giá "
        "trên tập TEST và viết báo cáo."
    )


def build_chapter2(doc: DocxBuilder, assets) -> None:
    doc.heading(1, "CHƯƠNG 2: CƠ SỞ LÝ THUYẾT")

    doc.heading(2, "2.1. Bài toán phân loại nhị phân trong dự báo chuỗi thời gian tài chính")
    doc.paragraph(
        "Dự báo giá tuyệt đối của cổ phiếu là bài toán hồi quy có phương sai rất lớn và khó "
        "kiểm chứng. Niên luận chuyển sang bài toán phân loại nhị phân trên biên độ: chỉ hỏi "
        "giá có tăng vượt một ngưỡng cho trước sau một khoảng thời gian cố định hay không. "
        "Cách phát biểu này cho phép dùng các độ đo phân loại rõ ràng (precision, recall, F1) "
        "và cho phép so sánh với baseline hằng số."
    )
    doc.paragraph(
        "Gọi C(t) là giá đóng cửa của một mã tại phiên t và t+5 là phiên giao dịch thứ năm "
        "sau t trên lịch phiên chung của toàn thị trường. Lợi suất tương lai năm phiên được "
        "định nghĩa:"
    )
    doc.equation(
        M.sub(M.r("r"), M.r("5")) + M.paren(M.r("t")) + M.r(" = ")
        + M.frac(_C("t+5"), _C("t")) + M.r(" − 1"),
        "2.1",
    )
    doc.paragraph(
        "Nhãn của bài toán và quy tắc quyết định khi suy luận lần lượt là:"
    )
    doc.equation(
        M.t("target") + M.paren(M.r("t")) + M.r(" = ")
        + M.cases(
            [
                M.r("1") + M.t("  (UP)") + M.r(",   ")
                + M.sub(M.r("r"), M.r("5")) + M.paren(M.r("t")) + M.r(" > θ"),
                M.r("0") + M.t("  (NOT_UP)") + M.r(",   ") + M.t("ngược lại"),
            ]
        ),
        "2.2",
    )
    doc.equation(
        M.t("dự báo") + M.paren(M.r("x")) + M.r(" = ")
        + M.cases(
            [
                M.t("UP") + M.r(",   s") + M.paren(M.r("x")) + M.r(" ≥ τ"),
                M.t("NOT_UP") + M.r(",   s") + M.paren(M.r("x")) + M.r(" < τ"),
            ]
        ),
        "2.3",
    )
    doc.where(
        "trong đó θ = 1% là ngưỡng tăng giá để gán nhãn UP, s(x) là Điểm UP do mô hình trả "
        "về và τ là ngưỡng quyết định được chọn từ dự báo ngoài fold (mục 2.4). Điểm cần "
        "nhấn mạnh: NOT_UP không đồng nghĩa với giá giảm, mà chỉ nghĩa là mức tăng không "
        "vượt 1% trong năm phiên. Toàn bộ giao diện và chatbot của hệ thống đều phải diễn "
        "đạt đúng nghĩa này để tránh gây hiểu sai cho người đọc kết quả."
    )

    doc.heading(2, "2.2. Đặc trưng kỹ thuật")
    doc.paragraph(
        "Hệ thống dùng 20 đặc trưng tính hoàn toàn từ dòng hiện tại và quá khứ của từng mã, "
        "chia thành năm nhóm. Mọi cửa sổ trượt đều đặt min_periods bằng độ dài cửa sổ nên "
        "các dòng đầu chuỗi chưa đủ lịch sử bị loại, không được nội suy."
    )
    doc.table(
        ["Nhóm", "Đặc trưng", "Ý nghĩa"],
        [
            ["Lợi suất", "return_1d, return_3d, return_5d, return_10d, return_20d, close_open_return",
             "Biến động giá theo nhiều khung thời gian và biên độ trong phiên"],
            ["Xu hướng", "sma5, sma20, sma50, close_vs_sma20, sma20_vs_sma50",
             "Trung bình động và vị thế giá so với xu hướng ngắn - trung hạn"],
            ["Động lượng và rủi ro", "rsi14, volatility_5d, volatility_20d, price_range",
             "Trạng thái mua/bán quá mức, độ lệch chuẩn lợi suất và biên độ giá"],
            ["Khối lượng", "volume_change_1d, volume_ratio_20",
             "Thay đổi khối lượng và khối lượng so với trung bình 20 phiên"],
            ["Vị thế và lịch", "dist_high20, dist_low20, month",
             "Khoảng cách tới đỉnh/đáy 20 phiên và yếu tố mùa vụ theo tháng"],
        ],
        widths=[1700, 3600, 3770],
        caption="Hai mươi đặc trưng kỹ thuật theo nhóm",
    )
    doc.paragraph(
        "Thứ tự 20 đặc trưng được khai báo một lần tại FEATURE_COLUMNS trong config/settings.py "
        "và trở thành hợp đồng dùng chung giữa bước sinh đặc trưng, tệp mô hình đã huấn luyện "
        "và bước suy luận. Nhờ vậy mô hình không bao giờ nhận sai thứ tự cột khi dự báo."
    )
    doc.paragraph(
        "Công thức của các đặc trưng chính được liệt kê dưới đây, với C(t) là giá đóng cửa "
        "phiên t, O(t) là giá mở cửa, H(t) và L(t) là giá cao nhất và thấp nhất, V(t) là khối "
        "lượng giao dịch, n là độ dài cửa sổ trượt. Nhóm lợi suất và biên độ trong phiên:"
    )
    doc.equation(
        M.sub(M.t("return"), M.r("n")) + M.paren(M.r("t")) + M.r(" = ")
        + M.frac(_C("t"), _C("t−n")) + M.r(" − 1"),
        "2.4",
    )
    doc.equation(
        _fn("close_open_return", "t") + M.r(" = ")
        + M.frac(_C("t"), M.r("O") + M.paren(M.r("t"))) + M.r(" − 1"),
        "2.5",
    )
    doc.paragraph("Nhóm xu hướng dựa trên trung bình động:")
    doc.equation(
        _sma("n") + M.r(" = ")
        + M.frac(M.r("1"), M.r("n"))
        + M.total(M.r("i=0"), M.r("n−1"), _C("t−i")),
        "2.6",
    )
    doc.equation(
        _fn("close_vs_sma20", "t") + M.r(" = ")
        + M.frac(_C("t"), _sma("20")) + M.r(" − 1"),
        "2.7",
    )
    doc.equation(
        _fn("sma20_vs_sma50", "t") + M.r(" = ")
        + M.frac(_sma("20"), _sma("50")) + M.r(" − 1"),
        "2.8",
    )
    doc.paragraph(
        "Nhóm động lượng và rủi ro, với r(t) là lợi suất ngày và std là độ lệch chuẩn mẫu:"
    )
    doc.equation(
        M.r("r") + M.paren(M.r("t")) + M.r(" = ")
        + M.frac(_C("t"), _C("t−1")) + M.r(" − 1"),
        "2.9",
    )
    doc.equation(
        M.sub(M.t("volatility"), M.r("n")) + M.paren(M.r("t")) + M.r(" = ")
        + M.t("std")
        + M.paren(M.r("r") + M.paren(M.r("t−n+1")) + M.r(", …, r") + M.paren(M.r("t"))),
        "2.10",
    )
    doc.equation(
        _fn("price_range", "t") + M.r(" = ")
        + M.frac(
            M.r("H") + M.paren(M.r("t")) + M.r(" − L") + M.paren(M.r("t")),
            _C("t"),
        ),
        "2.11",
    )
    doc.paragraph("Nhóm khối lượng và vị thế giá trong vùng đỉnh - đáy 20 phiên:")
    doc.equation(
        _fn("volume_change_1d", "t") + M.r(" = ")
        + M.frac(M.r("V") + M.paren(M.r("t")), M.r("V") + M.paren(M.r("t−1")))
        + M.r(" − 1"),
        "2.12",
    )
    doc.equation(
        _fn("volume_ratio_20", "t") + M.r(" = ")
        + M.frac(
            M.r("V") + M.paren(M.r("t")),
            M.frac(M.r("1"), M.r("20"))
            + M.total(M.r("i=0"), M.r("19"), M.r("V") + M.paren(M.r("t−i"))),
        ),
        "2.13",
    )
    doc.equation(
        _fn("dist_high20", "t") + M.r(" = ")
        + M.frac(
            _C("t"),
            M.sub(M.t("max"), M.r("0≤i<20")) + M.r(" H") + M.paren(M.r("t−i")),
        )
        + M.r(" − 1"),
        "2.14",
    )
    doc.equation(
        _fn("dist_low20", "t") + M.r(" = ")
        + M.frac(
            _C("t"),
            M.sub(M.t("min"), M.r("0≤i<20")) + M.r(" L") + M.paren(M.r("t−i")),
        )
        + M.r(" − 1"),
        "2.15",
    )
    doc.paragraph("Chỉ số sức mạnh tương đối RSI với cửa sổ n = 14 phiên:")
    doc.equation(
        M.t("RS") + M.paren(M.r("t")) + M.r(" = ")
        + M.frac(
            M.sub(M.t("AvgGain"), M.r("n")) + M.paren(M.r("t")),
            M.sub(M.t("AvgLoss"), M.r("n")) + M.paren(M.r("t")),
        ),
        "2.16",
    )
    doc.equation(
        M.t("RSI") + M.paren(M.r("t")) + M.r(" = 100 − ")
        + M.frac(M.r("100"), M.r("1 + ") + M.t("RS") + M.paren(M.r("t"))),
        "2.17",
    )
    doc.where(
        "trong đó AvgGain là trung bình n phiên của max(C(t) − C(t−1), 0) và AvgLoss là "
        "trung bình n phiên của max(C(t−1) − C(t), 0)."
    )
    doc.paragraph(
        "Các công thức (2.4) đến (2.17) chỉ dùng dữ liệu tại phiên t và các phiên trước đó, "
        "nên không có đặc trưng nào nhìn vào tương lai. Riêng RSI ở (2.16) - (2.17) có hai "
        "trường hợp biên cần xử lý riêng: khi AvgLoss bằng 0 thì RSI được đặt bằng 100, khi "
        "cả AvgGain và AvgLoss đều bằng 0 (giá không đổi trong toàn cửa sổ) thì RSI được đặt "
        "bằng 50. Vì vậy chỉ số RSI được cài đặt trực tiếp thay vì dùng thư viện ngoài:"
    )
    doc.code_block(
        "delta = close.diff()\n"
        "gain  = delta.clip(lower=0).rolling(14, min_periods=14).mean()\n"
        "loss  = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()\n"
        "rsi   = 100 - 100 / (1 + gain / loss)      # loss = 0  -> rsi = 100\n"
        "                                           # gain = loss = 0 -> rsi = 50"
    )

    doc.heading(2, "2.3. Ba thuật toán học máy được so sánh")
    doc.heading(3, "2.3.1. Logistic Regression")
    doc.paragraph(
        "Hồi quy logistic mô hình hóa xác suất lớp UP bằng hàm sigmoid của một tổ hợp tuyến "
        "tính các đặc trưng. Đây là mô hình tuyến tính, dễ giải thích, và trong niên luận "
        "đóng vai trò mốc tham chiếu: nếu các mô hình phi tuyến không vượt được nó thì phần "
        "phi tuyến không đóng góp thông tin. Do các đặc trưng khác nhau về thang đo, mô hình "
        "được đặt trong Pipeline cùng StandardScaler; tham số class_weight='balanced' bù lệch "
        "lớp và max_iter=1000 bảo đảm hội tụ."
    )
    doc.paragraph(
        "Với vector đặc trưng x đã chuẩn hóa, trọng số w và hệ số chệch b, mô hình và hàm mất "
        "mát có dạng:"
    )
    doc.equation(
        M.sub(M.r("z"), M.r("j")) + M.r(" = ")
        + M.frac(
            M.sub(M.r("x"), M.r("j")) + M.r(" − ") + M.sub(M.r("μ"), M.r("j")),
            M.sub(M.r("σ"), M.r("j")),
        ),
        "2.18",
    )
    doc.equation(
        M.r("σ") + M.paren(M.r("u")) + M.r(" = ")
        + M.frac(M.r("1"), M.r("1 + ") + M.sup(M.r("e"), M.r("−u"))),
        "2.19",
    )
    doc.equation(
        M.sub(M.r("s"), M.t("UP")) + M.paren(M.r("x")) + M.r(" = σ")
        + M.paren(M.sup(M.r("w"), M.r("T")) + M.r("z + b")),
        "2.20",
    )
    doc.equation(
        M.r("J") + M.paren(M.r("w, b")) + M.r(" = −")
        + M.frac(M.r("1"), M.r("N"))
        + M.total(M.r("i=1"), M.r("N"),
                  M.sub(M.r("c"), M.sub(M.r("y"), M.r("i")))
                  + M.paren(
                      M.sub(M.r("y"), M.r("i")) + M.t(" log ") + M.sub(M.r("p"), M.r("i"))
                      + M.r(" + ")
                      + M.paren(M.r("1 − ") + M.sub(M.r("y"), M.r("i")))
                      + M.t(" log")
                      + M.paren(M.r("1 − ") + M.sub(M.r("p"), M.r("i"))),
                      beg="[", end="]",
                  ))
        + M.r(" + ")
        + M.frac(M.r("1"), M.r("2C"))
        + M.sup(M.norm(M.r("w")), M.r("2")),
        "2.21",
    )
    doc.where(
        "trong đó μ_j và σ_j chỉ được tính trên phần TRAIN của từng fold, p_i = s_UP(x_i), "
        "c_y = N / (2·N_y) là trọng số lớp sinh từ class_weight='balanced' và C là nghịch "
        "đảo cường độ chính quy hóa L2."
    )
    doc.paragraph(
        "Công thức (2.18) giải thích lý do phải đặt StandardScaler bên trong Pipeline thay vì "
        "chuẩn hóa toàn bộ dữ liệu trước khi chia fold: nếu tính trung bình và độ lệch chuẩn "
        "trên cả tập dữ liệu thì thống kê của tập kiểm định đã rò rỉ vào bước huấn luyện. "
        "Trọng số c_y trong (2.21) khiến mỗi dòng thuộc lớp thiểu số UP đóng góp nhiều hơn "
        "vào hàm mất mát, bù cho tỷ lệ UP chỉ 37,6%."
    )
    doc.heading(3, "2.3.2. Random Forest")
    doc.paragraph(
        "Random Forest là tập hợp nhiều cây quyết định huấn luyện trên các mẫu bootstrap và "
        "tập con đặc trưng ngẫu nhiên, dự báo bằng bình quân xác suất các cây. Cơ chế này "
        "giảm phương sai của cây đơn và bắt được quan hệ phi tuyến cũng như tương tác giữa "
        "các đặc trưng, đồng thời cung cấp độ quan trọng đặc trưng phục vụ phân tích. Tham "
        "số class_weight='balanced_subsample' xử lý mất cân bằng lớp trong từng mẫu bootstrap."
    )
    doc.paragraph(
        "Mỗi cây được nuôi bằng cách chọn điểm chia làm giảm mạnh nhất độ vẩn Gini của nút; "
        "xác suất của rừng là bình quân xác suất của các cây:"
    )
    doc.equation(
        M.r("G") + M.paren(M.r("m")) + M.r(" = 1 − ")
        + M.total(M.r("k"), "",
                  M.sup(M.sub(M.r("p"), M.r("m,k")), M.r("2"))),
        "2.22",
    )
    doc.equation(
        M.r("ΔG = G") + M.paren(M.r("m"))
        + M.r(" − ")
        + M.frac(M.sub(M.r("N"), M.r("L")), M.sub(M.r("N"), M.r("m")))
        + M.r("G") + M.paren(M.r("L"))
        + M.r(" − ")
        + M.frac(M.sub(M.r("N"), M.r("R")), M.sub(M.r("N"), M.r("m")))
        + M.r("G") + M.paren(M.r("R")),
        "2.23",
    )
    doc.equation(
        M.sub(M.r("s"), M.t("UP")) + M.paren(M.r("x")) + M.r(" = ")
        + M.frac(M.r("1"), M.r("T"))
        + M.total(M.r("t=1"), M.r("T"),
                  M.sub(M.r("p"), M.r("t"))
                  + M.paren(M.t("UP") + M.r(" | x"))),
        "2.24",
    )
    doc.equation(
        M.t("imp") + M.paren(M.r("j")) + M.r(" = ")
        + M.frac(M.r("1"), M.r("T"))
        + M.total(M.r("t=1"), M.r("T"),
                  M.total(M.r("m ∈ ") + M.sub(M.r("S"), M.r("j")), "",
                          M.frac(M.sub(M.r("N"), M.r("m")), M.r("N"))
                          + M.r("ΔG") + M.paren(M.r("m")))),
        "2.25",
    )
    doc.where(
        "trong đó p_{m,k} là tỷ lệ (có trọng số lớp) của lớp k trong nút m; N_m, N_L, N_R là "
        "số mẫu của nút và hai nút con; T = n_estimators; p_t(UP | x) là tỷ lệ lớp UP tại lá "
        "mà x rơi vào ở cây t; S_j là tập các nút chia theo đặc trưng j. Điểm chia tại mỗi "
        "nút được chọn để cực đại ΔG trên một tập con đặc trưng ngẫu nhiên kích thước "
        "max_features × 20."
    )
    doc.paragraph(
        "Công thức (2.23) cho thấy vai trò của max_features: chỉ một tập con đặc trưng được "
        "xét tại mỗi nút, nhờ đó các cây trong rừng ít tương quan với nhau và phương sai của "
        "bình quân (2.24) giảm. Công thức (2.25) là cơ sở của biểu đồ độ quan trọng đặc trưng "
        "trình bày ở mục 3.3 và trang đánh giá mô hình. Ràng buộc min_samples_leaf đặt sàn "
        "cho N_m nên cây không tách tới các lá chỉ có vài dòng nhiễu."
    )
    doc.heading(3, "2.3.3. Gradient Boosting")
    doc.paragraph(
        "Gradient Boosting cộng dồn các cây nông theo hướng giảm gradient của hàm mất mát, "
        "mỗi cây sửa phần dư của tổ hợp trước đó. Mô hình thường mạnh trên dữ liệu dạng bảng "
        "nhưng nhạy với learning_rate và số stage nên dễ quá khớp. Vì "
        "GradientBoostingClassifier của scikit-learn không có tham số class_weight, hệ thống "
        "truyền sample_weight tính bằng compute_sample_weight('balanced', y) khi fit."
    )
    doc.paragraph(
        "Mô hình được xây dựng theo từng stage trên thang log-odds. Gọi F_M(x) là tổ hợp sau M "
        "stage, hàm mất mát là log loss có trọng số mẫu:"
    )
    doc.equation(
        M.r("L") + M.paren(M.r("y, F")) + M.r(" = −")
        + M.paren(
            M.r("y") + M.t(" log ") + M.r("σ") + M.paren(M.r("F"))
            + M.r(" + ")
            + M.paren(M.r("1 − y")) + M.t(" log")
            + M.paren(M.r("1 − σ") + M.paren(M.r("F"))),
            beg="[", end="]",
        ),
        "2.26",
    )
    doc.equation(
        M.sub(M.r("F"), M.r("0")) + M.paren(M.r("x")) + M.r(" = ")
        + M.t("log")
        + M.paren(
            M.frac(
                M.sub(M.r("p"), M.t("UP")),
                M.r("1 − ") + M.sub(M.r("p"), M.t("UP")),
            )
        ),
        "2.27",
    )
    doc.equation(
        M.sub(M.r("g"), M.r("i")) + M.r(" = ")
        + M.sub(M.r("y"), M.r("i")) + M.r(" − σ")
        + M.paren(
            M.sub(M.r("F"), M.r("m−1"))
            + M.paren(M.sub(M.r("x"), M.r("i")))
        ),
        "2.28",
    )
    doc.equation(
        M.sub(M.r("F"), M.r("m")) + M.paren(M.r("x")) + M.r(" = ")
        + M.sub(M.r("F"), M.r("m−1")) + M.paren(M.r("x"))
        + M.r(" + ν · ")
        + M.sub(M.r("h"), M.r("m")) + M.paren(M.r("x")),
        "2.29",
    )
    doc.equation(
        M.sub(M.r("s"), M.t("UP")) + M.paren(M.r("x")) + M.r(" = σ")
        + M.paren(M.sub(M.r("F"), M.r("M")) + M.paren(M.r("x"))),
        "2.30",
    )
    doc.equation(
        M.sub(M.r("w"), M.r("i")) + M.r(" = ")
        + M.frac(
            M.r("N"),
            M.r("2 · ") + M.sub(M.r("N"), M.sub(M.r("y"), M.r("i"))),
        ),
        "2.31",
    )
    doc.where(
        "trong đó h_m là cây hồi quy được fit trên các cặp (x_i, g_i) với trọng số mẫu w_i, "
        "ν = learning_rate và số stage m chạy từ 1 đến n_estimators."
    )
    doc.paragraph(
        "Công thức (2.28) là điểm khác biệt cốt lõi so với Random Forest: cây thứ m không học "
        "lại nhãn gốc mà học phần sai số còn lại của tổ hợp trước đó, nên các cây phụ thuộc "
        "nhau và phải huấn luyện tuần tự. Hệ số ν trong (2.29) co nhỏ mức đóng góp của mỗi "
        "cây; ν lớn kết hợp nhiều stage khiến mô hình khớp cả nhiễu, đó là lý do learning_rate "
        "và n_estimators phải được tinh chỉnh cùng nhau. Công thức (2.31) thay cho tham số "
        "class_weight mà GradientBoostingClassifier không có."
    )

    doc.heading(2, "2.4. Kiểm định chéo chuỗi thời gian có purge")
    doc.paragraph(
        "Kiểm định chéo k-fold ngẫu nhiên không dùng được cho dữ liệu bảng theo thời gian vì "
        "hai lý do. Thứ nhất, dòng dữ liệu tương lai sẽ được dùng để dự báo quá khứ. Thứ hai, "
        "cùng một phiên giao dịch có gần 400 dòng của 400 mã, nếu chia theo dòng thì cùng một "
        "ngày sẽ nằm ở cả hai phía. Hệ thống dùng TimeSeriesSplit trên danh sách phiên giao "
        "dịch duy nhất, cộng thêm hai lớp bảo vệ:"
    )
    doc.bullets(
        [
            "gap = 5 phiên: bỏ đúng năm phiên giao dịch chung ngay trước tập kiểm định, "
            "tương ứng đúng độ dài chân trời dự báo.",
            "purge nhãn: loại các dòng huấn luyện có label_end_date lớn hơn hoặc bằng phiên "
            "bắt đầu tập kiểm định, vì nhãn của chúng đã kết thúc trong vùng kiểm định.",
        ]
    )
    doc.paragraph(
        "Vòng lặp chia fold được cài đặt tại services/time_splitting.py và dùng chung cho cả "
        "Tuning Lab lẫn pipeline chính thức, nên hai nơi không thể lệch giao thức:"
    )
    doc.code_block(
        "market_dates = np.sort(dates.unique())\n"
        "market_dates = market_dates[market_dates >= pd.Timestamp(CV_START_DATE)]\n"
        "splitter = TimeSeriesSplit(n_splits=4, gap=5)\n"
        "for train_date_idx, val_date_idx in splitter.split(market_dates):\n"
        "    validation_start = market_dates[val_date_idx][0]\n"
        "    train_mask = dates.isin(train_dates) & (label_ends < validation_start)\n"
        "    validation_mask = dates.isin(market_dates[val_date_idx])"
    )

    doc.heading(2, "2.5. Độ đo đánh giá và baseline hằng số")
    doc.paragraph(
        "Tập dữ liệu lệch lớp: 37,6% dòng có nhãn UP. Với tỷ lệ này, accuracy là độ đo gây "
        "nhầm lẫn vì một mô hình luôn trả NOT_UP đã đạt hơn 62% accuracy mà không dự báo được "
        "gì. Hệ thống chọn F1_UP làm độ đo chính, kèm precision_up và recall_up, và luôn báo "
        "cáo hai baseline hằng số trên cả VALIDATION và TEST: luôn dự báo UP và luôn dự báo "
        "NOT_UP. Một mô hình chỉ được xem là có ích khi F1_UP vượt baseline luôn UP."
    )
    doc.paragraph(
        "Gọi TP là số dòng nhãn UP được dự báo UP, FP là số dòng nhãn NOT_UP bị dự báo UP, FN "
        "là số dòng nhãn UP bị dự báo NOT_UP, TN là số dòng NOT_UP được dự báo đúng, N là tổng "
        "số dòng và π_UP là tỷ lệ UP thực tế. Các độ đo được định nghĩa:"
    )
    doc.equation(
        M.sub(M.t("precision"), M.t("UP")) + M.r(" = ")
        + M.frac(M.t("TP"), M.t("TP") + M.r(" + ") + M.t("FP")),
        "2.32",
    )
    doc.equation(
        M.sub(M.t("recall"), M.t("UP")) + M.r(" = ")
        + M.frac(M.t("TP"), M.t("TP") + M.r(" + ") + M.t("FN")),
        "2.33",
    )
    doc.equation(
        M.sub(M.t("F1"), M.t("UP")) + M.r(" = ")
        + M.frac(
            M.r("2 · ") + M.sub(M.t("precision"), M.t("UP"))
            + M.r(" · ") + M.sub(M.t("recall"), M.t("UP")),
            M.sub(M.t("precision"), M.t("UP")) + M.r(" + ")
            + M.sub(M.t("recall"), M.t("UP")),
        ),
        "2.34",
    )
    doc.equation(
        M.t("accuracy") + M.r(" = ")
        + M.frac(M.t("TP") + M.r(" + ") + M.t("TN"), M.r("N")),
        "2.35",
    )
    doc.equation(
        M.t("up_rate") + M.r(" = ")
        + M.frac(M.t("TP") + M.r(" + ") + M.t("FP"), M.r("N")),
        "2.36",
    )
    doc.paragraph(
        "Hai baseline hằng số có dạng đóng. Baseline luôn dự báo UP có precision bằng π_UP và "
        "recall bằng 1, nên:"
    )
    doc.equation(
        M.sup(M.sub(M.t("F1"), M.t("UP")), M.t("luôn UP")) + M.r(" = ")
        + M.frac(
            M.r("2") + M.sub(M.r("π"), M.t("UP")),
            M.r("1 + ") + M.sub(M.r("π"), M.t("UP")),
        ),
        "2.37",
    )
    doc.paragraph(
        "Baseline luôn dự báo NOT_UP có TP bằng 0 nên F1_UP bằng 0, trong khi accuracy đạt:"
    )
    doc.equation(
        M.sup(M.t("accuracy"), M.t("luôn NOT_UP")) + M.r(" = 1 − ")
        + M.sub(M.r("π"), M.t("UP")),
        "2.38",
    )
    doc.paragraph(
        "Thay π_UP = 0,376 vào (2.37) cho F1_UP của baseline luôn UP bằng 0,546, còn (2.38) "
        "cho accuracy 0,624 với F1_UP bằng 0. Hai con số này là mốc so sánh bắt buộc trong "
        "chương 3: một mô hình có accuracy 0,60 và F1_UP 0,45 tuy nghe khá nhưng thực chất còn "
        "kém cả hai baseline. Ràng buộc up_rate ở (2.36) được dùng khi chọn ngưỡng quyết định "
        "để mô hình không suy biến thành luôn dự báo UP nhằm ăn điểm recall."
    )

    doc.heading(2, "2.6. Công cụ và công nghệ sử dụng")
    doc.table(
        ["Thành phần", "Phiên bản", "Vai trò trong hệ thống", "Lý do chọn"],
        [
            ["Python", "3.12", "Ngôn ngữ cài đặt toàn hệ thống",
             "Hệ sinh thái khoa học dữ liệu và web đầy đủ trong cùng một ngôn ngữ"],
            ["pandas", "2.3.3", "Đọc, làm sạch, sinh đặc trưng và gán nhãn",
             "Xử lý dữ liệu bảng theo nhóm và theo thời gian hiệu quả"],
            ["NumPy", "2.2.6", "Tính toán số học và mặt nạ chỉ số fold",
             "Nền tảng vector hóa cho pandas và scikit-learn"],
            ["scikit-learn", "1.8.0", "Ba mô hình, TimeSeriesSplit, độ đo phân loại",
             "API nhất quán, có clone/Pipeline giúp tránh rò rỉ khi chuẩn hóa"],
            ["joblib", "1.5.3", "Lưu và nạp tệp mô hình final_model.pkl",
             "Tuần tự hóa hiệu quả cho đối tượng scikit-learn"],
            ["matplotlib", "3.10.9", "Vẽ ma trận nhầm lẫn xuất ra PNG",
             "Đủ cho biểu đồ tĩnh nhúng vào báo cáo và giao diện"],
            ["Flask", "3.1.3", "Máy chủ web, các route giao diện và API chat",
             "Nhẹ, phù hợp ứng dụng một tiến trình chạy nội bộ"],
            ["vnstock", "4.0.4", "Tải dữ liệu OHLCV bổ sung theo từng mã",
             "Thư viện dữ liệu chứng khoán Việt Nam, có nhiều nguồn dự phòng"],
            ["openai (client)", "2.48.0", "Gọi Chat Completions cho chatbot",
             "Chuẩn API phổ biến, trỏ được sang endpoint cấu hình qua biến môi trường"],
            ["python-dotenv", "1.2.2", "Nạp cấu hình LLM từ tệp .env",
             "Giữ khóa API ngoài mã nguồn"],
            ["Chart.js", "bản dựng UMD kèm theo", "Vẽ biểu đồ tương tác trên giao diện",
             "Chạy hoàn toàn cục bộ, không cần CDN"],
        ],
        widths=[1500, 1300, 3000, 3270],
        caption="Công cụ, thư viện sử dụng và lý do lựa chọn",
    )
    doc.paragraph(
        "Toàn bộ đường dẫn, danh sách đặc trưng, ranh giới thời gian và tham số giao thức "
        "được tập trung tại config/settings.py. Các module khác import từ đó thay vì ghi cứng "
        "giá trị, nên khi đổi bài toán dự báo chỉ cần sửa một tệp cấu hình."
    )
