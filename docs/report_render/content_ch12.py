"""Chuong 1 (Gioi thieu) va Chuong 2 (Co so ly thuyet)."""

from __future__ import annotations

from docx_builder import DocxBuilder


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
    doc.paragraph("Định nghĩa nhãn và quy tắc quyết định của hệ thống:")
    doc.code_block(
        "target = 1 (UP)  neu  close(t+5 phien thi truong) / close(t) - 1 > 0.01\n"
        "target = 0 (NOT_UP) trong cac truong hop con lai\n"
        "du bao   = UP  neu  score_up >= decision_threshold"
    )
    doc.paragraph(
        "Điểm cần nhấn mạnh: NOT_UP không đồng nghĩa với giá giảm. NOT_UP chỉ nghĩa là mức "
        "tăng không vượt 1% trong năm phiên. Toàn bộ giao diện và chatbot của hệ thống đều "
        "phải diễn đạt đúng nghĩa này để tránh gây hiểu sai cho người đọc kết quả."
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
        "Công thức của các đặc trưng chính được liệt kê dưới đây, với P(t) là giá đóng cửa "
        "phiên t, O(t) là giá mở cửa, H(t) và L(t) là giá cao nhất và thấp nhất, V(t) là khối "
        "lượng giao dịch, n là độ dài cửa sổ trượt:"
    )
    doc.code_block(
        "(1)  Loi suat n phien:      return_nd(t) = P(t) / P(t-n) - 1\n"
        "(2)  Bien do trong phien:   close_open_return(t) = P(t) / O(t) - 1\n"
        "(3)  Trung binh dong:       SMA_n(t) = (1/n) * SUM_{i=0..n-1} P(t-i)\n"
        "(4)  Vi the so voi SMA:     close_vs_sma20(t) = P(t) / SMA_20(t) - 1\n"
        "(5)  Do doc xu huong:       sma20_vs_sma50(t) = SMA_20(t) / SMA_50(t) - 1\n"
        "(6)  Loi suat ngay:         r(t) = P(t) / P(t-1) - 1\n"
        "(7)  Do bien dong:          volatility_nd(t) = std({ r(t-i) : i = 0..n-1 })\n"
        "(8)  Bien do gia:           price_range(t) = (H(t) - L(t)) / P(t)\n"
        "(9)  Thay doi khoi luong:   volume_change_1d(t) = V(t) / V(t-1) - 1\n"
        "(10) Khoi luong tuong doi:  volume_ratio_20(t) = V(t) / ((1/20) * SUM V(t-i))\n"
        "(11) Khoang cach dinh 20:   dist_high20(t) = P(t) / max{ H(t-i) : i<20 } - 1\n"
        "(12) Khoang cach day 20:    dist_low20(t)  = P(t) / min{ L(t-i) : i<20 } - 1\n"
        "(13) RSI n phien:           RS(t)  = AvgGain_n(t) / AvgLoss_n(t)\n"
        "                            RSI(t) = 100 - 100 / (1 + RS(t))\n"
        "     voi AvgGain_n(t) = trung binh n phien cua max(P(t)-P(t-1), 0)\n"
        "         AvgLoss_n(t) = trung binh n phien cua max(P(t-1)-P(t), 0)"
    )
    doc.paragraph(
        "Các công thức (1) đến (13) chỉ dùng dữ liệu tại phiên t và các phiên trước đó, nên "
        "không có đặc trưng nào nhìn vào tương lai. Riêng công thức (13) có hai trường hợp "
        "biên cần xử lý riêng: khi AvgLoss bằng 0 thì RSI được đặt bằng 100, khi cả AvgGain và "
        "AvgLoss đều bằng 0 (giá không đổi trong toàn cửa sổ) thì RSI được đặt bằng 50. Vì vậy "
        "chỉ số RSI được cài đặt trực tiếp thay vì dùng thư viện ngoài:"
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
    doc.code_block(
        "(14) Chuan hoa dac trung:  z_j = (x_j - mean_j) / std_j\n"
        "                           mean_j, std_j chi tinh tren TRAIN cua tung fold\n"
        "(15) Ham sigmoid:          sigma(u) = 1 / (1 + exp(-u))\n"
        "(16) Xac suat lop UP:      score_up(x) = sigma(w . z + b)\n"
        "(17) Ham mat mat (L2):     J(w, b) = -(1/N) * SUM_i c_{y_i} * [\n"
        "                               y_i * log(p_i) + (1-y_i) * log(1-p_i) ]\n"
        "                             + (1 / (2*C)) * ||w||^2\n"
        "     voi p_i = score_up(x_i), c_{y} = trong so lop tu class_weight='balanced',\n"
        "         c_y = N / (2 * N_y), C = nghich dao cuong do chinh quy hoa"
    )
    doc.paragraph(
        "Công thức (14) giải thích lý do phải đặt StandardScaler bên trong Pipeline thay vì "
        "chuẩn hóa toàn bộ dữ liệu trước khi chia fold: nếu tính mean và std trên cả tập dữ"
        "liệu thì thống kê của tập kiểm định đã rò rỉ vào bước huấn luyện. Trọng số c_y trong "
        "(17) khiến mỗi dòng thuộc lớp thiểu số UP đóng góp nhiều hơn vào hàm mất mát, bù cho "
        "tỷ lệ UP chỉ 37,6%."
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
    doc.code_block(
        "(18) Do van Gini cua nut m:  G(m) = 1 - SUM_k p_{m,k}^2\n"
        "     voi p_{m,k} = ty le (co trong so lop) cua lop k trong nut m\n"
        "(19) Do giam do van:         dG = G(m) - (N_L/N_m) * G(L) - (N_R/N_m) * G(R)\n"
        "     diem chia duoc chon la diem lam cuc dai dG tren tap con dac trung\n"
        "     co kich thuoc max_features * 20\n"
        "(20) Xac suat cua rung:      score_up(x) = (1/T) * SUM_{t=1..T} p_t(UP | x)\n"
        "     voi T = n_estimators, p_t = ty le lop UP tai la ma x roi vao o cay t\n"
        "(21) Do quan trong dac trung: imp(j) = (1/T) * SUM_t SUM_{m: chia theo j}\n"
        "                                        (N_m / N) * dG(m)"
    )
    doc.paragraph(
        "Công thức (19) cho thấy vai trò của max_features: chỉ một tập con đặc trưng được xét "
        "tại mỗi nút, nhờ đó các cây trong rừng ít tương quan với nhau và phương sai của bình "
        "quân (20) giảm. Công thức (21) là cơ sở của biểu đồ độ quan trọng đặc trưng trình bày "
        "ở mục 3.2 và trang đánh giá mô hình. Ràng buộc min_samples_leaf đặt sàn cho N_m nên "
        "cây không tách tới các lá chỉ có vài dòng nhiễu."
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
    doc.code_block(
        "(22) Ham mat mat:        L(y, F) = -[ y * log(sigma(F)) + (1-y) * log(1-sigma(F)) ]\n"
        "(23) Khoi tao:           F_0(x) = log( p_UP / (1 - p_UP) )\n"
        "(24) Phan du gia (gradient am tai stage m):\n"
        "                         g_i = y_i - sigma(F_{m-1}(x_i))\n"
        "(25) Cap nhat:           F_m(x) = F_{m-1}(x) + nu * h_m(x)\n"
        "     voi h_m = cay hoi quy fit tren { (x_i, g_i) } co trong so w_i,\n"
        "         nu = learning_rate, so stage m = 1..n_estimators\n"
        "(26) Xac suat cuoi:      score_up(x) = sigma(F_M(x))\n"
        "(27) Trong so mau can bang lop:  w_i = N / (2 * N_{y_i})"
    )
    doc.paragraph(
        "Công thức (24) là điểm khác biệt cốt lõi so với Random Forest: cây thứ m không học lại "
        "nhãn gốc mà học phần sai số còn lại của tổ hợp trước đó, nên các cây phụ thuộc nhau và "
        "phải huấn luyện tuần tự. Hệ số nu trong (25) co nhỏ mức đóng góp của mỗi cây; nu lớn "
        "kết hợp nhiều stage khiến mô hình khớp cả nhiễu, đó là lý do learning_rate và "
        "n_estimators phải được tinh chỉnh cùng nhau. Công thức (27) thay cho tham số "
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
        "số dòng và pi_UP là tỷ lệ UP thực tế. Các độ đo được định nghĩa:"
    )
    doc.code_block(
        "(28) Precision lop UP:  precision_up = TP / (TP + FP)\n"
        "(29) Recall lop UP:     recall_up    = TP / (TP + FN)\n"
        "(30) F1 lop UP:         F1_UP = 2 * precision_up * recall_up\n"
        "                                / (precision_up + recall_up)\n"
        "(31) Accuracy:          accuracy = (TP + TN) / N\n"
        "(32) Ty le du bao UP:   up_rate = (TP + FP) / N\n"
        "(33) Baseline luon UP:      precision = pi_UP, recall = 1,\n"
        "                            F1_UP = 2 * pi_UP / (1 + pi_UP)\n"
        "(34) Baseline luon NOT_UP:  TP = 0 nen F1_UP = 0,\n"
        "                            accuracy = 1 - pi_UP"
    )
    doc.paragraph(
        "Thay pi_UP = 0,376 vào (33) cho F1_UP của baseline luôn UP bằng 0,546, còn (34) cho "
        "accuracy 0,624 với F1_UP bằng 0. Hai con số này là mốc so sánh bắt buộc trong chương 3: "
        "một mô hình có accuracy 0,60 và F1_UP 0,45 tuy nghe khá nhưng thực chất còn kém cả hai "
        "baseline. Ràng buộc up_rate ở (32) được dùng khi chọn ngưỡng quyết định để mô hình "
        "không suy biến thành luôn dự báo UP nhằm ăn điểm recall."
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
