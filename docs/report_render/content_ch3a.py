"""Chuong 3 phan A: dac ta yeu cau phan mem."""

from __future__ import annotations

from docx_builder import DocxBuilder


def build_srs(doc: DocxBuilder, assets) -> None:
    doc.heading(1, "CHƯƠNG 3: KẾT QUẢ ỨNG DỤNG")

    doc.heading(2, "3.1. Đặc tả yêu cầu phần mềm")

    doc.heading(3, "3.1.1. Mô tả tổng quan")
    doc.paragraph(
        "Hệ thống có tên HOSE Stock Trend Prediction, chạy hoàn toàn cục bộ trên máy của "
        "người dùng, không phụ thuộc dịch vụ trả phí nào ngoài phần chatbot tùy chọn. Sản "
        "phẩm gồm ba khối: một pipeline xử lý dữ liệu và huấn luyện mô hình chạy theo lệnh, "
        "một ứng dụng web Flask phục vụ tra cứu và dự báo, và một lớp công cụ truy vấn dữ "
        "liệu nội bộ cho chatbot."
    )
    doc.paragraph(
        "Người dùng mục tiêu là sinh viên hoặc nhà đầu tư cá nhân muốn xem tín hiệu kỹ "
        "thuật đã được mô hình hóa cho toàn sàn HOSE, kèm thông tin minh bạch về chất lượng "
        "mô hình. Hệ thống không đưa khuyến nghị mua bán; đầu ra là nhãn UP hoặc NOT_UP kèm "
        "Điểm UP, luôn hiển thị cùng cảnh báo khi mô hình chưa vượt baseline."
    )
    doc.paragraph(
        "Giả định và ràng buộc: dữ liệu OHLCV lịch sử của sàn HOSE được lưu tại một tệp CSV "
        "dùng chung ngoài repo; mọi dự báo là suy luận ngoại tuyến trên dữ liệu đã tải, "
        "không phải giá thời gian thực; ứng dụng web chạy một tiến trình duy nhất nên chỉ "
        "cho phép tối đa một tác vụ tuning hoặc một lần chạy pipeline tại một thời điểm."
    )
    doc.table(
        ["Thành phần", "Tệp/thư mục chính", "Trách nhiệm"],
        [
            ["Cấu hình trung tâm", "config/settings.py",
             "Đường dẫn, 20 đặc trưng, ranh giới thời gian, tham số CV, ngưỡng UP"],
            ["Tiền xử lý", "services/preprocessing.py",
             "Kiểm tra dữ liệu thô, làm sạch OHLCV, lọc mã đủ điều kiện huấn luyện"],
            ["Sinh đặc trưng và nhãn", "services/feature_engineering.py",
             "Tính 20 đặc trưng theo từng mã, gán nhãn t+5 theo lịch phiên chung, chia split"],
            ["Chia thời gian", "services/protocol_dates.py, services/time_splitting.py",
             "Suy ra ba mốc rolling và sinh fold CV theo ngày có purge"],
            ["Tinh chỉnh mô hình", "services/model_tuning.py, services/tuning_lab.py",
             "Dựng estimator, chạy CV, chọn ngưỡng quyết định từ OOF, ghi lịch sử tuning"],
            ["Đánh giá và phát hành", "services/model_evaluation.py",
             "Chấm VALIDATION, chọn họ mô hình, refit, đánh giá TEST một lần, ghi báo cáo"],
            ["Trạng thái thực nghiệm", "services/experiment_state.py",
             "Fingerprint dữ liệu, lịch sử tuning, registry đánh giá, lock pipeline"],
            ["Suy luận", "services/prediction_service.py, scripts/predict_stock.py",
             "Nạp artifact, tính đặc trưng mới nhất, trả nhãn và Điểm UP"],
            ["Chatbot", "services/chatbot_service.py, services/chatbot_tools.py",
             "Định tuyến bằng luật, dựng context có cấu trúc từ dữ liệu nội bộ, gọi mô hình một lần"],
            ["Giao diện web", "app.py, templates/, static/",
             "Bảy trang giao diện và các endpoint POST cho tuning, fetch, chat"],
        ],
        widths=[1800, 3100, 4170],
        caption="Các thành phần chính của hệ thống",
    )

    doc.heading(3, "3.1.2. Mô hình use case")
    doc.paragraph(
        "Hệ thống có hai tác nhân. Người dùng (nhà đầu tư) tra cứu kết quả dự báo và thông "
        "tin đánh giá mô hình. Người quản trị mô hình là người vận hành pipeline: tinh chỉnh "
        "siêu tham số, chạy pipeline để phát hành mô hình mới và cập nhật dữ liệu OHLCV. Hai "
        "tác nhân dùng chung một ứng dụng web, phân biệt theo trang chức năng chứ không theo "
        "cơ chế đăng nhập, vì hệ thống chạy cục bộ trên máy cá nhân."
    )
    doc.image(
        assets / "use_case_diagram.png",
        "Lược đồ use case của hệ thống",
        width_cm=13.0,
    )
    doc.paragraph(
        "Lược đồ trên đặt tám use case trong biên hệ thống, tác nhân Người dùng ở bên trái và "
        "tác nhân Người quản trị mô hình ở bên phải. Người dùng truy cập nhóm use case tra "
        "cứu, từ UC-01 đến UC-05. Người quản trị mô hình truy cập nhóm vận hành gồm UC-06, "
        "UC-07, UC-08 và cũng xem được UC-04 để kiểm tra chất lượng mô hình sau khi phát hành. "
        "Quan hệ include từ UC-07 tới UC-06 thể hiện việc chạy pipeline luôn sử dụng lại cấu "
        "hình siêu tham số đã được lưu từ Tuning Lab thay vì nhập lại tham số."
    )
    doc.table(
        ["Mã", "Use case", "Tác nhân", "Mô tả ngắn"],
        [
            ["UC-01", "Dự báo xu hướng một mã", "Người dùng",
             "Nhập mã cổ phiếu tại trang chủ, nhận nhãn UP/NOT_UP, Điểm UP, ngưỡng quyết "
             "định và cảnh báo baseline"],
            ["UC-02", "So sánh hai mã", "Người dùng",
             "Nhập hai mã, xem bảng đối chiếu Điểm UP và khoảng cách tới ngưỡng"],
            ["UC-03", "Xếp hạng toàn sàn", "Người dùng",
             "Xem danh sách các mã đủ điều kiện, sắp theo Điểm UP giảm dần"],
            ["UC-04", "Xem kết quả đánh giá mô hình", "Người dùng, Người quản trị",
             "Xem tách biệt kết quả CV, VALIDATION và TEST một lần, ma trận nhầm lẫn, độ "
             "quan trọng đặc trưng"],
            ["UC-05", "Tra cứu qua chatbot", "Người dùng",
             "Đặt câu hỏi bằng ngôn ngữ tự nhiên, hệ thống trả lời dựa trên dữ liệu nội bộ "
             "được nạp sẵn vào ngữ cảnh"],
            ["UC-06", "Tinh chỉnh siêu tham số", "Người quản trị",
             "Nhập cấu hình cho từng họ mô hình tại Tuning Lab, chạy CV 4 fold có purge, lưu "
             "kết quả kèm fingerprint dữ liệu"],
            ["UC-07", "Chạy pipeline và phát hành mô hình", "Người quản trị",
             "Chạy chuỗi bước từ tiền xử lý tới đánh giá TEST, phát hành artifact nguyên tử "
             "nếu vượt kiểm tra"],
            ["UC-08", "Cập nhật dữ liệu OHLCV", "Người quản trị",
             "Chạy tác vụ nền tải dữ liệu bổ sung theo từng mã, theo dõi tiến độ và lỗi"],
        ],
        widths=[800, 2300, 1900, 4070],
        caption="Đặc tả tóm lược các use case",
    )

    doc.heading(3, "3.1.3. Yêu cầu cụ thể")
    doc.paragraph(
        "Bài toán được đặc tả bằng bốn quy tắc bắt buộc, mọi thành phần trong hệ thống đều "
        "phải tuân theo cùng một định nghĩa:"
    )
    doc.code_block(
        "target      = (close[t+5 phiên thị trường] / close[t] - 1) > 0.01\n"
        "nhãn        = UP nếu target đúng, ngược lại NOT_UP\n"
        "dự báo      = UP nếu score_up >= decision_threshold\n"
        "score_up    = predict_proba(...)[lớp 1], hiển thị dưới tên Điểm UP"
    )
    doc.bullets([
        "Chân trời dự báo là 5 phiên, lấy trên lịch phiên chung của toàn sàn, không phải "
        "dòng thứ năm của riêng từng mã.",
        "Ngưỡng tăng giá để gán nhãn UP là 1%.",
        "Ngưỡng quyết định được chọn từ xác suất out-of-fold trên TRAIN, lưu trong artifact "
        "và dùng lại y nguyên ở CV, VALIDATION, TEST, CLI và giao diện; mô hình đang phục vụ "
        "dùng ngưỡng 0,49.",
        "Nhãn NOT_UP chỉ có nghĩa chưa đạt điều kiện UP, không đồng nghĩa giá sẽ giảm.",
    ])

    doc.heading(3, "3.1.4. Yêu cầu chức năng")
    doc.table(
        ["Mã", "Chức năng", "Mô tả và điểm truy cập"],
        [
            ["FR-01", "Kiểm tra và làm sạch dữ liệu",
             "Đọc CSV thô, thống kê thiếu dữ liệu, trùng lặp, dòng OHLC sai; loại dòng lỗi "
             "và lọc mã có ít nhất 250 phiên (services/preprocessing.py)"],
            ["FR-02", "Sinh đặc trưng kỹ thuật",
             "Tính 20 đặc trưng cho từng mã theo thứ tự cố định, chỉ dùng dữ liệu quá khứ "
             "và hiện tại (services/feature_engineering.py)"],
            ["FR-03", "Gán nhãn theo lịch phiên chung",
             "Xác định phiên đích t+5 trên lịch thị trường, yêu cầu mã có giá đóng cửa đúng "
             "phiên đó, nếu thiếu thì loại dòng"],
            ["FR-04", "Chia TRAIN/VALIDATION/TEST theo thời gian",
             "Ba mốc rolling neo vào phiên mới nhất có nhãn hợp lệ, purge dòng có nhãn cắt "
             "qua ranh giới (services/protocol_dates.py)"],
            ["FR-05", "Tinh chỉnh siêu tham số trong Tuning Lab",
             "Người dùng nhập cấu hình cho từng mô hình tại /tuning, hệ thống chạy CV 4 fold "
             "purge và lưu kết quả kèm fingerprint dữ liệu"],
            ["FR-06", "Chọn ngưỡng quyết định từ OOF",
             "Quét ngưỡng 0,05 đến 0,95 bước 0,01, chọn ngưỡng tối đa F1_UP với ràng buộc tỷ "
             "lệ dự báo UP không quá 50% và precision không thấp hơn tỷ lệ UP thực tế"],
            ["FR-07", "Chọn họ mô hình trên VALIDATION",
             "So sánh Logistic Regression, Random Forest, Gradient Boosting theo F1_UP, rồi "
             "Recall_UP, rồi độ đơn giản; TEST không tham gia xếp hạng"],
            ["FR-08", "Refit và đánh giá TEST một lần",
             "Huấn luyện lại mô hình thắng trên TRAIN+VALIDATION, đánh giá TEST đúng một lần "
             "cùng hai baseline hằng số, không refit sau khi đã xem TEST"],
            ["FR-09", "Phát hành artifact nguyên tử",
             "Ghi final_model.pkl và model_metadata.json qua tệp tạm rồi os.replace, kèm "
             "policy id, fingerprint và toàn bộ chỉ số"],
            ["FR-10", "Dự báo một hoặc hai mã",
             "Trang chủ và CLI nhận mã, tính đặc trưng phiên mới nhất, trả nhãn, Điểm UP, "
             "ngưỡng, tên mô hình và cảnh báo baseline"],
            ["FR-11", "So sánh hai mã",
             "Trang /compare trả bảng đối chiếu Điểm UP và khoảng cách tới ngưỡng"],
            ["FR-12", "Xếp hạng toàn sàn",
             "Trang /screener xếp hạng các mã theo Điểm UP, dùng kết quả suy luận có cache "
             "theo chữ ký dữ liệu và mô hình"],
            ["FR-13", "Trang đánh giá mô hình",
             "Trang /evaluation trình bày tách biệt kết quả CV, chọn trên VALIDATION và TEST "
             "một lần, kèm ma trận nhầm lẫn và độ quan trọng đặc trưng"],
            ["FR-14", "Cập nhật dữ liệu từ vnstock",
             "Nút fetch chạy tiến trình nền tải OHLCV bổ sung theo từng mã, có retry và báo "
             "tiến độ tại /tuning/fetch-status"],
            ["FR-15", "Chatbot trả lời trên dữ liệu nội bộ",
             "Trang /chat và endpoint /api/chat định tuyến câu hỏi bằng luật, gọi handler dữ "
             "liệu nội bộ để dựng context rồi gọi mô hình ngôn ngữ đúng một lượt"],
        ],
        widths=[900, 2600, 5570],
        caption="Danh sách yêu cầu chức năng",
    )

    doc.heading(3, "3.1.5. Yêu cầu phi chức năng")
    doc.table(
        ["Nhóm", "Yêu cầu", "Cách hệ thống đáp ứng"],
        [
            ["Toàn vẹn thực nghiệm",
             "Không rò rỉ dữ liệu tương lai vào huấn luyện",
             "Nhãn theo lịch phiên chung, CV theo ngày có purge, kiểm tra ranh giới bằng "
             "verify_protocol_splits và bộ kiểm thử tự động"],
            ["Khả năng tái lập",
             "Cùng dữ liệu và cấu hình cho ra cùng kết quả",
             "RANDOM_STATE = 42, fingerprint nội dung dữ liệu, lịch sử tuning ghi đủ tham số "
             "và dải ngày từng fold"],
            ["Tính nhất quán",
             "Ngưỡng và chỉ số hiển thị giống nhau ở mọi nơi",
             "Ngưỡng lưu trong artifact, giao diện và CLI đều đọc từ metadata; metadata lệch "
             "artifact thì hệ thống rơi về artifact đang nạp"],
            ["An toàn phát hành",
             "Pipeline lỗi không được phá mô hình đang phục vụ",
             "Ghi nguyên tử qua tệp tạm, pipeline lock theo tiến trình, registry chặn đánh "
             "giá lại cùng một snapshot dữ liệu"],
            ["Hiệu năng",
             "Trang xếp hạng toàn sàn phản hồi nhanh",
             "Kết quả suy luận toàn sàn được cache theo chữ ký dữ liệu và chữ ký mô hình, "
             "chỉ tính lại khi một trong hai đổi"],
            ["Minh bạch",
             "Không che giới hạn của mô hình",
             "baseline_passed và cảnh báo được ghi vào metadata rồi hiển thị trên trang dự "
             "báo và trang đánh giá; Điểm UP không được gọi là độ chính xác"],
            ["Khả năng truy cập",
             "Giao diện dùng được bằng bàn phím và trình đọc màn hình",
             "Mỗi trang có một vùng main duy nhất, một tiêu đề h1, liên kết bỏ qua điều "
             "hướng; các ràng buộc này được kiểm bằng test_ui_shell.py"],
            ["Bảo mật cấu hình",
             "Không đưa khóa API vào mã nguồn",
             "LLM_BASE_URL, LLM_API_KEY, LLM_MODEL nạp từ .env; chatbot chặn endpoint không "
             "hợp lệ và trả lỗi 503 ổn định"],
        ],
        widths=[1700, 2900, 4470],
        caption="Yêu cầu phi chức năng",
    )
