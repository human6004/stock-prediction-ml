"""Chuong 3 phan B: thiet ke ung dung (3.2)."""

from __future__ import annotations

from pathlib import Path

from docx_builder import DocxBuilder


def build_design(doc: DocxBuilder, assets: Path, diagrams: Path) -> None:
    doc.heading(2, "3.2. Thiết kế ứng dụng")

    doc.heading(3, "3.2.1. Kiến trúc ứng dụng")
    doc.paragraph(
        "Hệ thống theo kiến trúc phân lớp một tiến trình, gồm bốn lớp: lớp dữ liệu, lớp "
        "dịch vụ, lớp trình bày và lớp lưu trữ hiện vật. Lớp dữ liệu đọc tệp CSV thô ngoài "
        "repo và sinh ra ba tệp đã xử lý trong data/processed. Lớp dịch vụ trong thư mục "
        "services chứa toàn bộ logic nghiệp vụ: tiền xử lý, sinh đặc trưng, gán nhãn, chia "
        "dữ liệu, tinh chỉnh, đánh giá, dự báo và chatbot. Lớp trình bày gồm app.py với các "
        "route Flask cùng thư mục templates và static. Lớp lưu trữ hiện vật gồm models, "
        "reports và experiments."
    )
    doc.paragraph(
        "Nguyên tắc thiết kế quan trọng nhất là mọi lớp đều lấy cấu hình từ một nguồn duy "
        "nhất: config/settings.py. Đường dẫn tệp, danh sách 20 đặc trưng theo đúng thứ tự, "
        "ngưỡng UP, tầm dự báo, số fold, số phiên purge và mã giao thức đều nằm ở đó. Nhờ "
        "vậy pipeline huấn luyện, ứng dụng web và công cụ dòng lệnh không thể dùng lệch cấu "
        "hình với nhau."
    )
    architecture = assets / "architecture_overview.png"
    if architecture.exists():
        doc.image(architecture, "Kiến trúc tổng thể của hệ thống HOSE Stock Trend Prediction")
    doc.paragraph(
        "Hình trên cho thấy dữ liệu chảy một chiều từ tệp thô qua các bước xử lý tới hiện "
        "vật mô hình, sau đó ứng dụng web và dòng lệnh chỉ đọc hiện vật đã công bố. Không "
        "có đường nào cho phép giao diện huấn luyện lại mô hình trực tiếp, vì việc công bố "
        "mô hình chỉ diễn ra trong scripts/run_pipeline.py."
    )
    doc.table(
        ["Thành phần", "Tệp chính", "Trách nhiệm"],
        [
            ["Cấu hình", "config/settings.py", "Đường dẫn, đặc trưng, ranh giới, tham số giao thức"],
            ["Tiền xử lý", "services/preprocessing.py", "Kiểm tra dữ liệu thô, làm sạch, lọc mã đủ điều kiện"],
            ["Đặc trưng và nhãn", "services/feature_engineering.py", "20 đặc trưng, nhãn t+5, chia TRAIN/VALIDATION/TEST"],
            ["Mốc thời gian", "services/protocol_dates.py", "Suy ra ba mốc rolling từ phiên mới nhất có nhãn"],
            ["Chia CV", "services/time_splitting.py", "Sinh fold theo ngày, gap 5 phiên, purge nhãn"],
            ["Tinh chỉnh", "services/model_tuning.py", "Dựng estimator, chạy CV, chọn ngưỡng OOF, fit TRAIN"],
            ["Tuning Lab", "services/tuning_lab.py", "Kiểm tra tham số người dùng nhập, chạy nền, ghi lịch sử"],
            ["Trạng thái thí nghiệm", "services/experiment_state.py", "Fingerprint, lịch sử, khóa pipeline, sổ đăng ký TEST"],
            ["Đánh giá", "services/model_evaluation.py", "Chấm VALIDATION, baseline, refit, TEST một lần, xuất báo cáo"],
            ["Dự báo", "services/prediction_service.py", "Nạp hiện vật, sinh đặc trưng mới nhất, suy luận"],
            ["Chatbot", "services/chatbot_service.py, chatbot_tools.py", "Gọi LLM một lần chọn action, validate quyết định, dispatcher và formatter cố định"],
            ["Giao diện", "app.py, templates/, static/", "Route Flask, HTML, CSS, biểu đồ Chart.js"],
        ],
        widths=[1900, 3100, 4070],
        caption="Các thành phần chính và trách nhiệm tương ứng",
    )

    doc.heading(3, "3.2.2. Thiết kế dữ liệu")
    doc.paragraph(
        "Dữ liệu đi qua bốn dạng: OHLCV thô, OHLCV đã làm sạch, bảng đặc trưng và tập dữ "
        "liệu học máy có nhãn. Mỗi dạng được ghi ra một tệp CSV riêng để có thể kiểm tra lại "
        "từng bước."
    )
    doc.table(
        ["Tệp", "Vai trò", "Khóa chính"],
        [
            ["hose_stock_raw.csv (ngoài repo)", "Dữ liệu OHLCV thô, 554.897 dòng, 400 mã", "symbol + trading_date"],
            ["data/processed/hose_stock_clean.csv", "OHLCV đã làm sạch, 554.664 dòng, 400 mã", "symbol + trading_date"],
            ["data/processed/hose_stock_features.csv", "20 đặc trưng, 516.940 dòng, 396 mã đủ điều kiện", "symbol + trading_date"],
            ["data/processed/ml_dataset.csv", "Đặc trưng kèm nhãn, 510.862 dòng", "symbol + trading_date"],
            ["models/final_model.pkl", "Đúng hiện vật vừa được đánh giá TEST", "-"],
            ["models/model_metadata.json", "Giao thức, fingerprint, thứ tự đặc trưng, chỉ số", "-"],
        ],
        widths=[3300, 3900, 1870],
        caption="Các tệp dữ liệu và hiện vật chính của hệ thống",
    )
    doc.paragraph(
        "Bảng dưới đây là lược đồ của ml_dataset.csv, tệp trung tâm của toàn bộ thí nghiệm. "
        "Hai cột trading_date và label_end_date là cơ sở để chia dữ liệu và purge nhãn, nên "
        "chúng được giữ trong tập dữ liệu nhưng bị loại khỏi danh sách đặc trưng."
    )
    doc.table(
        ["Nhóm cột", "Cột", "Ý nghĩa"],
        [
            ["Định danh", "symbol, trading_date", "Mã cổ phiếu và ngày giao dịch tham chiếu"],
            ["Giá gốc", "open, high, low, close, volume", "OHLCV đã làm sạch của phiên"],
            ["Đặc trưng", "20 cột theo FEATURE_COLUMNS", "Đầu vào duy nhất của mô hình"],
            ["Nhãn", "future_close_5d, future_return_5d", "Giá và lợi suất tại đúng phiên t+5"],
            ["Nhãn", "label_end_date", "Ngày kết thúc nhãn, dùng để purge"],
            ["Nhãn", "target, target_label", "1/0 và UP/NOT_UP theo ngưỡng 1%"],
        ],
        widths=[1800, 3400, 3870],
        caption="Lược đồ tập dữ liệu học máy ml_dataset.csv",
    )
    doc.paragraph(
        "Quy tắc chống rò rỉ được cài đặt thành kiểm tra tự động trong "
        "verify_protocol_splits: bốn cột future_close_5d, future_return_5d, target và "
        "label_end_date không được xuất hiện trong FEATURE_COLUMNS; nếu có, hàm ném lỗi và "
        "pipeline dừng."
    )
    doc.paragraph(
        "Ba mốc thời gian không ghi cứng mà được suy ra động theo phiên mới nhất có nhãn "
        "hợp lệ (rolling walk-forward) tại services/protocol_dates.py: TEST_END neo vào phiên "
        "đó, VALIDATION_END lùi lại 94 ngày lịch và TRAIN_END lùi thêm 274 ngày lịch. Nhờ vậy "
        "mỗi lần dữ liệu được làm mới, cửa sổ thí nghiệm trượt về phía trước thay vì đóng "
        "cứng ở một mốc cũ. Bảng dưới đây là ba tập thực tế của lần chạy đang phục vụ, ghi "
        "trong reports/split_summary.csv."
    )
    doc.table(
        ["Tập", "Điều kiện chia", "Khoảng ngày giao dịch", "Số dòng", "Tỷ lệ UP"],
        [
            ["TRAIN", "label_end_date <= 10/07/2025", "23/10/2019 - 03/07/2025", "419.807", "38,9%"],
            ["VALIDATION", "trading_date > 10/07/2025 và label_end_date <= 10/04/2026",
             "11/07/2025 - 03/04/2026", "67.047", "33,2%"],
            ["TEST", "10/04/2026 < trading_date <= 13/07/2026", "13/04/2026 - 13/07/2026",
             "20.350", "23,8%"],
            ["PURGED", "Nhãn vắt qua ranh giới nên bị loại", "Hai ranh giới",
             "1.883 + 1.775", "-"],
        ],
        widths=[1300, 3100, 2300, 1200, 970],
        caption="Ba tập dữ liệu thực tế và các dòng bị purge",
    )
    doc.paragraph(
        "Tỷ lệ UP giảm dần từ TRAIN sang TEST (38,9% xuống 23,8%) cho thấy giai đoạn TEST là "
        "một chế độ thị trường khó hơn: số cơ hội tăng hơn 1% trong năm phiên ít hơn hẳn so "
        "với giai đoạn huấn luyện. Đây là thông tin cần thiết để đọc đúng kết quả ở mục 3.3."
    )
    dataflow = assets / "dataflow_pipeline.png"
    if dataflow.exists():
        doc.image(dataflow, "Luồng dữ liệu từ tệp thô đến hiện vật mô hình và suy luận")
    doc.paragraph(
        "Hình trên mô tả bốn dạng dữ liệu nối tiếp nhau: tệp OHLCV thô được làm sạch thành "
        "hose_stock_clean.csv, sinh đặc trưng thành hose_stock_features.csv, gắn nhãn t+5 "
        "thành ml_dataset.csv, rồi chia thành ba tập theo mốc thời gian trước khi vào bước "
        "tinh chỉnh và đánh giá. Hai hiện vật cuối là final_model.pkl và model_metadata.json; "
        "ứng dụng web và dòng lệnh chỉ đọc hai hiện vật này, không chạm vào bất kỳ bước xử lý "
        "nào ở phía trước."
    )

    doc.heading(3, "3.2.3. Thiết kế chi tiết")
    doc.paragraph(
        "Mục này trình bày bảy thuật toán và luồng xử lý cốt lõi. Mỗi mục con kèm một lưu đồ "
        "hoặc lược đồ tuần tự, trong đó nhãn của các khối được lấy đúng theo tên hàm, tên tệp "
        "và tham số mô tả trong phần văn bản tương ứng."
    )

    doc.heading(4, "3.2.3.1. Thuật toán gán nhãn theo lịch phiên chung")
    doc.paragraph(
        "Nhãn được sinh trong hàm create_labels. Điểm cốt lõi là lịch phiên được lấy từ tập "
        "hợp toàn bộ trading_date của thị trường, không phải từ chuỗi ngày của riêng một mã. "
        "Nhờ vậy mọi mã có cùng ngày tham chiếu đều có cùng ngày kết thúc nhãn."
    )
    doc.code_block(
        "market_dates = sorted(prices['trading_date'].unique())\n"
        "future_by_date = dict(zip(market_dates, market_dates.shift(-5)))\n"
        "labels['label_end_date'] = labels['trading_date'].map(future_by_date)\n"
        "labels = labels.merge(future_prices, on=['symbol', 'label_end_date'], how='left')\n"
        "labels['future_return_5d'] = labels['future_close_5d'] / labels['close_at_label_start'] - 1\n"
        "labels['target'] = (labels['future_return_5d'] > 0.01).astype('int8')"
    )
    doc.paragraph(
        "Nếu một mã không có giá tại đúng phiên đích, chẳng hạn do tạm ngừng giao dịch, dòng "
        "đó bị loại thay vì nhảy sang phiên kế tiếp. Trong lần chạy hiện tại có 1.768 dòng "
        "không còn đủ phiên tương lai trên lịch thị trường và 4.468 dòng thiếu giá của chính "
        "mã tại phiên đích, tất cả đều bị loại. Tập dữ liệu cuối cùng còn 510.862 dòng với "
        "tỷ lệ UP là 37,61%."
    )
    flow_labeling = assets / "flow_labeling.png"
    if flow_labeling.exists():
        doc.image(flow_labeling, "Lưu đồ thuật toán gán nhãn theo lịch phiên chung của thị trường")
    doc.paragraph(
        "Lưu đồ trên đọc từ trên xuống theo đúng thứ tự các bước trong create_labels. Hai "
        "nhánh rẽ phải là hai trường hợp bị loại dòng: không còn đủ 5 phiên tương lai trên "
        "lịch thị trường, và mã không có giá tại đúng phiên đích. Chỉ những dòng vượt qua cả "
        "hai nhánh mới được tính lợi suất và gán nhãn theo ngưỡng 1%, nên tập dữ liệu cuối "
        "cùng luôn nhỏ hơn bảng đặc trưng đầu vào."
    )

    doc.heading(4, "3.2.3.2. Kiểm định chéo theo ngày, có gap và purge")
    doc.paragraph(
        "Hàm iter_purged_date_splits trong services/time_splitting.py chia dữ liệu theo danh "
        "sách ngày giao dịch duy nhất chứ không theo chỉ số dòng. TimeSeriesSplit của "
        "scikit-learn được áp lên mảng ngày với gap bằng 5, sau đó dòng dữ liệu được lọc lại "
        "theo hai điều kiện."
    )
    doc.code_block(
        "market_dates = np.sort(dates.unique())\n"
        "market_dates = market_dates[market_dates >= pd.Timestamp('2021-01-01')]\n"
        "splitter = TimeSeriesSplit(n_splits=4, gap=5)\n"
        "for train_date_idx, validation_date_idx in splitter.split(market_dates):\n"
        "    validation_start = market_dates[validation_date_idx][0]\n"
        "    train_mask = dates.isin(train_dates) & (label_ends < validation_start)\n"
        "    validation_mask = dates.isin(validation_dates)"
    )
    doc.paragraph(
        "Điều kiện dates.isin(train_dates) bảo đảm một ngày giao dịch không bị chia sang cả "
        "hai phía. Điều kiện label_ends < validation_start là bước purge: mọi dòng huấn luyện "
        "có nhãn kết thúc sau khi tập kiểm định bắt đầu đều bị loại. Vì gap được đếm trên "
        "mảng ngày, khoảng cách là đúng 5 phiên thị trường chứ không phải 5 dòng dữ liệu gộp."
    )
    flow_cv = assets / "flow_purged_cv.png"
    if flow_cv.exists():
        doc.image(flow_cv, "Lưu đồ chia kiểm định chéo theo ngày giao dịch, có gap và purge")
    doc.paragraph(
        "Trong lưu đồ, vòng lặp chạy qua bốn fold do TimeSeriesSplit sinh ra trên mảng ngày. "
        "Mỗi fold lấy ngày đầu tiên của phần kiểm định làm mốc validation_start, rồi áp hai bộ "
        "lọc liên tiếp lên dòng dữ liệu: lọc theo tập ngày để một phiên không nằm cả hai phía, "
        "và lọc purge để bỏ những dòng huấn luyện có nhãn kết thúc vượt qua mốc đó. Khối cuối "
        "cùng cho thấy cặp train_mask và validation_mask mới là dữ liệu thực sự dùng để fit và "
        "chấm điểm."
    )

    doc.heading(4, "3.2.3.3. Chọn ngưỡng quyết định từ xác suất OOF")
    doc.paragraph(
        "Do lớp UP là lớp thiểu số, ngưỡng 0,5 mặc định không phù hợp. Hàm "
        "select_oof_threshold gom xác suất out-of-fold của cả bốn fold rồi quét ngưỡng từ "
        "0,05 đến 0,95 theo bước 0,01. Một ngưỡng chỉ được xét nếu thỏa hai ràng buộc: tỷ lệ "
        "dòng được dự báo UP không vượt 50% và precision của lớp UP không thấp hơn tỷ lệ UP "
        "thực tế. Hai ràng buộc này ngăn mô hình thoái hóa thành baseline luôn dự báo UP để "
        "ăn điểm F1."
    )
    doc.paragraph(
        "Trong ba ứng viên hiện tại, cả ba đều chọn ngưỡng 0,49. Tỷ lệ dự báo UP trên OOF lần "
        "lượt là 0,4998 với Logistic Regression, 0,4997 với Random Forest và 0,4771 với "
        "Gradient Boosting, đều nằm dưới trần 0,50."
    )
    flow_threshold = assets / "flow_threshold.png"
    if flow_threshold.exists():
        doc.image(flow_threshold, "Lưu đồ chọn ngưỡng quyết định từ xác suất out-of-fold")
    doc.paragraph(
        "Lưu đồ mô tả vòng quét ngưỡng: mỗi giá trị thử lần lượt đi qua hai điều kiện chặn. "
        "Ngưỡng làm tỷ lệ dự báo UP vượt 50%, hoặc làm precision của lớp UP tụt xuống dưới tỷ "
        "lệ UP thực tế, bị loại ngay và vòng lặp nhảy sang ngưỡng kế tiếp. Chỉ những ngưỡng "
        "vượt cả hai điều kiện mới được tính F1_UP và đưa vào so sánh; ngưỡng có F1_UP cao nhất "
        "trong số đó được ghi vào hiện vật mô hình."
    )

    doc.heading(4, "3.2.3.4. Quy trình chọn Final Model")
    doc.paragraph(
        "Việc chọn mô hình cuối cùng tuân thủ ba giai đoạn tách biệt để tập TEST không tham "
        "gia vào bất kỳ quyết định nào."
    )
    doc.bullets(
        [
            "Giai đoạn TRAIN: ba cấu hình đã chốt trong Tuning Lab được fit trên toàn bộ TRAIN, tạo ba ứng viên.",
            "Giai đoạn VALIDATION: ba ứng viên được chấm trên VALIDATION; hàm select_final_model xếp hạng theo F1_UP, rồi Recall_UP, rồi độ đơn giản của mô hình.",
            "Giai đoạn TEST: mô hình thắng được fit lại trên TRAIN cộng VALIDATION rồi đánh giá TEST đúng một lần; đúng đối tượng vừa đánh giá được lưu thành final_model.pkl, không refit thêm.",
        ],
        numbered=True,
    )
    selection = assets / "final_model_selection.png"
    if selection.exists():
        doc.image(selection, "Quy trình chọn Final Model qua ba giai đoạn TRAIN, VALIDATION và TEST")
    doc.paragraph(
        "Hàm verify_model_selection kiểm tra sau cùng: đúng một mô hình được đánh dấu selected, "
        "mô hình đó phải thuộc ba họ được phép và tên trong hiện vật phải trùng với dòng đã "
        "chọn trong bảng so sánh. Việc công bố dùng ghi tệp tạm rồi os.replace, nên nếu bước "
        "nào lỗi thì mô hình đang phục vụ vẫn nguyên vẹn."
    )

    doc.heading(4, "3.2.3.5. Tuning Lab và bảo toàn tính hợp lệ của thí nghiệm")
    doc.paragraph(
        "Tuning Lab tại /tuning cho phép người dùng nhập trực tiếp siêu tham số cho từng mô "
        "hình. Giá trị nhập được kiểm tra theo TUNABLE_PARAM_SCHEMA, ví dụ C phải là số thực "
        "dương, solver chỉ nhận lbfgs hoặc liblinear, subsample phải nằm trong khoảng (0, 1]. "
        "Mỗi lần chạy được thực hiện trên một luồng nền, chỉ cho phép một tác vụ tại một thời "
        "điểm, và kết quả được ghi vào experiments/tuning_history.csv kèm mã giao thức, "
        "fingerprint dữ liệu và tham số dạng JSON."
    )
    doc.paragraph(
        "Fingerprint là cơ chế chống lệch dữ liệu. Lịch sử tinh chỉnh dùng fingerprint của "
        "riêng TRAIN, còn khóa TEST dùng fingerprint của toàn bộ ảnh chụp TRAIN cộng "
        "VALIDATION cộng TEST. Nhờ tách hai fingerprint, việc làm mới dữ liệu để suy luận "
        "không mở lại tập TEST. Khi chạy pipeline, hàm is_config_complete kiểm tra cả ba mô "
        "hình đã có cấu hình, đúng schema, đúng giao thức và đúng fingerprint hiện tại; thiếu "
        "một điều kiện là pipeline dừng ngay."
    )
    flow_tuning = assets / "flow_tuning_lab.png"
    if flow_tuning.exists():
        doc.image(flow_tuning, "Lưu đồ một lượt tinh chỉnh trong Tuning Lab và điều kiện cho phép công bố mô hình")
    doc.paragraph(
        "Lưu đồ trên cho thấy ba chốt chặn đặt trước khi tốn chi phí tính toán: kiểm tra kiểu "
        "và miền giá trị theo TUNABLE_PARAM_SCHEMA, kiểm tra đã có tác vụ nền nào đang chạy, "
        "và kiểm tra is_config_complete trước khi pipeline được phép công bố. Ba nhánh rẽ phải "
        "là ba trường hợp dừng sớm; chỉ khi cả ba chốt đều đạt thì scripts/run_pipeline.py mới "
        "được mở khóa tập TEST và ghi hiện vật mới."
    )

    doc.heading(4, "3.2.3.6. Luồng suy luận trên giao diện và dòng lệnh")
    doc.paragraph(
        "Hàm predict_symbols nhận một đến năm mã, đọc dữ liệu đã làm sạch, sinh đặc trưng cho "
        "riêng các mã đó, lấy dòng mới nhất của từng mã rồi gọi mô hình. Thứ tự cột đầu vào "
        "được lấy từ feature_order trong metadata, không lấy theo thứ tự cột của DataFrame, "
        "nên không thể xảy ra lệch cột giữa lúc huấn luyện và lúc suy luận."
    )
    doc.code_block(
        "probability_up = [float(row[classes.index(1)]) for row in model.predict_proba(x_latest)]\n"
        "labels = ['UP' if p >= decision_threshold else 'NOT_UP' for p in probability_up]"
    )
    doc.paragraph(
        "Ngưỡng quyết định được đọc từ hiện vật thay vì ghi cứng, nên khi pipeline công bố mô "
        "hình mới với ngưỡng khác thì giao diện và dòng lệnh tự đồng bộ. Kết quả trả về gồm "
        "nhãn UP hoặc NOT_UP, Điểm UP, ngày tham chiếu, danh sách phiên dự kiến và cảnh báo "
        "baseline nếu có. Trang xếp hạng dùng predict_all_symbols với cache theo chữ ký dữ "
        "liệu và chữ ký mô hình, nên chỉ tính lại toàn sàn khi một trong hai thay đổi."
    )
    sequence_predict = assets / "sequence_predict.png"
    if sequence_predict.exists():
        doc.image(sequence_predict, "Lược đồ tuần tự một lượt dự báo từ giao diện đến hiện vật mô hình")
    doc.paragraph(
        "Lược đồ tuần tự trên cho thấy giao diện không giữ bất kỳ tri thức nào về mô hình. "
        "Trình duyệt chỉ gửi mã cần dự báo, tầng dịch vụ đọc dữ liệu đã làm sạch, sinh đặc "
        "trưng, lấy thứ tự cột từ metadata rồi gọi predict_proba; ngưỡng quyết định cũng đến "
        "từ metadata. Nhờ vậy khi hiện vật được thay, không dòng mã giao diện nào phải sửa."
    )

    doc.heading(4, "3.2.3.7. Chatbot chỉ trả lời trên dữ liệu nội bộ")
    doc.paragraph(
        "Chatbot dùng kiến trúc action decision: mô hình ngôn ngữ lớn được gọi đúng một lần "
        "mỗi lượt và chỉ trả về một JSON thuần gồm hai khóa action và arguments, trong đó "
        "action là một trong năm giá trị cố định GENERAL_CHAT, STOCK_SIGNAL, STOCK_RANKING, "
        "PROJECT_INFO và OUT_OF_SCOPE. Máy chủ kiểm tra chặt schema của quyết định (đúng hai "
        "khóa, action hợp lệ, arguments đúng kiểu và miền giá trị) rồi mới gọi handler dữ liệu "
        "tương ứng qua một dispatcher cố định; không gửi raw CSV, mã nguồn hay pickle tới nhà "
        "cung cấp LLM."
    )
    doc.paragraph(
        "Request không gửi tools, tool_choice hoặc response_format; không retry và không có "
        "lần gọi thứ hai. Câu hỏi tối đa 1.000 ký tự, lịch sử gửi kèm tối đa 6 tin nhắn, thời "
        "hạn toàn lượt 60 giây. Mọi câu chữ tới người dùng do máy chủ soạn: câu xã giao và câu "
        "từ chối là chuỗi cố định theo kind hoặc reason, câu trả lời dữ liệu do formatter ghép "
        "từ kết quả suy luận thật kèm khuyến cáo dữ liệu offline; yêu cầu khuyên mua bán được "
        "định tuyến sang OUT_OF_SCOPE và nhận câu từ chối cố định. Khung chat nổi và trang "
        "chat dùng chung sessionStorage."
    )
    sequence_chatbot = assets / "sequence_chatbot.png"
    if sequence_chatbot.exists():
        doc.image(sequence_chatbot, "Lược đồ tuần tự một lượt hỏi đáp của chatbot theo kiến trúc action decision")
    doc.paragraph(
        "Điểm đáng lưu ý trong lược đồ là ranh giới tin cậy: nhà cung cấp LLM chỉ nhận câu hỏi "
        "cùng lịch sử ngắn và chỉ trả về quyết định action; toàn bộ số liệu trong câu trả lời "
        "do máy chủ đọc từ artifact và report đã công bố, sau khi quyết định đã qua bước kiểm "
        "tính hợp lệ. Nhà cung cấp LLM không có đường nào chạm tới tệp CSV, mã nguồn hay tệp "
        "pickle của mô hình."
    )
