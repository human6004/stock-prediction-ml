"""Chuong 4 (Ket luan), Tai lieu tham khao va Phu luc."""

from __future__ import annotations

from pathlib import Path

from docx_builder import DocxBuilder


def build_chapter4(doc: DocxBuilder) -> None:
    doc.heading(1, "CHƯƠNG 4: KẾT LUẬN")

    doc.heading(2, "4.1. Tổng kết kết quả đạt được")
    doc.paragraph(
        "Niên luận đã hoàn thành một hệ thống dự báo xu hướng ngắn hạn cổ phiếu HOSE "
        "chạy được đầu cuối: từ tệp OHLCV thô của 400 mã, qua làm sạch, sinh 20 đặc trưng "
        "kỹ thuật, gán nhãn UP/NOT_UP theo đúng phiên thị trường thứ năm, chia dữ liệu theo "
        "thời gian, tinh chỉnh siêu tham số, chọn mô hình và công bố hiện vật dùng cho dự "
        "báo trên web và dòng lệnh."
    )
    doc.paragraph(
        "Đối chiếu với năm mục tiêu đặt ra ở mục 1.2, kết quả cụ thể như sau."
    )
    doc.table(
        ["Mục tiêu", "Mức độ hoàn thành", "Bằng chứng trong sản phẩm"],
        [
            [
                "Xây dựng pipeline dữ liệu và đặc trưng đầy đủ",
                "Đạt",
                "510.862 dòng có nhãn hợp lệ với 20 đặc trưng; các tệp "
                "hose_stock_clean.csv, hose_stock_features.csv, ml_dataset.csv",
            ],
            [
                "Định nghĩa nhãn không rò rỉ theo lịch phiên chung",
                "Đạt",
                "services/feature_engineering.py: create_labels; kiểm thử "
                "test_target_uses_exact_fifth_common_market_session",
            ],
            [
                "So sánh ba mô hình theo giao thức có kỷ luật",
                "Đạt",
                "reports/tuning_results.csv, reports/model_comparison.csv, "
                "reports/final_model_evaluation.csv",
            ],
            [
                "Công bố hiện vật và giao diện sử dụng",
                "Đạt",
                "models/final_model.pkl, models/model_metadata.json, sáu trang Flask "
                "và chatbot dữ liệu nội bộ",
            ],
            [
                "Đạt F1_UP trên TEST cao hơn baseline luôn dự báo UP",
                "Không đạt",
                "TEST F1_UP 0,3754 so với baseline 0,3839; hệ thống ghi "
                "baseline_passed = false và hiển thị cảnh báo",
            ],
        ],
        widths=[2600, 1500, 4970],
        caption="Đối chiếu mục tiêu và kết quả thực tế",
    )
    doc.paragraph(
        "Bốn mục tiêu về kỹ thuật và sản phẩm đã đạt. Mục tiêu về chất lượng dự báo không "
        "đạt và điều này được báo cáo công khai thay vì che đi. Đây là kết quả có ý nghĩa "
        "phương pháp: khi loại bỏ ba nguồn rò rỉ thường gặp, năng lực dự báo thực tế của "
        "một tập đặc trưng kỹ thuật thuần với dữ liệu ngày là rất hạn chế. Nhiều báo cáo "
        "dự báo chứng khoán cho chỉ số cao hơn chủ yếu vì tập kiểm tra bị nhiễm thông tin "
        "tương lai hoặc vì baseline hằng số không được đưa vào so sánh."
    )

    doc.heading(2, "4.2. Sản phẩm giao nộp")
    doc.bullets(
        [
            "Mã nguồn hệ thống gồm config, services, scripts, templates, static và app.py.",
            "Bộ dữ liệu đã xử lý: hose_stock_clean.csv, hose_stock_features.csv, ml_dataset.csv.",
            "Hiện vật mô hình: models/final_model.pkl và models/model_metadata.json.",
            "Mười bốn tệp báo cáo trong thư mục reports, gồm tuning_results.csv, "
            "cv_fold_results.csv, model_comparison.csv, final_model_evaluation.csv, "
            "classification_report.csv, confusion_matrix.csv/png, feature_importance.csv, "
            "split_summary.csv, pipeline_summary.json và model_selection_report.txt.",
            "Sổ ghi thực nghiệm: experiments/tuning_history.csv, manual_config.json, "
            "evaluation_registry.json và các bản lưu trữ theo mốc phát hành.",
            "Bộ kiểm thử tự động trong thư mục tests, gồm các nhóm dữ liệu, mô hình, chatbot và giao diện.",
            "Tài liệu kỹ thuật: README.md, docs/GIAI_THICH_PROJECT.md, "
            "docs/SO_DO_KIEN_TRUC_HE_THONG.md và các sơ đồ trong docs/diagrams.",
        ]
    )

    doc.heading(2, "4.3. Hạn chế")
    doc.bullets(
        [
            "Chất lượng dự báo chưa vượt baseline hằng số trên TEST; hệ thống chỉ nên dùng "
            "để học tập và nghiên cứu, không dùng cho quyết định đầu tư.",
            "Đặc trưng chỉ khai thác giá và khối lượng theo ngày; chưa có dữ liệu cơ bản, "
            "dòng tiền khối ngoại, chỉ số ngành hay tin tức.",
            "Ngưỡng quyết định được chọn từ xác suất OOF trên TRAIN theo một tiêu chí duy "
            "nhất và giữ cố định; chưa hiệu chỉnh lại theo chế độ thị trường.",
            "Suy luận dùng dữ liệu tĩnh đến phiên gần nhất trong tệp, không phải giá trực tuyến.",
            "Ứng dụng web không có xác thực người dùng và chỉ chạy một tiến trình; khóa "
            "tuning là khóa trong tiến trình nên chưa an toàn khi triển khai nhiều worker.",
            "Chưa có kiểm thử hiệu năng và kiểm thử chấp nhận có người dùng thực tham gia.",
        ]
    )

    doc.heading(2, "4.4. Hướng phát triển")
    doc.bullets(
        [
            "Mở rộng đặc trưng: chỉ số thị trường VN-Index, sức mạnh tương đối theo ngành, "
            "dữ liệu giao dịch khối ngoại và các chỉ báo cơ bản theo quý.",
            "Thử nhóm mô hình mạnh hơn cho dữ liệu bảng như XGBoost, LightGBM hoặc mô hình "
            "chuỗi thời gian LSTM, đồng thời giữ nguyên giao thức chống rò rỉ hiện có.",
            "Thay việc chọn một ngưỡng cố định bằng hiệu chỉnh xác suất, ví dụ Platt "
            "scaling hoặc isotonic regression, rồi chọn ngưỡng theo mục tiêu kinh doanh.",
            "Đánh giá theo hướng đầu tư thay vì chỉ theo độ đo phân loại: mô phỏng chiến "
            "lược có chi phí giao dịch, đo lợi nhuận tích lũy và mức sụt giảm tối đa.",
            "Tự động hóa walk-forward định kỳ: mỗi tháng dịch cửa sổ, chạy lại pipeline và "
            "theo dõi độ trôi của chất lượng dự báo qua thời gian.",
            "Bổ sung xác thực người dùng, khóa liên tiến trình và ghi vết truy cập trước khi "
            "triển khai hệ thống ra mạng.",
        ]
    )


REFERENCES = [
    "F. Pedregosa et al., \u201cScikit-learn: Machine learning in Python,\u201d Journal of "
    "Machine Learning Research, vol. 12, pp. 2825\u20132830, 2011.",
    "L. Breiman, \u201cRandom forests,\u201d Machine Learning, vol. 45, no. 1, pp. 5\u201332, 2001.",
    "J. H. Friedman, \u201cGreedy function approximation: A gradient boosting machine,\u201d "
    "The Annals of Statistics, vol. 29, no. 5, pp. 1189\u20131232, 2001.",
    "T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning: "
    "Data Mining, Inference, and Prediction, 2nd ed. New York, NY, USA: Springer, 2009.",
    "M. L. de Prado, Advances in Financial Machine Learning. Hoboken, NJ, USA: Wiley, 2018.",
    "C. Bergmeir and J. M. Ben\u00edtez, \u201cOn the use of cross-validation for time series "
    "predictor evaluation,\u201d Information Sciences, vol. 191, pp. 192\u2013213, 2012.",
    "E. F. Fama, \u201cEfficient capital markets: A review of theory and empirical work,\u201d "
    "The Journal of Finance, vol. 25, no. 2, pp. 383\u2013417, 1970.",
    "J. J. Murphy, Technical Analysis of the Financial Markets. New York, NY, USA: "
    "New York Institute of Finance, 1999.",
    "J. W. Wilder, New Concepts in Technical Trading Systems. Greensboro, NC, USA: "
    "Trend Research, 1978.",
    "W. McKinney, \u201cData structures for statistical computing in Python,\u201d in Proc. 9th "
    "Python in Science Conf. (SciPy), Austin, TX, USA, 2010, pp. 56\u201361.",
    "C. R. Harris et al., \u201cArray programming with NumPy,\u201d Nature, vol. 585, "
    "pp. 357\u2013362, 2020.",
    "Pallets Projects, \u201cFlask documentation (3.1.x).\u201d [Online]. Available: "
    "https://flask.palletsprojects.com. [Accessed: Jul. 25, 2026].",
    "scikit-learn developers, \u201cTimeSeriesSplit \u2014 scikit-learn 1.8 documentation.\u201d "
    "[Online]. Available: https://scikit-learn.org/stable/modules/generated/"
    "sklearn.model_selection.TimeSeriesSplit.html. [Accessed: Jul. 25, 2026].",
    "Vnstock developers, \u201cVnstock: Python library for Vietnamese stock market data.\u201d "
    "[Online]. Available: https://vnstocks.com. [Accessed: Jul. 25, 2026].",
    "Ho Chi Minh City Stock Exchange, \u201cTrading rules and market information.\u201d "
    "[Online]. Available: https://www.hsx.vn. [Accessed: Jul. 25, 2026].",
]


def build_references(doc: DocxBuilder) -> None:
    doc.heading(1, "TÀI LIỆU THAM KHẢO")
    doc.paragraph(
        "Danh mục tài liệu được trình bày theo chuẩn IEEE. Số trong ngoặc vuông tại phần "
        "nội dung tương ứng với số thứ tự trong danh mục này."
    )
    for index, entry in enumerate(REFERENCES, start=1):
        doc.paragraph(f"[{index}] {entry}", spacing_after=80)


def build_appendices(doc: DocxBuilder, assets: Path, diagrams: Path) -> None:
    doc.heading(1, "PHỤ LỤC")

    doc.heading(2, "Phụ lục A. Hướng dẫn cài đặt và sử dụng")
    doc.heading(3, "A.1. Cài đặt môi trường")
    doc.code_block(
        "python -m venv .venv\n"
        ".\\.venv\\Scripts\\Activate.ps1\n"
        "pip install -r requirements.txt"
    )
    doc.paragraph(
        "Tệp .env cần hai biến LLM_BASE_URL và LLM_API_KEY nếu muốn dùng chatbot; các chức "
        "năng dự báo, so sánh, xếp hạng và đánh giá không cần cấu hình này."
    )

    doc.heading(3, "A.2. Chạy pipeline huấn luyện")
    doc.code_block(
        "python scripts/fetch_hose_data.py      # cap nhat OHLCV tu vnstock (tuy chon)\n"
        "python scripts/preprocess_data.py      # lam sach du lieu\n"
        "python scripts/build_features.py       # sinh 20 dac trung va nhan\n"
        "python scripts/run_pipeline.py         # pipeline chinh thuc, cong bo Final Model"
    )
    doc.paragraph(
        "Trước khi chạy run_pipeline.py phải chốt đủ cấu hình cho ba mô hình tại trang "
        "Tuning Lab, vì pipeline từ chối chạy nếu manual_config.json thiếu mô hình, sai "
        "policy hoặc sai fingerprint dữ liệu."
    )

    doc.heading(3, "A.3. Dự báo")
    doc.code_block(
        "python scripts/predict_stock.py --symbol FPT\n"
        "python scripts/predict_stock.py --symbol SSI\n"
        "python app.py                          # mo http://127.0.0.1:5000"
    )
    doc.paragraph(
        "Trang chủ nhận một mã, trang /compare so sánh đúng hai mã; cả hai trả về nhãn UP "
        "hoặc NOT_UP kèm Điểm UP. Trang /evaluation trình bày ba khối kết quả tách biệt: "
        "kiểm định chéo trên TRAIN, chọn họ mô hình trên VALIDATION và đánh giá một lần "
        "trên TEST."
    )

    doc.heading(3, "A.4. Chạy kiểm thử")
    doc.code_block("python -m pytest tests -q")

    doc.heading(2, "Phụ lục B. Sơ đồ bổ sung")
    doc.paragraph(
        "Phụ lục này tập hợp bốn sơ đồ chi tiết được nhắc tới trong Chương 3 nhưng không đặt "
        "trong phần thân để giữ mạch trình bày. Mỗi sơ đồ kèm một đoạn diễn giải ngắn ngay bên "
        "dưới."
    )
    for name, caption, note in [
        ("dataflow_pipeline.png", "Sơ đồ luồng dữ liệu chi tiết của pipeline",
         "Sơ đồ mở rộng luồng dữ liệu ở mục 3.2.2, chỉ rõ tên tệp đầu vào và đầu ra của từng "
         "bước cùng số dòng còn lại sau mỗi lần lọc, nên có thể đối chiếu trực tiếp với các "
         "con số trong bảng hiện vật dữ liệu."),
        ("tuning_logistic_regression.png",
         "Lưu đồ tinh chỉnh Logistic Regression với hai siêu tham số C và solver",
         "Lưu đồ cho thấy không gian tham số của Logistic Regression chỉ gồm hai trục là C và "
         "solver, nên một lượt tinh chỉnh kết thúc nhanh nhất trong ba họ mô hình."),
        ("tuning_random_forest.png",
         "Lưu đồ tinh chỉnh Random Forest với bốn nhóm siêu tham số",
         "Random Forest có bốn nhóm tham số điều khiển số cây, độ sâu, số mẫu tối thiểu ở lá "
         "và tỷ lệ đặc trưng mỗi lần chia, tương ứng bốn khối nhập liệu trên lưu đồ; đây là "
         "cấu hình sinh ra Final Model được trình bày ở mục 3.3."),
        ("tuning_gradient_boosting.png",
         "Lưu đồ tinh chỉnh Gradient Boosting với bốn nhóm siêu tham số",
         "Gradient Boosting thêm hai tham số đặc thù là learning_rate và subsample; do các cây "
         "được huấn luyện tuần tự, khối chạy kiểm định chéo bốn fold trên lưu đồ này là bước "
         "tốn thời gian nhất trong toàn bộ Tuning Lab."),
    ]:
        path = assets / name
        if path.exists():
            doc.image(path, caption)
            doc.paragraph(note)

    doc.heading(2, "Phụ lục C. Cấu trúc thư mục mã nguồn")
    doc.code_block(
        "config/           duong dan, dac trung, moc thoi gian, dinh nghia model\n"
        "data/processed/    du lieu da lam sach, dac trung va tap ML\n"
        "docs/              tai lieu giai thich va so do kien truc\n"
        "experiments/       tuning_history.csv, manual_config.json, registry, lock\n"
        "models/            final_model.pkl va model_metadata.json\n"
        "reports/           14 tep bao cao danh gia va provenance\n"
        "scripts/           lenh chay tung buoc va pipeline chinh thuc\n"
        "services/          preprocessing, feature, tuning, evaluation, prediction, chatbot\n"
        "static/            CSS, theme va Chart.js dong goi san\n"
        "templates/         sau trang HTML ke thua base.html\n"
        "tests/             bo kiem thu tu dong\n"
        "app.py             Flask backend va cac route"
    )

    doc.heading(2, "Phụ lục D. Tham số của Final Model")
    doc.table(
        ["Trường", "Giá trị"],
        [
            ["Mô hình", "Random Forest (model_id = 3)"],
            ["n_estimators", "130"],
            ["max_depth", "8"],
            ["min_samples_leaf", "100"],
            ["max_features", "0.2"],
            ["class_weight", "balanced_subsample"],
            ["random_state", "42"],
            ["Ngưỡng quyết định", "0,49 chọn từ xác suất OOF trên TRAIN"],
            ["CV F1_UP", "0,4703 (độ lệch chuẩn 0,0250)"],
            ["Huấn luyện lại đến", "10/04/2026 (TRAIN + VALIDATION, 486.854 dòng)"],
            ["Fingerprint nội dung", "f162b3e8 (dạng rút gọn)"],
            ["Thời điểm huấn luyện", "20/07/2026 23:08:36"],
        ],
        widths=[3200, 5870],
        caption="Tham số và siêu dữ liệu của Final Model đang phục vụ",
    )
