"""Chuong 3 phan C: kiem thu (3.3)."""

from __future__ import annotations

from pathlib import Path

from docx_builder import DocxBuilder


def build_testing(doc: DocxBuilder, assets: Path) -> None:
    doc.heading(2, "3.3. Kiểm thử")

    doc.heading(3, "3.3.1. Kế hoạch kiểm thử")
    doc.paragraph(
        "Với một hệ thống học máy, kiểm thử không chỉ là kiểm tra phần mềm chạy đúng mà còn "
        "phải chứng minh quy trình thực nghiệm không rò rỉ dữ liệu. Vì vậy kế hoạch kiểm thử "
        "chia thành bốn mức, dùng chung công cụ unittest có sẵn trong Python và Flask test "
        "client cho phần giao diện."
    )
    doc.table(
        ["Mức kiểm thử", "Phạm vi", "Tệp kiểm thử tiêu biểu"],
        [
            ["Kiểm thử đơn vị", "Gán nhãn, chia fold, chọn ngưỡng, tham số Tuning Lab",
             "test_data_protocol.py, test_recent_cv.py, test_decision_policy.py, test_tuning_lab.py"],
            ["Kiểm thử tích hợp", "Chọn mô hình, refit, khóa TEST, sổ đăng ký, khóa pipeline",
             "test_model_selection.py, test_unified_pipeline.py, test_tuning_history.py"],
            ["Kiểm thử hệ thống", "Luồng suy luận đầu cuối, route Flask, chatbot và tool",
             "test_prediction_flow.py, test_chatbot.py, test_chatbot_tools_upgrade.py"],
            ["Kiểm thử chấp nhận", "Giao diện, điều hướng, khả năng truy cập, đối thoại tự nhiên",
             "test_ui_shell.py, test_chatbot_ui_upgrade.py, test_chatbot_upgrade_integration.py"],
        ],
        widths=[1900, 3300, 3870],
        caption="Bốn mức kiểm thử và phạm vi tương ứng",
    )
    doc.paragraph(
        "Bộ kiểm thử hiện có 14 tệp với 183 hàm kiểm thử, chạy bằng lệnh "
        "python -m pytest tests/ hoặc python -m unittest discover -s tests. Toàn bộ kiểm thử "
        "chạy trên dữ liệu tổng hợp nhỏ được sinh ngay trong tệp kiểm thử, không phụ thuộc tệp "
        "CSV thật, nên thời gian chạy ngắn và kết quả không thay đổi theo dữ liệu."
    )
    doc.paragraph(
        "Bảng dưới đây liệt kê các ca kiểm thử quan trọng nhất, mỗi ca gắn với một quy tắc mà "
        "hệ thống bắt buộc phải giữ."
    )
    doc.table(
        ["Ca kiểm thử", "Quy tắc được bảo vệ", "Kết quả mong đợi"],
        [
            ["test_target_uses_exact_fifth_common_market_session",
             "Nhãn phải trỏ đúng phiên thứ năm trên lịch thị trường chung",
             "label_end_date khớp phiên thứ năm của thị trường"],
            ["test_missing_price_on_exact_future_session_is_dropped_not_skipped",
             "Thiếu giá tại phiên đích thì loại dòng, không nhảy phiên",
             "Dòng thiếu giá không xuất hiện trong tập dữ liệu"],
            ["test_folds_use_dates_gap_sessions_and_purge_label_overlap",
             "CV chia theo ngày, đủ 5 phiên gap và purge nhãn",
             "Không ngày nào nằm ở hai phía; không nhãn nào chồng lấn"],
            ["test_default_boundaries_roll_with_latest_labeled_session",
             "Ba mốc thời gian trượt theo phiên mới nhất có nhãn",
             "Mốc TEST bằng phiên có nhãn mới nhất"],
            ["test_cv_chooses_constrained_oof_probability_cutoff",
             "Ngưỡng OOF phải thỏa trần tỷ lệ UP và sàn precision",
             "Ngưỡng được chọn không thoái hóa thành luôn dự báo UP"],
            ["test_only_three_candidates_are_scored_on_validation",
             "Chỉ ba họ mô hình được chấm trên VALIDATION",
             "Baseline chỉ để so sánh, không thể được chọn"],
            ["test_candidate_below_baseline_stops_before_test",
             "Ứng viên yếu hơn baseline bị chặn trước khi chạm TEST",
             "Pipeline không đánh giá TEST khi cổng VALIDATION không đạt"],
            ["test_fresh_winner_is_refit_then_the_exact_tested_artifact_is_saved",
             "Mô hình được lưu đúng là đối tượng vừa đánh giá TEST",
             "Không có bước refit nào sau khi xem TEST"],
            ["test_metadata_separates_validation_selection_from_final_test",
             "Metadata tách chỉ số chọn mô hình và chỉ số TEST",
             "Hai nhóm chỉ số nằm ở hai khóa riêng biệt"],
            ["test_registry_keeps_multiple_snapshots_and_blocks_duplicates",
             "Một ảnh chụp dữ liệu chỉ được đánh giá TEST một lần",
             "Lần chạy lặp lại bị từ chối"],
            ["test_pipeline_lock_can_only_be_released_by_owner",
             "Chỉ tiến trình giữ khóa được giải phóng khóa pipeline",
             "Tiến trình khác không thể mở khóa"],
            ["test_legacy_wrong_policy_and_wrong_fingerprint_never_enter_ranking",
             "Lần chạy sai giao thức hoặc sai fingerprint bị loại",
             "Xếp hạng chỉ gồm lần chạy hợp lệ"],
            ["test_inference_uses_threshold_from_artifact_metadata",
             "Suy luận dùng ngưỡng từ hiện vật, không ghi cứng",
             "Nhãn đổi theo ngưỡng trong metadata"],
            ["test_inference_exposes_baseline_warning_from_metadata",
             "Cảnh báo baseline phải hiển thị ra ngoài",
             "Kết quả dự báo kèm cảnh báo khi baseline_passed sai"],
            ["test_stock_signals_reject_duplicates_and_out_of_scope_before_inference",
             "Tool chatbot chặn mã trùng và mã ngoài phạm vi",
             "Trả lỗi có mã lỗi rõ ràng, không suy luận"],
            ["test_page_has_exactly_one_h1",
             "Mỗi trang có đúng một tiêu đề cấp một",
             "Cấu trúc tiêu đề hợp chuẩn truy cập"],
        ],
        widths=[3500, 2900, 2670],
        caption="Các ca kiểm thử tiêu biểu và quy tắc được bảo vệ",
    )
    doc.paragraph(
        "Ngoài kiểm thử tự động, pipeline còn có các cổng kiểm tra chạy ngay trong lúc thực "
        "thi: verify_protocol_splits chặn rò rỉ ranh giới, verify_model_selection chặn sai "
        "lệch mô hình được chọn, kiểm tra fingerprint sau khi ghi ml_dataset.csv để bảo đảm "
        "tệp ghi ra khớp dữ liệu vừa tính, và sổ đăng ký đánh giá chặn việc chạy TEST hai lần "
        "trên cùng một ảnh chụp dữ liệu."
    )

    doc.heading(3, "3.3.2. Kết quả và phân tích")

    doc.heading(4, "3.3.2.1. Kết quả kiểm định chéo trên TRAIN")
    doc.paragraph(
        "Ba cấu hình đã chốt được chấm bằng kiểm định chéo bốn fold trên TRAIN với dữ liệu từ "
        "năm 2021. Kết quả trong reports/tuning_results.csv như sau."
    )
    doc.table(
        ["Mô hình", "Siêu tham số đã chốt", "CV F1_UP", "Độ lệch chuẩn", "Precision_UP", "Recall_UP", "Ngưỡng"],
        [
            ["Logistic Regression", "C = 2,68e-05; solver = liblinear", "0,4573", "0,0420", "0,4055", "0,5429", "0,49"],
            ["Random Forest", "130 cây; max_depth 8; leaf 100; max_features 0,2", "0,4703", "0,0250", "0,4123", "0,5548", "0,49"],
            ["Gradient Boosting", "110 stage; lr 0,25; depth 2; subsample 0,6", "0,4714", "0,0237", "0,4200", "0,5408", "0,49"],
        ],
        widths=[1700, 2600, 900, 1000, 1100, 950, 820],
        caption="Kết quả kiểm định chéo bốn fold trên TRAIN",
    )
    doc.paragraph(
        "Ba mô hình cho CV F1_UP rất gần nhau, chênh lệch giữa cao nhất và thấp nhất chỉ "
        "khoảng 0,014. Điều này cho thấy tín hiệu nằm ở đặc trưng và ở cách đặt bài toán, "
        "không nằm ở việc chọn thuật toán. Random Forest và Gradient Boosting có độ lệch chuẩn "
        "giữa các fold nhỏ hơn Logistic Regression, tức ổn định hơn qua các giai đoạn thị "
        "trường khác nhau."
    )
    doc.paragraph(
        "Chi tiết từng fold trong reports/cv_fold_results.csv cho thấy chất lượng giảm theo "
        "thời gian: với Random Forest, F1_UP lần lượt là 0,4842 ở fold 1 (kiểm định từ "
        "26/11/2021 đến 18/10/2022), 0,5019 ở fold 2, 0,4361 ở fold 3 và 0,4590 ở fold 4 "
        "(kiểm định đến 23/06/2025). Biến động này phản ánh thay đổi chế độ thị trường và là "
        "lý do độ lệch chuẩn giữa các fold được báo cáo cùng giá trị trung bình."
    )

    doc.heading(4, "3.3.2.2. Chọn họ mô hình trên VALIDATION")
    doc.paragraph(
        "Ba ứng viên được fit trên toàn bộ TRAIN rồi chấm trên VALIDATION gồm 67.047 dòng, "
        "từ 11/07/2025 đến 03/04/2026. Bảng so sánh trong reports/model_comparison.csv gồm cả "
        "hai baseline hằng số."
    )
    doc.table(
        ["Đối tượng", "Accuracy", "Precision_UP", "Recall_UP", "F1_UP", "F1_NOT_UP"],
        [
            ["Logistic Regression", "0,5736", "0,3881", "0,4949", "0,4350", "0,6575"],
            ["Random Forest (được chọn)", "0,5622", "0,3952", "0,6026", "0,4773", "0,6234"],
            ["Gradient Boosting", "0,5735", "0,4009", "0,5778", "0,4734", "0,6416"],
            ["Baseline luôn UP", "0,3318", "0,3318", "1,0000", "0,4982", "0,0000"],
            ["Baseline luôn NOT_UP", "0,6682", "0,0000", "0,0000", "0,0000", "0,8011"],
        ],
        widths=[2300, 1300, 1500, 1350, 1300, 1320],
        caption="Kết quả trên VALIDATION của ba ứng viên và hai baseline",
    )
    doc.paragraph(
        "Random Forest thắng vì F1_UP cao nhất trong ba ứng viên. Tuy nhiên baseline luôn dự "
        "báo UP có F1_UP 0,4982, cao hơn 0,4773 của Random Forest. Hệ thống ghi nhận đúng "
        "thực tế này: trong model_metadata.json, khóa validation_baseline_passed có giá trị "
        "false kèm cảnh báo Selected candidate VALIDATION F1_UP=0.477340 is below Always "
        "UP=0.498219."
    )
    doc.paragraph(
        "Cần đọc con số này đúng bản chất. Baseline luôn UP đạt F1_UP cao nhờ Recall_UP bằng "
        "1, nhưng Precision_UP của nó chỉ bằng tỷ lệ UP trong dữ liệu là 0,3318 và F1_NOT_UP "
        "bằng 0, tức nó vô dụng khi cần phân biệt. Random Forest có Precision_UP 0,3952 và "
        "F1_NOT_UP 0,6234, nên xét trên cả hai lớp thì mô hình vẫn mang thông tin. Điều đó "
        "không xóa được sự thật là theo tiêu chí F1_UP đã chốt trước, mô hình chưa vượt "
        "baseline."
    )

    doc.heading(4, "3.3.2.3. Đánh giá một lần trên TEST")
    doc.paragraph(
        "Random Forest được fit lại trên TRAIN cộng VALIDATION gồm 486.854 dòng, sau đó đánh "
        "giá đúng một lần trên TEST gồm 20.350 dòng của 392 mã, từ 13/04/2026 đến 13/07/2026. "
        "Tỷ lệ UP trong TEST chỉ còn 23,75%, thấp hơn nhiều so với 33,18% của VALIDATION."
    )
    doc.table(
        ["Đối tượng", "Accuracy", "Precision_UP", "Recall_UP", "F1_UP", "F1_NOT_UP"],
        [
            ["Random Forest (Final Model)", "0,5766", "0,2890", "0,5356", "0,3754", "0,6798"],
            ["Baseline luôn UP", "0,2375", "0,2375", "1,0000", "0,3839", "0,0000"],
            ["Baseline luôn NOT_UP", "0,7625", "0,0000", "0,0000", "0,0000", "0,8652"],
        ],
        widths=[2500, 1300, 1500, 1350, 1250, 1170],
        caption="Kết quả đánh giá một lần trên TEST của Final Model và hai baseline",
    )
    doc.paragraph(
        "F1_UP trên TEST là 0,3754, thấp hơn baseline luôn UP là 0,3839. Hệ thống ghi "
        "baseline_passed bằng false và lưu cảnh báo Final Model chưa vượt baseline always-UP "
        "trên TEST vào metadata; cảnh báo này hiện ra ở trang đánh giá và trang dự báo thay vì "
        "bị ẩn đi. Đây là lựa chọn thiết kế có chủ đích: báo cáo trung thực quan trọng hơn một "
        "con số đẹp."
    )
    doc.paragraph(
        "Ma trận nhầm lẫn trong reports/confusion_matrix.csv giải thích rõ hơn. Mô hình dự báo "
        "UP cho 8.960 dòng, trong đó 2.589 dòng đúng và 6.371 dòng sai; với lớp NOT_UP, mô "
        "hình đúng 9.145 dòng và bỏ sót 2.245 dòng UP."
    )
    doc.table(
        ["", "Dự báo NOT_UP", "Dự báo UP"],
        [
            ["Thực tế NOT_UP", "9.145", "6.371"],
            ["Thực tế UP", "2.245", "2.589"],
        ],
        widths=[3070, 3000, 3000],
        caption="Ma trận nhầm lẫn của Final Model trên TEST",
    )
    confusion = assets.parent.parent / "reports" / "confusion_matrix.png"
    if confusion.exists():
        doc.image(confusion, "Ma trận nhầm lẫn của Final Model trên tập TEST")
        doc.paragraph(
            "Hình trên trực quan hóa đúng bốn ô của bảng vừa nêu. Ô sáng nhất nằm ở hàng thực "
            "tế NOT_UP, cho thấy phần lớn dữ liệu TEST thuộc lớp NOT_UP. Hai ô lỗi có ý nghĩa "
            "khác nhau với người dùng: 6.371 dòng dự báo UP nhưng thực tế không tăng là tín "
            "hiệu sai gây thiệt hại nếu hành động theo, còn 2.245 dòng UP bị bỏ sót chỉ là cơ "
            "hội không được báo. Ngưỡng 0,49 nghiêng về phía tăng recall nên loại lỗi thứ nhất "
            "chiếm tỷ trọng lớn hơn."
        )
    doc.paragraph(
        "Khoảng cách giữa VALIDATION và TEST cho thấy hai nguyên nhân. Thứ nhất, phân phối "
        "nhãn dịch chuyển: tỷ lệ UP giảm gần 10 điểm phần trăm, nên ngưỡng 0,49 được chọn trên "
        "dữ liệu cũ trở nên quá rộng và sinh nhiều dự báo UP sai. Thứ hai, TEST chỉ dài khoảng "
        "ba tháng nên chịu ảnh hưởng mạnh của một giai đoạn thị trường cụ thể. Cả hai đều là "
        "biểu hiện của độ khó thực sự của bài toán dự báo ngắn hạn, không phải lỗi cài đặt."
    )

    doc.heading(4, "3.3.2.4. Mức quan trọng của đặc trưng")
    doc.paragraph(
        "Random Forest cung cấp mức quan trọng toàn cục của đặc trưng, được ghi vào "
        "reports/feature_importance.csv. Tám đặc trưng dẫn đầu chiếm khoảng 65% tổng mức quan "
        "trọng."
    )
    doc.table(
        ["Đặc trưng", "Mức quan trọng", "Diễn giải"],
        [
            ["volatility_20d", "0,1968", "Độ biến động 20 phiên, yếu tố mạnh nhất"],
            ["month", "0,0908", "Hiệu ứng mùa và bối cảnh thời gian"],
            ["volatility_5d", "0,0703", "Độ biến động ngắn hạn"],
            ["return_20d", "0,0677", "Đà giá trung hạn"],
            ["return_1d", "0,0634", "Biến động phiên gần nhất"],
            ["dist_high20", "0,0581", "Khoảng cách tới đỉnh 20 phiên"],
            ["dist_low20", "0,0557", "Khoảng cách tới đáy 20 phiên"],
            ["return_3d", "0,0544", "Đà giá rất ngắn hạn"],
        ],
        widths=[2400, 1800, 4870],
        caption="Tám đặc trưng có mức quan trọng cao nhất",
    )
    importance = assets / "feature_importance_top8.png"
    if importance.exists():
        doc.image(importance, "Tám đặc trưng có mức quan trọng cao nhất trong Final Model")
    doc.paragraph(
        "Kết quả này hợp lý về mặt tài chính: khả năng một mã tăng hơn 1% trong năm phiên phụ "
        "thuộc trước hết vào việc mã đó có biến động mạnh hay không, sau đó mới đến hướng của "
        "đà giá và vị thế so với vùng đỉnh đáy gần nhất. Ba đặc trưng trung bình động sma5, "
        "sma20, sma50 có mức quan trọng thấp nhất vì chúng mang giá tuyệt đối, còn thông tin "
        "hữu ích đã được các tỷ số close_vs_sma20 và sma20_vs_sma50 biểu diễn. Cần lưu ý mức "
        "quan trọng này là tính chất toàn cục của mô hình, không phải lời giải thích cho dự báo "
        "của một mã cụ thể."
    )