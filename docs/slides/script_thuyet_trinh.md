# Script thuyết trình niên luận CT239H
# Dự báo xu hướng giá cổ phiếu HOSE bằng Machine Learning

Bản đọc kèm: docs/slides/thuyet_trinh_nien_luan.pptx (24 slide).
Thời lượng mục tiêu: 15 phút trình bày + Q&A.
Mọi số liệu trong script đã đối chiếu trực tiếp với reports/, models/model_metadata.json,
experiments/manual_config.json và tests/ tại thời điểm soạn.

Cách dùng:
- NÓI: lời thoại đọc được ngay.
- NẾU BỊ HỎI: câu trả lời dự trữ cho hội đồng.
- CƠ SỞ TRONG CODE: để bạn hiểu mình đang nói gì, không đọc lớn.

---

## Phân bổ thời gian

| Khối | Slide | Thời lượng |
|------|-------|-----------|
| Mở đầu | 1-2 | 40 giây |
| Chương 1 | 3-4 | 1 phút 25 |
| Chương 2 | 5 | 40 giây |
| Chương 3 - dữ liệu | 6-8 | 2 phút 20 |
| Chương 3 - kiến trúc + chatbot | 9-11 | 2 phút |
| Chương 3 - giao thức | 12-15 | 3 phút 15 |
| Chương 3 - kết quả | 16-19 | 3 phút 10 |
| Sản phẩm + kiểm thử | 20-21 | 1 phút 15 |
| Chương 4 | 22-24 | 1 phút 45 |

Nếu bị áp thời gian: rút slide 11 (demo chatbot) và slide 15 (Tuning Lab) còn một câu mỗi slide.
Không cắt slide 7, 12, 17 - đó là ba slide tạo ra giá trị học thuật của đề tài.

---

## SLIDE 1 - Bìa (20 giây)

NÓI:
"Kính thưa cô và các bạn. Em là Lê Trần Hiếu Nhân, mã số B2308203. Em xin trình bày niên luận
cơ sở học phần CT239H, đề tài Dự báo xu hướng giá cổ phiếu HOSE bằng machine learning, dưới sự
hướng dẫn của cô Phan Phương Lan. Bài trình bày khoảng 15 phút, đi theo đúng bốn chương của
báo cáo."

LƯU Ý: không đọc lại tên đề tài lần hai. Sang slide ngay.

---

## SLIDE 2 - Nội dung trình bày (20 giây)

NÓI:
"Bài gồm năm khối. Chương 1 đặt vấn đề và mục tiêu. Chương 2 cơ sở lý thuyết. Chương 3 là phần
lớn nhất, gồm dữ liệu, feature, kiến trúc hệ thống và thực nghiệm. Sau đó là kết quả trên
VALIDATION và TEST. Cuối cùng chương 4 kết luận và hướng phát triển.

Em xin lưu ý trước một điểm: em sẽ trình bày kỹ giao thức chống rò rỉ dữ liệu trước khi công bố
kết quả, vì với bài toán dự báo chứng khoán, con số chỉ có nghĩa khi giao thức đúng."

---

## SLIDE 3 - Đặt vấn đề (45 giây)

NÓI:
"Bài toán xuất phát từ hai khó khăn.

Thứ nhất là khối lượng. Dữ liệu em thu thập có 400 mã trên sàn HOSE, hơn 554 nghìn dòng OHLCV
từ tháng 8 năm 2019 đến tháng 7 năm 2026. Mỗi phiên, một nhà đầu tư cá nhân không thể tính lại
chỉ báo kỹ thuật cho toàn bộ số mã đó bằng tay.

Thứ hai, và đây là khó khăn nghiêm trọng hơn: rất nhiều thử nghiệm dự báo chứng khoán cho ra
chỉ số rất đẹp nhưng vô giá trị, vì rò rỉ dữ liệu. Em nhận diện ba dạng rò rỉ.

Dạng một: nhãn t+5 được lấy theo dòng thứ năm của riêng từng mã. Mã nào nghỉ giao dịch vài
phiên thì dòng thứ năm của nó đã là một ngày khác, nên hai mã có nhãn ở hai mốc thời gian khác
nhau, không so sánh được với nhau.

Dạng hai: kiểm định chéo trộn các dòng của cùng một phiên vào cả train và test. Vài mã của ngày
5 tháng 7 nằm ở train, vài mã cùng ngày đó nằm ở test, mô hình gián tiếp biết trước trạng thái
thị trường của phiên đang được chấm.

Dạng ba: chọn mô hình dựa trên tập TEST rồi báo cáo luôn kết quả TEST đó. Khi đó TEST không còn
là dữ liệu chưa từng thấy.

Vì vậy trọng tâm của niên luận không chỉ là huấn luyện một mô hình, mà là xây một quy trình
thực nghiệm có kỷ luật."

NẾU BỊ HỎI - "Em lấy dữ liệu ở đâu?":
"Bằng scripts/fetch_hose_data.py, dùng thư viện vnstock, nguồn KBS, có retry và nghỉ 3,5 giây
giữa các mã. Dữ liệu thô lưu thành một tệp CSV chia sẻ nằm ngoài repo để pipeline và các lần
chạy khác đọc chung một bản."

CƠ SỞ TRONG CODE: reports/pipeline_summary.json khối dataset_report - 554.897 dòng, 400 mã,
2019-08-14 đến 2026-07-20, 0 giá trị thiếu, 0 dòng trùng symbol+trading_date, 233 dòng sai
logic OHLC.

---

## SLIDE 4 - Mục tiêu và phạm vi (40 giây)

NÓI:
"Bài toán được phát biểu rõ là phân loại nhị phân. Với một mã tại một phiên t, câu hỏi là: giá
đóng cửa tại đúng phiên thị trường thứ năm sau đó có cao hơn 1% so với hiện tại hay không. Nếu
có thì nhãn UP, không thì NOT_UP.

Em đặt sáu mục tiêu: một pipeline từ OHLCV thô đến dataset học máy; nhãn tính theo đúng phiên
thị trường thứ năm; chia dữ liệu theo thời gian có purge; tuning ba họ mô hình bằng kiểm định
chéo chuỗi thời gian; chọn mô hình trên VALIDATION và chấm TEST đúng một lần; và cuối cùng là
sản phẩm gồm web Flask, chatbot nội bộ, bộ kiểm thử tự động.

Phạm vi giới hạn ở dữ liệu giá và khối lượng theo ngày, không dùng tin tức hay báo cáo tài
chính. Kết quả phục vụ học tập, không phải khuyến nghị đầu tư."

NẾU BỊ HỎI - "Vì sao chọn 5 phiên và 1%?":
"5 phiên là khoảng một tuần giao dịch, đủ dài để vượt nhiễu trong ngày nhưng vẫn dùng được cho
quyết định ngắn hạn. Ngưỡng 1% để loại các dao động rất nhỏ vốn không có ý nghĩa sau chi phí
giao dịch. Hai giá trị này khai báo tại config/settings.py qua PREDICTION_HORIZON và
UP_THRESHOLD, đổi một chỗ là toàn pipeline đổi theo."

---

## SLIDE 5 - Cơ sở lý thuyết và công cụ (40 giây)

NÓI:
"Sau khi sinh feature, dữ liệu của em là dữ liệu bảng, mỗi dòng là một cặp mã-phiên với 20 cột
số. Vì vậy em chọn ba họ mô hình phù hợp dữ liệu bảng: Logistic Regression làm mốc tuyến tính,
Random Forest theo hướng bagging để giảm variance, và Gradient Boosting theo hướng boosting
tuần tự trên cây nông.

Về độ đo, em chọn F1 của lớp UP làm metric chính, không dùng accuracy. Lý do là lớp UP chỉ
chiếm 37,6% dữ liệu. Trên tập TEST tỷ lệ này còn thấp hơn, chỉ 23,8%, nên một mô hình luôn dự
báo NOT_UP đã đạt accuracy hơn 76% mà không có giá trị sử dụng nào.

Để tránh tự đánh giá quá cao, em đưa vào hai baseline hằng số: always-UP và always-NOT_UP. Mọi
kết quả đều phải so với hai mốc này.

Công cụ đều là thư viện Python phổ biến: scikit-learn cho mô hình, CV và metric; pandas và numpy
xử lý bảng; Flask cho web; matplotlib vẽ confusion matrix; vnstock thu thập dữ liệu."

NẾU BỊ HỎI - "Sao không dùng LSTM?":
"Em có nêu trong hướng phát triển. Lý do chưa làm ở niên luận này: mục tiêu chính của em là
thiết lập giao thức chống rò rỉ và có được một con số trung thực làm mốc. Ba họ mô hình bảng đã
đủ làm mốc đó với chi phí huấn luyện thấp, và khi đã có giao thức thì thay mô hình là bước
tương đối cơ học."

---

## SLIDE 6 - Làm sạch và lọc mã (45 giây)

NÓI:
"Slide này là phễu dữ liệu, cho thấy từng bước mất bao nhiêu dòng.

Bắt đầu 554.897 dòng thô. Bước làm sạch loại 233 dòng có nến sai logic OHLC, ví dụ giá cao nhất
thấp hơn giá đóng cửa. Không có dòng nào thiếu giá trị và không có dòng nào trùng mã với ngày,
nên hai bước đó không mất dòng.

Tiếp theo em lọc điều kiện mã: mã phải có tối thiểu 250 phiên giao dịch. Bốn mã bị loại vì mới
lên sàn hoặc dữ liệu quá ngắn, còn lại 396 mã.

Bước sinh feature làm mất khoảng 37 nghìn dòng, vì cửa sổ rolling dài nhất là 50 phiên nên mỗi
mã phải bỏ 50 phiên đầu tiên.

Cuối cùng bước gắn nhãn loại thêm hai nhóm: 1.768 dòng nằm ở 5 phiên cuối dataset nên chưa có
phiên t+5, và 4.468 dòng mà mã đó không có giá đóng tại phiên t+5. Kết quả còn 510.862 dòng có
nhãn hợp lệ trên 396 mã."

NẾU BỊ HỎI - "Vì sao 250 phiên?":
"250 phiên xấp xỉ một năm giao dịch. Dưới mức đó, sau khi đã bỏ 50 phiên đầu cho rolling
feature, phần dữ liệu còn lại quá ít để mã đó xuất hiện ổn định qua các fold thời gian."

CƠ SỞ TRONG CODE: services/preprocessing.py và khối clean_report trong pipeline_summary.json.

---

## SLIDE 7 - Định nghĩa nhãn chống rò rỉ (55 giây, slide bắt buộc)

NÓI:
"Đây là điểm kỹ thuật quan trọng nhất của phần dữ liệu, và cũng là chỗ em sửa khác với cách làm
thông thường.

Cách thông thường là với mỗi mã, lấy giá đóng cửa dịch xuống năm dòng. Vấn đề: nếu mã đó nghỉ
giao dịch, dòng thứ năm của nó có thể là phiên thứ bảy hoặc thứ tám của thị trường. Hai mã sẽ
có nhãn ở hai kỳ hạn khác nhau, và khi chia dữ liệu theo ngày thì nhãn có thể lấn qua ranh giới
mà ta không biết.

Cách của em: trước tiên dựng lịch phiên chung của toàn sàn từ tất cả dữ liệu đã làm sạch, rồi
dịch chính danh sách ngày đó xuống năm bước. Kết quả là một ánh xạ dùng chung cho mọi mã: phiên
t ứng với phiên đích t+5 nào. Sau đó em tự join bảng giá theo cặp mã và ngày đích để lấy đúng
giá đóng cửa tại phiên đó.

Nếu mã không có giao dịch đúng phiên đích, em bỏ dòng, có ý không lấy giá gần nhất để lấp, vì
lấp như vậy sẽ tạo nhãn sai kỳ hạn. Đó là 4.468 dòng ở slide trước.

Mỗi dòng còn lưu thêm một cột label_end_date, tức ngày mà nhãn của dòng đó kết thúc. Cột này
được dùng lại ở bước chia dữ liệu để purge, em sẽ nói ở slide 12.

Kết quả phân bố nhãn: 37,6% UP và 62,4% NOT_UP, tức lớp cần dự báo là lớp thiểu số."

NẾU BỊ HỎI - "Làm sao chứng minh nhãn không rò rỉ?":
"Có kiểm thử tự động trong tests/test_data_protocol.py: dựng một bảng giá nhỏ có mã nghỉ giao
dịch, rồi khẳng định nhãn phải bằng đúng giá tại phiên thị trường thứ năm, và dòng thiếu giá
tại phiên đó phải bị loại thay vì được lấp. Ngoài ra có test khẳng định bốn cột liên quan
tương lai không nằm trong danh sách feature."

CƠ SỞ TRONG CODE: services/feature_engineering.py, hàm create_labels - market_dates.shift(-5)
tạo future_by_date, merge với validate="many_to_one", rồi dropna theo future_close_5d.

---

## SLIDE 8 - Feature engineering (40 giây)

NÓI:
"Em xây 20 feature, chia bảy nhóm. Nhóm return gồm lợi suất 1, 3, 5, 10, 20 phiên và lợi suất
trong ngày từ giá mở đến giá đóng. Nhóm trung bình động gồm SMA 5, 20, 50 và hai tỷ lệ so sánh
tương đối. Nhóm động lượng có RSI 14. Nhóm biến động gồm độ lệch chuẩn lợi suất 5 và 20 phiên,
cùng biên độ giá trong phiên. Nhóm khối lượng gồm thay đổi khối lượng và tỷ lệ so với trung
bình 20 phiên. Nhóm vị thế giá là khoảng cách tới đỉnh và đáy 20 phiên. Cuối cùng là tháng.

Hai ràng buộc quan trọng. Thứ nhất, mọi phép rolling đều groupby theo mã, không trộn dữ liệu
giữa các mã, nên feature của FPT không bao giờ chứa giá của SSI. Thứ hai, mọi feature chỉ dùng
dòng hiện tại và các dòng quá khứ, không dùng center window.

Bốn cột liên quan tương lai là future_close_5d, future_return_5d, target và label_end_date. Cả
bốn đều bị kiểm tra tự động để không lọt vào tập feature. Thứ tự 20 feature được khai báo một
lần trong config/settings.py và lưu lại trong model_metadata.json, nên lúc dự báo, cột được sắp
đúng thứ tự mô hình đã học."

NẾU BỊ HỎI - "Vì sao có feature month?":
"Ban đầu em thêm month để mô hình bắt được yếu tố mùa vụ. Nhưng ở slide diễn giải, month lại là
feature quan trọng thứ hai, và em xem đó là dấu hiệu cần thận trọng chứ không phải điểm tốt: nó
có thể chỉ phản ánh chế độ thị trường của giai đoạn huấn luyện. Đây là một trong các việc em nêu
ở hướng phát triển."

---

## SLIDE 9 - Kiến trúc ứng dụng (40 giây)

NÓI:
"Hệ thống tách làm hai phần rõ rệt.

Phần thứ nhất là pipeline offline. Chạy bằng scripts/run_pipeline.py, đi qua kiểm tra dữ liệu,
làm sạch, sinh feature, gán nhãn, chia tập, tuning, chọn mô hình, đánh giá TEST, rồi ghi ra hiện
vật: final_model.pkl, model_metadata.json và các tệp báo cáo trong reports/.

Phần thứ hai là tầng phục vụ. Web Flask và CLI chỉ đọc lại hiện vật đó, không tự huấn luyện
lại. Đây là lựa chọn thiết kế có chủ đích: nhờ vậy số hiển thị trên web luôn khớp số trong báo
cáo, và người chấm có thể mở tệp report để đối chiếu.

Về mã nguồn: config giữ đường dẫn, danh sách feature và mốc thời gian; services giữ logic xử
lý, tuning, đánh giá, dự báo; scripts là các lệnh chạy; models và reports là hiện vật công bố;
app.py là backend Flask."

NẾU BỊ HỎI - "Ba mốc thời gian là hằng số cứng?":
"Không. services/protocol_dates.py neo mốc TEST vào phiên có nhãn hợp lệ mới nhất của dataset
rồi lùi lại theo độ dài cửa sổ cố định, gọi là rolling walk-forward. Ba hằng số trong
config/settings.py chỉ là fallback khi dataset chưa đủ dài. Điểm đáng nói là cả hàm cắt split
thật và hàm tính fingerprint đều gọi cùng một hàm này, nên split và fingerprint không thể lệch
nhau. Mốc thực tế của bản đã publish nằm trong model_metadata.json."

---

## SLIDE 10 - Chatbot kiến trúc Action-Decision (45 giây)

NÓI:
"Hệ thống có một chatbot trả lời câu hỏi về chính dự án này. Kiến trúc là Action-Decision: không
vector database, không embedding, không RAG, và không để LLM tự gọi tool trong vòng lặp.

Một lượt hỏi đi qua bốn bước. Trước hết, LLM được gọi lần thứ nhất chỉ để ra quyết định: chọn đúng
một trong năm action và trích xuất arguments theo schema cố định. Năm action đó là GENERAL_CHAT cho
hội thoại ngắn, STOCK_SIGNAL cho tín hiệu một đến năm mã, STOCK_RANKING cho xếp hạng Điểm UP,
PROJECT_INFO cho bảy chủ đề về đồ án, và OUT_OF_SCOPE cho yêu cầu ngoài phạm vi.

Bước hai, backend validate lại chính quyết định đó: đúng hai khóa action và arguments, đúng kiểu và
khoảng giá trị, mã được chuẩn hóa rồi kiểm scope tập trung trước khi chạy inference. Sai schema thì
trả 502, không hạ xuống router keyword.

Bước ba, một dispatcher cố định gọi handler tương ứng. Mọi prediction, Điểm UP, xếp hạng và metric
đều do backend tính từ artifact đã publish; LLM không đọc CSV, không gọi model, không tự chọn ngưỡng.

Bước bốn, LLM được gọi lần thứ hai để diễn đạt lại câu trả lời từ đúng JSON số liệu backend đưa.
Nếu call này lỗi, quá deadline hoặc trả sai định dạng, hệ thống giữ nguyên bản formatter
deterministic. Nhờ vậy văn phong mỗi lần có thể khác nhau nhưng số liệu và disclaimer thì không đổi.

Về runtime: mỗi lượt đúng hai LLM call, gửi kèm ba cặp hỏi đáp gần nhất, và một deadline chung phủ
cả hai call. Trang /chat với dock nổi dùng chung API và transcript trong sessionStorage."

NẾU BỊ HỎI - "Nếu LLM chọn sai action hoặc bịa số thì sao?":
"Đây là hai chuyện khác nhau. Bịa số thì bị chặn ngay từ cấu trúc: LLM không cầm dữ liệu, mọi con
số đến từ backend, và call compose chỉ nhận đúng JSON kết quả nên không có nguồn nào để bịa; compose
lỗi thì rơi về bản formatter cố định. Còn chọn sai action thì validator chỉ chặn được sai schema,
không chặn được sai ý; em xử lý bằng bộ scenario evaluator 37 tình huống chạy với provider thật để
chỉnh prompt, chứ không thêm router thứ hai."

---

## SLIDE 11 - Demo chatbot (35 giây)

NÓI:
"Đây là ảnh chụp từ phiên làm việc thật với Flask đang chạy, không phải ảnh minh họa.

Câu thứ nhất hỏi về mô hình và kết quả TEST. Decision chọn PROJECT_INFO, backend đọc metadata và
report nên trả về đúng Random Forest với F1_UP 0,3754, trùng khớp bảng kết quả ở slide 17.

Câu thứ hai hỏi tín hiệu của FPT. Decision chọn STOCK_SIGNAL với đúng một mã; backend kiểm scope
rồi chạy inference, trả về Điểm UP kèm ngưỡng quyết định, và nói rõ đây là dữ liệu offline theo
phiên gần nhất trong dataset, không phải giá hiện tại.

Câu thứ ba hỏi về P/E và tin tức doanh nghiệp. Decision chọn OUT_OF_SCOPE, chatbot từ chối và mời
lại đúng những việc nó làm được. Đó chính là ranh giới phạm vi đang hoạt động, vì hai thông tin đó
không có trong dataset.

Điểm cần nhấn: cả ba câu đều không có con số nào do LLM tự viết ra. Thanh trạng thái phía trên ghi
rõ dữ liệu offline đến ngày nào, và dòng cuối trang là disclaimer do backend gắn, nói rõ kết quả
chỉ để tham khảo chứ không phải khuyến nghị đầu tư."

---

## SLIDE 12 - Giao thức chia dữ liệu (55 giây, slide bắt buộc)

NÓI:
"Em chia ba tập theo mốc thời gian, không trộn ngẫu nhiên. TRAIN có 419.807 dòng từ tháng 10 năm
2019 đến 3 tháng 7 năm 2025. VALIDATION có 67.047 dòng từ 11 tháng 7 năm 2025 đến 3 tháng 4 năm
2026. TEST có 20.350 dòng từ 13 tháng 4 đến 13 tháng 7 năm 2026.

Điểm cần giải thích là hai khoảng trống giữa các tập. Chúng không phải lỗi làm tròn ngày mà là
purge có chủ đích.

Cụ thể: TRAIN không lọc theo ngày giao dịch mà lọc theo label_end_date, tức ngày nhãn kết thúc.
Một dòng ngày 8 tháng 7 có nhãn nhìn tới ngày 15 tháng 7. Nếu em chỉ cắt theo ngày giao dịch thì
mô hình đã học từ một dòng mà đáp án của nó nằm bên trong khoảng VALIDATION. Ràng buộc
label_end_date nhỏ hơn hoặc bằng mốc TRAIN loại đúng những dòng biên đó. Cái giá phải trả là
1.883 dòng ở biên TRAIN-VALIDATION và 1.775 dòng ở biên VALIDATION-TEST bị bỏ, không thuộc tập
nào cả.

Về quy trình: mô hình được chọn trên VALIDATION, sau đó refit trên TRAIN cộng VALIDATION, tổng
486.854 dòng, rồi chấm TEST đúng một lần.

Để bảo đảm 'đúng một lần' không chỉ là lời nói, em tính fingerprint nội dung của snapshot dữ
liệu và ghi vào một registry. Trước khi đánh giá, pipeline kiểm tra fingerprint đó; nếu snapshot
này từng được chấm TEST, pipeline dừng và báo lỗi. Như vậy không thể chấm TEST nhiều lần rồi
chọn lần cho số đẹp nhất."

NẾU BỊ HỎI - "Fingerprint gồm những gì?":
"Có hai loại. Một là fingerprint phạm vi, gồm số dòng, mốc ngày, horizon, ngưỡng và danh sách
feature. Hai là fingerprint nội dung, là SHA-256 trên chính giá trị dữ liệu. Cần cả hai, vì nếu
nhà cung cấp sửa lại giá cũ mà số dòng không đổi thì fingerprint phạm vi vẫn khớp trong khi dữ
liệu đã khác. Điểm nữa: khóa TEST dùng fingerprint của snapshot TRAIN+VALIDATION+TEST, còn lịch
sử tuning dùng fingerprint riêng của TRAIN. Nhờ tách vậy nên fetch dữ liệu mới cho inference
không mở lại khóa TEST."

CƠ SỞ TRONG CODE: services/feature_engineering.py hàm protocol_time_split;
scripts/run_pipeline.py hàm _guard_unevaluated_snapshot.

---

## SLIDE 13 - Thiết lập kiểm định chéo (50 giây)

NÓI:
"Tuning được chấm bằng kiểm định chéo 4 fold trên TRAIN, và có ba chi tiết quan trọng.

Thứ nhất, fold được cắt theo ngày thị trường, không theo chỉ số dòng. Em lấy danh sách ngày duy
nhất của sàn rồi mới chia, nên mọi mã của cùng một phiên luôn nằm cùng một fold. Nếu cắt theo
dòng thì một phiên bị xẻ đôi và sinh ra rò rỉ chéo theo mã.

Thứ hai, giữa train và validation của mỗi fold bỏ 5 phiên, và dòng train chỉ hợp lệ nếu nhãn của
nó đã đóng trước phiên đầu tiên của validation. Đây là purge giống slide trước, nhưng áp trong
từng fold.

Thứ ba, kiểm định chéo chỉ chấm giai đoạn từ năm 2021 trở đi, vì trước đó số mã có đủ lịch sử
còn mỏng.

Về ngưỡng quyết định: em không dùng 0,5 mặc định. Em gom xác suất out-of-fold trên TRAIN, tức
mỗi dòng được chấm bởi mô hình chưa từng thấy nó, rồi quét ngưỡng để tối ưu F1_UP. Ngưỡng chọn
được là 0,49 cho cả ba mô hình, và một ngưỡng duy nhất đó được áp lại cho mỗi fold.

Ngưỡng còn bị ràng buộc: precision không được thấp hơn tỷ lệ UP thật, và tỷ lệ dự báo UP không
vượt 50%. Ràng buộc thứ hai ngăn mô hình suy biến thành gần như luôn dự báo UP để ăn điểm F1.

Kết quả CV: Logistic Regression 0,4573, Random Forest 0,4703, Gradient Boosting 0,4714."

NẾU BỊ HỎI - "Chọn ngưỡng trên TRAIN có phải rò rỉ?":
"Không, vì xác suất dùng để chọn ngưỡng là xác suất out-of-fold, và toàn bộ quá trình chỉ diễn
ra trong TRAIN. VALIDATION và TEST không tham gia bước chọn ngưỡng. Điều cần nói thẳng là ngưỡng
này tối ưu cho phân phối của TRAIN; khi tỷ lệ UP trên TEST tụt xuống 23,8% thì ngưỡng đó trở nên
quá thoáng, và em có phân tích ở slide 18."

---

## SLIDE 14 - Siêu tham số và độ ổn định (45 giây)

NÓI:
"Siêu tham số được chốt thủ công tại Tuning Lab rồi khóa lại trong manual_config.json trước khi
chạy pipeline chính thức.

Hướng chung là giữ mô hình đơn giản để giảm overfit. Random Forest được chốt là 130 cây, độ sâu
tối đa 8, mỗi lá tối thiểu 100 mẫu, và mỗi lần split chỉ xét 20% số feature. Với dữ liệu tài
chính nhiễu cao, để cây mọc sâu là cách nhanh nhất để mô hình học thuộc nhiễu của giai đoạn
huấn luyện.

Logistic Regression có C rất nhỏ, khoảng 2,68 nhân 10 âm 5, tức regularization rất mạnh. Gradient
Boosting dùng cây sâu 2 với learning rate 0,25 và subsample 0,6.

Random Forest thêm class_weight balanced_subsample vì lớp UP là lớp thiểu số. Toàn bộ dùng
random_state 42 nên chạy lại cho ra đúng kết quả cũ.

Biểu đồ bên phải là F1_UP từng fold. Với Random Forest, giá trị dao động từ 0,436 ở fold 3 tới
0,502 ở fold 2, độ lệch chuẩn 0,025. Nghĩa là chất lượng phụ thuộc rõ vào chế độ thị trường của
từng giai đoạn, và đây là thông tin em thấy cần nói ra thay vì chỉ báo cáo giá trị trung bình."

---

## SLIDE 15 - Tuning Lab (40 giây)

NÓI:
"Em không dò siêu tham số bằng GridSearch tự động, mà thử tay qua trang /tuning, vì em muốn hiểu
ảnh hưởng của từng tham số thay vì nhận về một cấu hình không giải thích được.

Mỗi lần chạy, hệ thống ghi một dòng vào experiments/tuning_history.csv gồm tham số, điểm CV từng
fold, ngưỡng và fingerprint dữ liệu. Tổng cộng 350 lần thử: 269 cho Logistic Regression, 51 cho
Random Forest, 30 cho Gradient Boosting. Như vậy quá trình thử là truy vết được.

Điểm quan trọng về giao thức: Tuning Lab chỉ chấm trên TRAIN bằng kiểm định chéo, tuyệt đối
không nhìn VALIDATION hay TEST.

Sau khi chọn, cấu hình được khóa vào manual_config.json kèm run_id và fingerprint. Pipeline
chính thức từ chối chạy nếu thiếu cấu hình của bất kỳ mô hình nào trong ba mô hình, hoặc nếu
fingerprint dữ liệu không khớp cấu hình đã chốt."

NẾU BỊ HỎI - "Chọn tay có chủ quan không?":
"Có phần chủ quan, em thừa nhận. Nhưng nó bị chặn bởi ba điều kiện: điểm dùng để so sánh luôn là
CV trên TRAIN, ngưỡng do thuật toán chọn từ OOF chứ không do em đặt, và mỗi lần thử đều được ghi
lại nên có thể kiểm tra lại. Ngoài ra không có cấu hình nào được chọn dựa trên VALIDATION hay
TEST, nên chủ quan ở đây không tạo ra rò rỉ."

---

## SLIDE 16 - Chọn họ mô hình trên VALIDATION (45 giây)

NÓI:
"Ba mô hình được chấm trên VALIDATION, dùng ngưỡng 0,49 đã chốt từ TRAIN, và không tinh chỉnh
thêm gì trên VALIDATION.

Random Forest đạt F1_UP 0,4773 với recall 0,6026. Gradient Boosting 0,4734. Logistic Regression
0,4350. Random Forest thắng theo tiêu chí thứ nhất là F1_UP cao nhất. Nếu bằng điểm thì tiêu chí
hai là recall_UP cao hơn, tiêu chí ba là mô hình đơn giản hơn.

Nhưng em cần nói thẳng một điều nằm ngay trên bảng: baseline luôn dự báo UP đạt F1_UP 0,4982 trên
VALIDATION, cao hơn cả ba mô hình. Baseline này chỉ đạt được nhờ recall bằng 1, tức đoán UP cho
mọi dòng, đánh đổi bằng accuracy chỉ 33%. Nó vô dụng trong thực tế nhưng vẫn là một mốc so sánh
đúng về mặt số học, và hệ thống ghi lại cảnh báo này vào metadata thay vì bỏ qua."

NẾU BỊ HỎI - "Vì sao baseline always-UP lại có F1 cao?":
"Vì F1 chỉ tính trên lớp UP. Always-UP có recall bằng 1 tuyệt đối, còn precision bằng đúng tỷ lệ
UP của tập đó. Với tỷ lệ UP 33% trên VALIDATION, F1 ra khoảng 0,498. Đây là tính chất đã biết
của F1 trên lớp thiểu số, và chính vì vậy em đưa cả hai baseline hằng số vào bảng để không tự
đánh giá quá cao mô hình của mình."

---

## SLIDE 17 - Đánh giá TEST (50 giây, slide bắt buộc)

NÓI:
"Đây là kết quả chính thức, chấm một lần trên 20.350 dòng TEST từ 13 tháng 4 đến 13 tháng 7 năm
2026.

Random Forest sau khi refit trên TRAIN cộng VALIDATION đạt F1_UP 0,3754, precision_UP 0,2890,
recall_UP 0,5356 và accuracy 0,5766.

Baseline luôn dự báo UP đạt F1_UP 0,3839. Nghĩa là mô hình của em thấp hơn baseline 0,0085 điểm.
Mục tiêu vượt baseline không đạt.

Hệ thống ghi baseline_passed bằng false trong model_metadata.json, kèm cảnh báo được hiển thị
trực tiếp trên giao diện web, không chỉ nằm trong tệp.

Em chọn báo cáo trung thực con số này. Em có thể nới giao thức để có số đẹp hơn: bỏ purge, cắt
ngẫu nhiên thay vì theo thời gian, hoặc chọn ngưỡng trực tiếp trên TEST. Cả ba đều cho chỉ số cao
hơn nhưng đều là rò rỉ dữ liệu, và con số thu được sẽ không có giá trị. Vì vậy em giữ giao thức
và báo cáo đúng kết quả."

NẾU BỊ HỎI - "Vậy đề tài thất bại?":
"Mục tiêu về chất lượng dự báo không đạt, em không tránh điều đó. Nhưng bốn mục tiêu còn lại về
pipeline, nhãn không rò rỉ, so sánh mô hình theo giao thức và công bố hiện vật đều đạt. Và bản
thân kết quả âm này là một kết luận có nội dung: khi loại bỏ ba nguồn rò rỉ, năng lực dự báo thực
tế của chỉ báo kỹ thuật thuần trên dữ liệu ngày là rất hạn chế. Một hệ thống báo F1 0,8 nhờ rò rỉ
sẽ vô dụng khi đưa vào dùng thật; hệ thống của em báo đúng 0,3754 và nói rõ vì sao."

---

## SLIDE 18 - Phân tích lỗi (45 giây)

NÓI:
"Ma trận nhầm lẫn cho thấy bản chất lỗi.

Trong 4.834 phiên tăng thật, mô hình bắt được 2.589, bỏ sót 2.245, tức recall 0,536. Nhưng để bắt
được số đó, mô hình phát ra 6.371 tín hiệu UP sai trên 15.516 dòng NOT_UP. Nên precision chỉ
0,289: cứ khoảng 10 tín hiệu UP thì chưa tới 3 tín hiệu đúng.

Nguyên nhân có hai phần.

Thứ nhất, ngưỡng 0,49 vốn đã ưu tiên recall hơn precision, vì metric mục tiêu là F1_UP.

Thứ hai, và quan trọng hơn: tỷ lệ UP trên TEST chỉ 23,8%, trong khi trên toàn bộ dữ liệu là
37,6% và trên tập OOF của TRAIN là 36,9%. Giai đoạn TEST là giai đoạn ít phiên tăng mạnh hơn hẳn.
Ngưỡng được chốt trên phân phối cũ nên khi áp vào phân phối mới, nó trở nên quá thoáng và
precision tụt mạnh.

Đây chính là hạn chế về hiệu chỉnh ngưỡng theo chế độ thị trường, và là lý do em đưa hiệu chỉnh
xác suất vào hướng phát triển."

---

## SLIDE 19 - Diễn giải mô hình (40 giây)

NÓI:
"Về độ quan trọng feature theo Random Forest.

Cao nhất là volatility_20d với 19,7%. Tiếp theo là month 9,1%, volatility_5d 7,0%, return_20d
6,8%, return_1d 6,3%.

Điều này nói lên một chuyện đáng chú ý: mô hình dựa vào mức biến động nhiều hơn là hướng đi của
giá. Biến động thì dự báo được tương đối, còn hướng đi thì khó. Điều đó nhất quán với việc F1_UP
thấp.

Feature month đứng thứ hai là điểm em thấy cần thận trọng. Nó có thể chỉ đang mã hóa chế độ thị
trường của giai đoạn huấn luyện, chứ không phải một quy luật mùa vụ bền vững. Đây là việc cần
kiểm tra thêm.

Ngược lại, các trung bình động thô như sma5, sma20, sma50 gần như không đóng góp, mỗi cái khoảng
1,3%. Hợp lý, vì thông tin của chúng đã nằm trong các tỷ lệ so sánh tương đối như close_vs_sma20."

---

## SLIDE 20 - Sản phẩm phần mềm (40 giây)

NÓI:
"Sản phẩm giao nộp gồm sáu trang chức năng.

Trang /predict dự báo UP hoặc NOT_UP cho một đến hai mã. Trang /compare so sánh điểm UP hai mã.
Trang /screener xếp hạng điểm UP toàn sàn. Trang /evaluation trình bày kết quả. Trang /tuning là
Tuning Lab. Trang /chat là chatbot.

Điểm thiết kế đáng nói: bốn trang đầu chỉ đọc lại cùng một model artifact do pipeline publish,
không train lại, nên số trên web luôn khớp báo cáo. Điểm UP mà web hiển thị chính là xác suất lớp
UP từ predict_proba, so với ngưỡng 0,49 đọc từ metadata.

Trang đánh giá cố tình tách ba khối: kiểm định chéo trên TRAIN, chọn mô hình trên VALIDATION, và
đánh giá TEST một lần. Em tách vậy để người xem không nhầm ba loại số này với nhau, vì đó là lỗi
đọc kết quả rất thường gặp.

Riêng Tuning Lab có nút chạy pipeline chính thức, nhưng bị chặn bởi ba điều kiện: đủ cấu hình cả
ba mô hình, đúng fingerprint dữ liệu, và snapshot chưa từng được đánh giá TEST.

Cảnh báo chưa vượt baseline được hiển thị ngay trên giao diện, không ẩn trong tệp."

LƯU Ý: nếu web đang chạy và hội đồng muốn xem, mở /evaluation trước, vì đó là trang khớp với
slide 16 và 17.

---

## SLIDE 21 - Kiểm thử (35 giây)

NÓI:
"Bộ kiểm thử trong thư mục tests được mở rộng cùng mã nguồn, nên slide không khóa cứng tổng số.

Quan trọng nhất là nhóm kiểm thử giao thức. tests/test_data_protocol.py có test khẳng định nhãn
dùng đúng phiên thị trường thứ năm và test chặn bốn cột tương lai lọt vào tập feature.
tests/test_recent_cv.py kiểm việc cắt fold theo ngày và purge trong fold.
tests/test_model_selection.py kiểm thứ tự tiêu chí chọn mô hình và khóa TEST.

Ngoài kiểm thử tự động, pipeline còn có cổng kiểm tra chạy trong lúc thực thi: verify_protocol_splits
xác nhận không có rò rỉ ranh giới, và verify_model_selection xác nhận mô hình được publish đúng
là mô hình đã thắng trên VALIDATION. Nếu một điều kiện sai, pipeline dừng chứ không ghi hiện vật
sai.

Phần chưa làm em xin nêu rõ: chưa có kiểm thử hiệu năng và chưa có kiểm thử chấp nhận với người
dùng thật."

---

## SLIDE 22 - Hạn chế (35 giây)

NÓI:
"Hạn chế lớn nhất là chất lượng dự báo chưa vượt baseline trên TEST, nên hệ thống chỉ dùng cho
học tập và nghiên cứu, không phải khuyến nghị đầu tư.

Nguyên nhân chính về mặt mô hình: tập feature chỉ khai thác giá và khối lượng theo ngày. Không có
dữ liệu cơ bản, không có VN-Index, không có dòng tiền khối ngoại, không có tin tức. Với thông tin
hạn chế như vậy, kỳ vọng dự báo hướng giá chính xác là không thực tế.

Thứ hai, ngưỡng quyết định chọn một lần từ OOF trên TRAIN và giữ cố định, chưa hiệu chỉnh theo
chế độ thị trường, và slide 18 cho thấy đúng cái giá của việc đó.

Thứ ba, suy luận chạy trên dữ liệu tĩnh trong CSV, không phải realtime.

Về phần mềm: web chưa có xác thực người dùng, và chưa có kiểm thử hiệu năng lẫn kiểm thử chấp
nhận."

---

## SLIDE 23 - Hướng phát triển (35 giây)

NÓI:
"Hướng phát triển em xếp theo thứ tự ưu tiên, dựa trên phân tích hạn chế.

Ưu tiên một là mở rộng feature: thêm VN-Index, sức mạnh ngành, dòng tiền khối ngoại. Đây là điểm
nghẽn rõ nhất, vì mô hình hiện tại đơn giản là thiếu thông tin.

Ưu tiên hai là hiệu chỉnh xác suất bằng Platt scaling hoặc isotonic regression, rồi chọn ngưỡng
theo mục tiêu sử dụng thay vì giữ một ngưỡng cố định.

Sau đó mới thử các họ mô hình mạnh hơn như XGBoost, LightGBM hay LSTM, nhưng giữ nguyên giao thức
chống rò rỉ hiện có. Em xếp việc này sau vì đổi mô hình mà giữ nguyên tập feature thì cải thiện
sẽ nhỏ.

Tiếp theo là đánh giá theo hướng đầu tư: lợi nhuận sau chi phí giao dịch và mức sụt giảm tối đa,
thay vì chỉ dựa vào độ đo phân loại.

Cuối cùng là tự động walk-forward định kỳ mỗi tháng, và về phần mềm thì thêm xác thực, khóa liên
tiến trình và ghi vết."

---

## SLIDE 24 - Kết luận (40 giây)

NÓI:
"Tổng kết theo năm mục tiêu ban đầu.

Pipeline dữ liệu và feature: đạt, 510.862 dòng có nhãn hợp lệ với 20 feature. Nhãn không rò rỉ
theo phiên thị trường chung: đạt, và có test kiểm chứng. So sánh ba mô hình theo giao thức: đạt,
ba mô hình với kiểm định chéo 4 fold có purge. Công bố hiện vật và web: đạt, gồm final_model.pkl,
web Flask và CLI. Vượt baseline always-UP trên TEST: không đạt, 0,3754 so với 0,3839.

Em báo cáo công khai mục tiêu không đạt vì nó có ý nghĩa phương pháp: khi loại bỏ ba nguồn rò rỉ
dữ liệu, năng lực dự báo thực tế của chỉ báo kỹ thuật thuần trên dữ liệu ngày là rất hạn chế.

Đóng góp chính của niên luận vì vậy không phải là con số F1, mà là một giao thức thực nghiệm có
kỷ luật và tái lập được, cùng một hệ thống cho phép kiểm chứng lại từng con số đã báo cáo.

Em xin hết. Em cảm ơn cô và các bạn đã lắng nghe, và kính mời hội đồng đặt câu hỏi."

---

# PHỤ LỤC A - Câu hỏi dự kiến và câu trả lời

## A1. Nhóm câu hỏi về kết quả

HỎI: "Mô hình không vượt baseline thì làm ra để làm gì?"
ĐÁP: "Baseline always-UP chỉ đạt F1 cao nhờ recall bằng 1, kèm accuracy 23,8% trên TEST. Nó không
dùng được để ra quyết định vì nó không phân biệt gì cả. Mô hình của em có accuracy 0,5766 và vẫn
tách được hai lớp, chỉ là chưa thắng baseline trên đúng chỉ số F1_UP. Giá trị của đề tài là giao
thức đo lường trung thực; nếu không có giao thức đó thì em đã báo một con số cao hơn mà không ai
biết là sai."

HỎI: "Nếu bỏ purge và cắt ngẫu nhiên thì kết quả bao nhiêu?"
ĐÁP: "Em không có con số đó trong repo vì em không chạy cấu hình rò rỉ để so sánh. Về mặt lý
thuyết, cắt ngẫu nhiên trên dữ liệu panel sẽ đưa các dòng cùng ngày và cùng mã vào cả train lẫn
test, nên chỉ số sẽ cao hơn đáng kể nhưng không phản ánh khả năng dự báo tương lai. Nếu hội đồng
muốn, đó là một thử nghiệm đối chứng em có thể bổ sung."

HỎI: "Vì sao chọn F1_UP mà không phải accuracy hay AUC?"
ĐÁP: "Accuracy không dùng được vì lớp lệch: luôn dự báo NOT_UP đã đạt 76,2% trên TEST mà không có
giá trị nào. F1_UP tập trung vào lớp UP là lớp mình cần bắt. AUC là lựa chọn hợp lý vì độc lập
với ngưỡng, nhưng bài toán cuối cùng vẫn phải ra một quyết định nhị phân, nên em chọn metric gắn
với ngưỡng thực dụng. Nếu bổ sung, em sẽ báo cáo cả AUC và PR-AUC."

## A2. Nhóm câu hỏi về giao thức

HỎI: "Purge là gì và vì sao cần?"
ĐÁP: "Một dòng ngày 28 tháng 6 có nhãn nhận tại phiên 5 tháng 7. Nếu em chỉ cắt theo ngày giao
dịch tại mốc 30 tháng 6, dòng đó vào TRAIN nhưng đáp án của nó nằm trong khoảng thời gian của
VALIDATION, tức mô hình đã học một phần tương lai. Purge nghĩa là loại các dòng có nhãn vắt qua
ranh giới. Trong pipeline, TRAIN lọc theo điều kiện label_end_date nhỏ hơn hoặc bằng mốc TRAIN,
chứ không lọc theo trading_date. Số dòng bị hy sinh là 1.883 ở biên TRAIN/VALIDATION và 1.775 ở
biên VALIDATION/TEST."

HỎI: "Khóa TEST bằng fingerprint hoạt động thế nào?"
ĐÁP: "Sau khi chấm TEST, pipeline ghi hash nội dung của snapshot vào experiments/evaluation_registry.json.
Lần chạy sau, hàm _guard_unevaluated_snapshot kiểm hash đó; nếu snapshot đã từng được đánh giá,
pipeline raise RuntimeError và dừng. Có hai fingerprint khác nhau: tuning_fingerprint chỉ tính
trên phạm vi TRAIN, còn fingerprint khóa TEST tính trên toàn snapshot. Như vậy làm mới dữ liệu để
suy luận không mở lại TEST."

HỎI: "Vì sao cross-validation cắt theo ngày mà không theo dòng?"
ĐÁP: "Dữ liệu là panel, mỗi phiên có gần 400 dòng của 400 mã. Nếu cắt theo chỉ số dòng thì cùng
một phiên bị xẻ đôi: vài mã của phiên đó vào train, vài mã còn lại vào validation. Mô hình sẽ học
được đặc trưng chung của đúng phiên đó rồi được chấm trên chính phiên đó. Trong code,
TimeSeriesSplit được chạy trên danh sách ngày duy nhất của sàn, rồi ánh xạ ngược lại thành chỉ số
dòng, nên cả phiên luôn đi cùng nhau."

HỎI: "Ngưỡng 0,49 lấy từ đâu, có nhìn TEST không?"
ĐÁP: "Không. Ngưỡng lấy từ xác suất out-of-fold trên TRAIN. Mỗi dòng TRAIN được chấm bởi mô hình
của fold không chứa nó, gộp lại thành một mảng OOF, rồi quét ngưỡng từ 0,05 đến 0,95 bước 0,01
chọn ngưỡng cho F1_UP cao nhất, với hai ràng buộc: precision không thấp hơn tỷ lệ UP thật của OOF
là 0,3692, và tỷ lệ dự báo UP không vượt 50%. Ràng buộc thứ hai chặn mô hình suy biến thành
always-UP để ăn điểm F1."

## A3. Nhóm câu hỏi về nhãn và feature

HỎI: "Vì sao không dùng shift(-5) theo từng mã?"
ĐÁP: "Vì mã nghỉ giao dịch vài phiên sẽ khiến shift(-5) nhảy xa hơn 5 phiên thị trường thật. Khi
đó mỗi mã có mốc t+5 khác nhau, các nhãn không so sánh được với nhau, và việc cắt fold theo ngày
cũng mất ý nghĩa. Code dùng lịch phiên chung của toàn sàn, shift(-5) trên chính danh sách ngày đó,
rồi self-join lấy giá đóng cửa tại đúng phiên đích của từng mã. Mã nào không có giao dịch đúng
phiên đó thì bỏ dòng, cố tình không lấy giá gần nhất thay thế vì đó là nhãn sai kỳ hạn. Số dòng
bị loại vì lý do này là 4.468."

HỎI: "Làm sao chắc chắn không có feature nhìn thấy tương lai?"
ĐÁP: "Ba lớp. Một, mọi phép rolling chạy trong groupby theo symbol nên không trộn mã. Hai, danh
sách 20 feature khai báo cố định trong config.settings.FEATURE_COLUMNS và là contract dùng chung
giữa feature engineering, model artifact và tầng dự báo. Ba, có test trong tests/test_data_protocol.py
khẳng định bốn cột future_close_5d, future_return_5d, target và label_end_date không nằm trong
tập feature."

HỎI: "Feature month có phải rò rỉ không?"
ĐÁP: "Không phải rò rỉ theo nghĩa nhìn thấy tương lai, vì tháng của phiên hiện tại đã biết tại
thời điểm dự báo. Nhưng em đồng ý đây là feature đáng nghi về mặt tổng quát hóa: nó có thể chỉ mã
hóa chế độ thị trường trong giai đoạn huấn luyện. Nó đứng thứ hai về importance với 9,1%, nên đây
là hạng mục em sẽ kiểm tra bằng cách chạy lại không có month để so sánh."

## A4. Nhóm câu hỏi về hệ thống

HỎI: "Chatbot có thể bịa số liệu không?"
ĐÁP: "Kiến trúc chặn việc đó từ cấu trúc chứ không chỉ bằng hậu kiểm. LLM chỉ làm hai việc: call
thứ nhất chọn một trong năm action và trích arguments, call thứ hai diễn đạt lại câu trả lời từ
đúng JSON số liệu backend đã tính. Prediction, Điểm UP, ranking và metric đều do backend lấy từ
artifact đã publish; call compose không nhận history, không nhận CSV, không nhận source code nên
không có nguồn nào để bịa. Nếu compose lỗi, quá deadline hoặc trả sai định dạng thì hệ thống giữ
nguyên bản formatter deterministic. Nhánh chào hỏi và từ chối còn bị bỏ đói dữ liệu: phần data gửi
cho compose chỉ có kind hoặc reason cùng danh sách năng lực, không có mã và không có con số nào."

HỎI: "Vì sao không dùng RAG hay vector database?"
ĐÁP: "Vì dữ liệu cần đọc là số có cấu trúc trong JSON và CSV, không phải văn bản dài. Với
loại dữ liệu này, tìm kiếm ngữ nghĩa vừa kém chính xác hơn vừa có nguy cơ lấy sai con số. Phạm vi
nghiệp vụ cũng đóng: năm action đã phủ hết các nhóm câu hỏi, và mỗi action đọc thẳng đúng trường
trong artifact nên kết quả xác định và truy nguồn được. Nếu sau này thêm tài liệu văn bản như báo
cáo phân tích thì vector store mới có lý do tồn tại."

HỎI: "Vậy chatbot định tuyến câu hỏi bằng gì?"
ĐÁP: "Bằng đúng một LLM call decision, không có bảng keyword nào. LLM trả về JSON hai khóa là
action và arguments; backend validate exact schema, chuẩn hóa mã, giới hạn top_n, rồi một dispatcher
cố định gọi handler tương ứng. Em cố ý không giữ router keyword làm dự phòng, vì hai đường định
tuyến song song sẽ khiến hành vi khó giải thích và khó kiểm thử; decision sai protocol thì trả 502
luôn. Cái đánh đổi là chatbot phụ thuộc provider, nên em bù bằng scenario evaluator để đo chất
lượng decision."

HỎI: "Web có bảo mật không?"
ĐÁP: "Chưa có xác thực người dùng, em nêu rõ trong phần hạn chế. Ứng dụng hiện chạy cục bộ cho
mục đích học tập. Riêng các thao tác nguy hiểm như chạy pipeline chính thức thì có khóa liên tiến
trình bằng pipeline.lock và ba điều kiện chặn, nhưng đó là chống chạy trùng chứ không phải phân
quyền. Trước khi triển khai thật thì xác thực là việc phải làm đầu tiên."

## A5. Câu hỏi khó nhất - về policy_id

HỎI: "Trong metadata thấy policy_id là legacy_pre_validation_baseline_gate, còn config ghi
rolling_recent_cv_oof_threshold. Vì sao khác nhau?"
ĐÁP: "Đúng, và em xin giải thích. Snapshot đang publish được tạo dưới luật cũ, trong đó việc mô
hình không vượt baseline trên VALIDATION chỉ tạo cảnh báo và vẫn cho publish. Sau đó em siết luật
trong hàm select_final_model: bây giờ nếu ứng viên tốt nhất không vượt baseline VALIDATION thì
pipeline raise RuntimeError và không publish. Em giữ nguyên tên policy cũ trong artifact để không
gán nhãn sai cho hiện vật đã công bố, vì nó thực sự được tạo dưới luật cũ. Đây là lý do tồn tại
trường policy_id: mỗi hiện vật tự khai nó được sinh dưới luật nào."

---

# PHỤ LỤC B - Các con số phải nhớ

Dữ liệu: 400 mã, 554.897 dòng thô, 233 dòng nến sai OHLC, 4 mã bị loại, 396 mã còn lại,
510.862 dòng có nhãn hợp lệ, 20 feature.
Nhãn: horizon 5 phiên, ngưỡng 1%, UP 37,6% (192.111 dòng), NOT_UP 62,4% (318.751 dòng).
Split: TRAIN 419.807 dòng đến 03/07/2025, VALIDATION 67.047 dòng đến 03/04/2026,
TEST 20.350 dòng từ 13/04/2026 đến 13/07/2026. Purge 1.883 và 1.775.
CV: 4 fold từ 2021-01-01, gap 5 phiên, ngưỡng 0,49.
CV F1_UP: LR 0,4573 (std 0,0420), RF 0,4703 (std 0,0250), GB 0,4714 (std 0,0237).
VALIDATION F1_UP: LR 0,4350, RF 0,4773, GB 0,4734, always-UP 0,4982.
TEST: RF F1_UP 0,3754, precision 0,2890, recall 0,5356, accuracy 0,5766; always-UP 0,3839;
always-NOT_UP accuracy 0,7625.
Confusion TEST: TP 2.589, FP 6.371, FN 2.245, TN 9.145; tỷ lệ UP trên TEST 23,8%.
Feature top: volatility_20d 19,7%, month 9,1%, volatility_5d 7,0%.
Hệ thống: 6 trang web, chatbot Action-Decision với 5 action và 2 LLM call mỗi lượt (decision +
compose), bộ kiểm thử tự động, 350 run tuning (LR 269, RF 51, GB 30).
Siêu tham số RF: n_estimators 130, max_depth 8, min_samples_leaf 100, max_features 0,2,
class_weight balanced_subsample, random_state 42.

---

# PHỤ LỤC C - Giải thích để bạn hiểu, không đọc lên

## C1. Vì sao đề tài này khó ăn điểm bằng F1

F1_UP của baseline always-UP luôn bằng 2p/(1+p) với p là tỷ lệ UP của tập đó. Trên TEST p = 0,2375
nên baseline = 0,3839. Trên VALIDATION p = 0,3318 nên baseline = 0,4982. Nghĩa là baseline này
không cố định mà phụ thuộc phân phối tập đánh giá. Muốn thắng nó, mô hình phải có precision đủ cao
trên lớp UP; với dữ liệu giá thuần thì đó là yêu cầu rất khắt khe. Đây là lý do kết quả âm của bạn
không phải lỗi lập trình mà là giới hạn thông tin của tập feature.

## C2. Ba nguồn rò rỉ mà project chặn

Rò rỉ nhãn: nhãn t+5 tính theo dòng riêng từng mã. Chặn bằng lịch phiên chung và self-join theo
đúng phiên đích.
Rò rỉ thời gian: fold CV trộn cùng một phiên vào cả train và validation, hoặc dòng train có nhãn
đóng sau khi validation bắt đầu. Chặn bằng cắt fold theo ngày, gap 5 phiên, và điều kiện
label_ends nhỏ hơn validation_start.
Rò rỉ chọn mô hình: chấm TEST nhiều lần rồi chọn số đẹp nhất. Chặn bằng chọn mô hình trên
VALIDATION, refit, chấm TEST một lần, ghi fingerprint vào registry.

## C3. Luồng dữ liệu tóm tắt

fetch_hose_data.py tải OHLCV qua vnstock vào CSV chia sẻ ngoài repo.
preprocessing.clean_data loại nến sai OHLC và lọc mã dưới 250 phiên.
feature_engineering.build_features sinh 20 feature, rolling theo từng mã.
feature_engineering.create_labels gán nhãn t+5 theo lịch phiên chung.
protocol_dates.resolve_protocol_dates suy ba mốc theo cửa sổ rolling từ phiên mới nhất.
feature_engineering.protocol_time_split cắt ba tập kèm purge.
model_tuning.tune_models đọc manual_config.json, kiểm ba fingerprint, chạy CV, chốt ngưỡng OOF.
model_evaluation.evaluate_validation_candidates và select_final_model chọn mô hình trên VALIDATION.
model_evaluation.refit_and_evaluate_final_model refit trên TRAIN+VALIDATION rồi chấm TEST một lần.
write_reports ghi CSV/JSON, prediction_service và app.py chỉ đọc lại hiện vật.

## C4. Điều cần tránh khi trả lời

Không hứa con số chưa có trong repo. Nếu bị hỏi số mà bạn không có, nói thẳng là chưa chạy thử
nghiệm đó và nêu cách chạy.
Không nói mô hình dùng được để đầu tư. Luôn kèm câu chỉ dùng cho học tập.
Không nhầm ba loại số: CV trên TRAIN, chọn mô hình trên VALIDATION, kết quả cuối trên TEST.
Nếu hội đồng hỏi con số nào, xác nhận lại đang hỏi tập nào trước khi trả lời.
