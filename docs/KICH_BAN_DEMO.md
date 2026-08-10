# KỊCH BẢN DEMO — HOSE Stock Prediction

> Vừa thao tác trên giao diện vừa giải thích, không dùng slide. Tổng ~12–15 phút.
>
> **Lưu ý về số liệu:** các con số cụ thể (ngưỡng quyết định, ngày train, % điểm UP, Precision_UP, base rate) **đọc đúng theo màn hình lúc demo**, không đọc thuộc lòng. Chỉ có mấy hằng số bản chất là cố định: tăng hơn **1%** sau **5 phiên**, **3 model**, **20 đặc trưng**.

---

## PHẦN 0 — Chuẩn bị trước khi vào phòng (làm trước ~10 phút, KHÔNG nói khi demo)

1. Mở terminal, chạy:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   python app.py
   ```
2. Mở trình duyệt: **http://127.0.0.1:5000**
3. **Test trước 3 nút gợi ý ở trang `/chat`** — chatbot gọi LLM ngoài, có thể chờ tới 60s nếu mạng chậm. Test trước để chắc nó sống.
4. Để sẵn 2 tab: tab 1 ở Dự báo (`/`), tab 2 ở Tuning Lab (`/tuning`).
5. **TUYỆT ĐỐI KHÔNG bấm 2 nút khi demo:**
   - **"Chạy pipeline chính thức"** → treo cả trang tới khi train xong (nhiều phút).
   - **"Lấy dữ liệu mới"** → chạy 20–25 phút, khóa cả hệ thống.

   Biết chúng tồn tại để nói, KHÔNG bấm.

---

## PHẦN 1 — Mở đầu: bài toán (~1 phút)

**THAO TÁC:** Đứng ở trang chủ `/`. Chỉ vào tiêu đề "Dự báo xu hướng cổ phiếu" và dòng phụ "Tăng hơn 1% trong 5 phiên kế tiếp".

**NÓI:**
> "Project của em giải một bài toán phân loại nhị phân: cho một mã cổ phiếu sàn HOSE, dự đoán sau đúng 5 phiên giao dịch nữa giá đóng cửa có tăng hơn 1% không — gọi là UP, ngược lại là NOT_UP. Đây không phải dự đoán giá cụ thể, mà là dự đoán xu hướng. Toàn bộ chạy offline trên dữ liệu lịch sử, không có giá realtime."

Chỉ vào chip header: "Offline", ngày dữ liệu, số mã, tên model đang phục vụ.
> "Trên đầu trang hệ thống luôn cho biết đang dùng dữ liệu tới ngày nào, bao nhiêu mã, và model nào đang phục vụ."

---

## PHẦN 2 — Demo Dự báo một mã (~2 phút) — *trái tim của demo*

**THAO TÁC:** Gõ `FPT` vào ô "Mã cổ phiếu" → bấm **"Dự báo"**. (Hoặc bấm 1 nút ở khối "Thử nhanh".)

**NÓI (trong lúc chờ spinner "Đang tính..."):**
> "Model nhận vào 20 đặc trưng kỹ thuật tính từ giá và khối lượng — các loại return nhiều khung thời gian, đường trung bình động, RSI, độ biến động, vị trí giá so với đỉnh/đáy 20 phiên."

**THAO TÁC:** Khi kết quả hiện ra, chỉ vào **thanh "Điểm UP"** và **vạch "Ngưỡng"**.

**NÓI:**
> "Đây là điểm quan trọng nhất. Model không trả lời UP/NOT_UP trực tiếp, mà cho ra một *xác suất* — 'Điểm UP' này. Rồi so với một **ngưỡng quyết định**. Điểm vượt ngưỡng thì kết luận UP. Ngưỡng này không phải 0.5 mặc định — em sẽ giải thích cách chọn nó ở phần Tuning Lab."

Chỉ vào 3 chip: Return 20 phiên, Volatility 20 phiên, Volume/AVG20, và biểu đồ đường có vùng "Khoảng dự báo".
> "Bên dưới là căn cứ dữ liệu: ngày tham chiếu, giá đóng cửa, và biểu đồ giá lịch sử kèm khoảng dự báo 5 phiên tới."

Chỉ disclaimer cuối trang.
> "Cuối trang luôn có dòng: kết quả chỉ mang tính tham khảo, không phải khuyến nghị đầu tư."

**Nếu bị hỏi "20 đặc trưng đó là gì / có bị nhìn trước tương lai không?"** → chuyển ý:
> "Về chống nhìn trước tương lai, em xin trình bày ở phần phương pháp — đó là chỗ em đầu tư nhiều nhất."

(Giữ để nói ở Phần 6.)

---

## PHẦN 3 — So sánh hai mã (~1 phút)

**THAO TÁC:** Vào tab "So sánh hai mã" (hoặc menu So sánh). Gõ `FPT` và `VNM` → **"So sánh"**.

**NÓI:**
> "Cùng cơ chế đó áp cho hai mã đặt cạnh nhau — điểm UP, kết quả, return, độ biến động, thanh khoản. Hệ thống tự đánh dấu mã nào tín hiệu UP cao hơn, dao động thấp hơn, giao dịch sôi động hơn."

Nhanh thôi — đây là biến thể của Phần 2.

---

## PHẦN 4 — Screener toàn sàn (~1 phút)

**THAO TÁC:** Menu **Xếp hạng** (`/screener`). Bấm nút lọc **"UP"**. Gõ vài ký tự vào ô tìm. Bấm vào 1 mã bất kỳ.

**NÓI:**
> "Thay vì tra từng mã, trang xếp hạng chấm điểm toàn bộ các mã đủ dữ liệu, sắp UP lên trước theo điểm giảm dần. Lọc, tìm kiếm, và bấm vào mã nào là nhảy thẳng sang trang dự báo mã đó."

Chỉ pill "X UP / Y NOT_UP" ở đầu.
> "Con số này cũng cho thấy hiện tại thị trường nghiêng về phía nào theo model."

---

## PHẦN 5 — Chatbot (~1.5 phút)

**THAO TÁC:** Menu **Trợ lý** (`/chat`). Bấm lần lượt 3 nút gợi ý: **"FPT đang thế nào?"** → **"So sánh FPT với VNM"** → **"Model học từ dữ liệu gì?"**.

**NÓI (câu phòng thủ quan trọng, nói chủ động):**
> "Đây là lớp giao diện hội thoại. Nhưng em nói rõ một điều: **LLM ở đây chỉ làm một việc là hiểu câu hỏi và định tuyến ý định** — nó phân câu hỏi vào một trong 5 nhóm hành động. **Mọi con số, mọi dự đoán đều đến từ model machine learning của em, không phải LLM bịa ra.** Câu trả lời cuối do hệ thống ghép từ kết quả model, LLM không được phép chế số liệu."

**Nếu bị hỏi "khác gì if-else / sao không dùng từ khóa?"** →
> "Phần định tuyến bên trong đúng là cố định. Giá trị của LLM nằm ở chỗ nó hiểu được nhiều cách diễn đạt khác nhau của cùng một ý, và hiểu câu hỏi nối tiếp như 'Còn HPG thì sao?' dựa vào ngữ cảnh trước đó — cái mà so khớp từ khóa thủ công rất khó làm gọn."

---

## PHẦN 6 — Tuning Lab: KHOE PHƯƠNG PHÁP (~3–4 phút) — *phần ăn điểm ML*

**THAO TÁC:** Menu **Tuning Lab** (`/tuning`). Chỉ vào banner fingerprint dataset, 3 card model (LR / RF / GB), biểu đồ "Diễn biến CV F1_UP", và bảng "Lịch sử huấn luyện". Bấm sort 1 cột, mở 1 dòng "Chi tiết fold & OOF".

**NÓI (đây là chỗ dồn lực — nói chậm, rõ):**
> "Đây là nơi thể hiện phần phương pháp em đầu tư nhiều nhất. Em có 3 model ứng viên: Logistic Regression, Random Forest, Gradient Boosting.
>
> Điểm cốt lõi là **cách đánh giá chống rò rỉ dữ liệu tương lai**. Vì đây là dữ liệu chuỗi thời gian, em không được phép trộn ngẫu nhiên. Em dùng **cross-validation cắt theo thời gian**, và có hai lớp bảo vệ:
> - **Lớp một:** chừa một khoảng trống 5 phiên giữa phần train và phần kiểm tra — đúng bằng horizon dự báo — để mẫu cuối phần train không 'nhìn thấy' giá trong vùng kiểm tra.
> - **Lớp hai:** cắt theo *ngày giao dịch của cả sàn*, không cắt theo từng dòng. Vì nếu cắt theo dòng, cùng một phiên sẽ bị xé — vài mã vào train, vài mã vào kiểm tra — gây rò rỉ chéo.
>
> Còn cái ngưỡng quyết định lúc nãy ở trang dự báo: em không lấy 0.5. Em gom xác suất dự báo của tất cả các fold lại thành một tập, rồi mới quét tìm một ngưỡng chung tối ưu F1 — kèm ràng buộc không cho model 'đoán UP hết' để ăn gian điểm."

**THAO TÁC:** Chỉ vào panel "Cấu hình tự chọn & pipeline" và nút "Chạy pipeline chính thức" (KHÔNG bấm).

**NÓI:**
> "Sau khi chọn cấu hình cho từng model, official pipeline sẽ chọn model tốt nhất **trên tập validation**, rồi mới mở tập test **đúng một lần duy nhất** để đo. Test không tham gia vào việc chọn model — đây là kỷ luật chống 'học tủ' trên tập test."

**Nếu bị hỏi về fingerprint / lock / atomic release** →
> "Phần đó là kỹ nghệ phần mềm để đảm bảo tái lập được kết quả và an toàn quy trình — mỗi lần train được gắn vân tay dữ liệu để biết chắc số liệu báo cáo khớp đúng dữ liệu đã train. Em xin phép không đi sâu vì nó là phần hạ tầng, không phải nội dung machine learning."

*(Đóng khung 1 câu rồi chuyển — đừng để đào sâu.)*

---

## PHẦN 7 — Đánh giá & sự trung thực (~2 phút) — *biến hạn chế thành điểm mạnh*

**THAO TÁC:** Menu **Đánh giá** (`/evaluation`). Sẽ thấy **banner vàng** và chỉ có bảng "Kết quả legacy", KHÔNG có biểu đồ TEST.

**NÓI (chủ động giải thích, đừng để bị bắt):**
> "Trang này đang hiện cảnh báo vàng, và em giải thích luôn: artifact model đang phục vụ thuộc một *phiên bản quy trình cũ*, có trước khi em thêm cổng kiểm tra baseline hiện hành. Hệ thống **cố tình từ chối trưng số liệu test của model cũ** dưới giao diện của quy trình mới, để không gây hiểu nhầm. Đây là cơ chế kỷ luật — thà báo 'không so sánh được' còn hơn trưng số sai lệch."

**NÓI (điểm trung thực khoa học — rất quan trọng, nói thẳng):**
> "Và em xin thành thật về kết quả: model của em **chưa đánh bại được baseline 'luôn đoán UP'** một cách rõ rệt. Lý do là bài toán dự báo cổ phiếu ngắn hạn vốn là bài toán tín hiệu rất yếu — thị trường gần với ngẫu nhiên. Nhưng model vẫn nhặt được *một chút* tín hiệu: độ chính xác khi báo UP có nhỉnh hơn tỷ lệ nền. Em không tô hồng con số, và em nghĩ việc nhận ra 'bài toán này khó tới mức nào' cũng là một kết quả có giá trị."

*(Đọc đúng con số Precision_UP và base rate hiện trên màn hình lúc đó — đừng đọc thuộc lòng số sai.)*

---

## PHẦN 8 — Kết (~30 giây)

**THAO TÁC:** Có thể bấm nút 💬 chat dock nổi ở góc phải để cho thấy trợ lý theo suốt mọi trang. Quay về trang chủ.

**NÓI:**
> "Tóm lại: một pipeline hoàn chỉnh từ lấy dữ liệu, tạo đặc trưng, huấn luyện có kiểm soát rò rỉ, tới giao diện dự báo và trợ lý hội thoại. Điểm em tự hào nhất không phải là con số dự báo, mà là **quy trình đánh giá trung thực và chống rò rỉ dữ liệu** — vì với bài toán này, làm đúng phương pháp quan trọng hơn là khoe một con số đẹp. Em xin demo tới đây."

---

## BẢNG TRA NHANH KHI BỊ HỎI KHÓ (in ra để cạnh máy)

| Câu hỏi | Trả lời 1 câu |
|---|---|
| "Model thua đoán bừa thì có giá trị gì?" | Bài toán tín hiệu yếu; Precision_UP vẫn > tỷ lệ nền; nhận ra độ khó là kết quả. |
| "Chatbot khác gì if-else?" | LLM hiểu đa dạng cách hỏi + câu nối tiếp; mọi số liệu từ model, không phải LLM. |
| "Artifact đang chạy có phải code hiện tại?" | Không — là bản legacy trước khi thêm cổng baseline; test đã đóng 1 lần trên snapshot đó. |
| "Fingerprint/lock để làm gì?" | Kỹ nghệ đảm bảo tái lập; không đi sâu, không phải nội dung ML. |
| "Sao `month` để 1–12, sao SMA giá tuyệt đối?" | Thừa nhận là điểm cải thiện được (one-hot/sin-cos, dùng tỷ lệ thay vì giá tuyệt đối). |
| "Có backtest lợi nhuận/phí giao dịch không?" | Chưa — đây là bài phân loại, chưa gắn chiến lược giao dịch; là hướng mở rộng. |

---

## HAI NHẮC NHỞ AN TOÀN CUỐI

1. Nếu lỡ chạm "Lấy dữ liệu mới" hoặc "Chạy pipeline chính thức", hệ thống sẽ khóa/treo — **không bấm 2 nút đó**.
2. Flask dev server: nếu sửa giao diện phút chót thì **phải tắt và chạy lại `python app.py`**, không thì trình duyệt hiện bản cũ.
