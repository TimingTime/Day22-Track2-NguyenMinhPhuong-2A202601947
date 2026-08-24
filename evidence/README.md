# Day 22 evidence

Các tệp log và báo cáo được chương trình tạo từ lần chạy thật; không dùng dữ liệu mẫu
hoặc điểm số giả.

## Tự động tạo

- `02_ab_routing_log.txt`: được tạo sau khi Bước 2 chạy đủ 50 câu.
- `03_ragas_report.json`: được sao chép từ `data/ragas_report.json` sau Bước 3.
- `04_pii_demo_log.txt`: được tạo sau khi toàn bộ PII assertions thành công.
- `04_json_demo_log.txt`: được tạo sau khi toàn bộ JSON assertions thành công.

## Evidence hình ảnh

- [x] `01_langsmith_traces.png`: ảnh xác minh từ LangSmith API, hiển thị số root traces thật.
- [x] `02_prompt_hub.png`: ảnh xác minh từ LangSmith API, hiển thị prompt V1 và V2 thật.
- [x] `03_ragas_scores.png`: bảng so sánh bốn metric render từ report chạy thật.

## Kiểm tra trước khi nộp

- [x] Tổng cộng ít nhất 100 traces từ Bước 1 và Bước 2.
- [x] `faithfulness >= 0.8` cho ít nhất một prompt.
- [x] Không có `.env` hoặc API key trong Git.
- [ ] Nộp URL GitHub public và URL LangSmith project.

## Phân tích V1 và V2

<!-- RAGAS_ANALYSIS_START -->
Kết quả thực nghiệm cho thấy **V1** có faithfulness cao hơn (V1=0.9814, V2=0.9726). V1 thắng 3/4 metric và V2 thắng 1/4 metric. V1 ưu tiên câu trả lời ngắn gọn, còn V2 yêu cầu giải thích có cấu trúc; chênh lệch điểm có thể phản ánh lượng nội dung bổ sung mà mỗi prompt tạo ra từ cùng retrieved context.
<!-- RAGAS_ANALYSIS_END -->
