import os # Thêm dòng này ở đầu file
from flask import Flask, render_template, request, redirect, url_for, flash
import gspread
import json
import barcode
from barcode.writer import ImageWriter
from io import BytesIO
import base64
from datetime import datetime
import pandas as pd

app = Flask(__name__)
app.secret_key = "vnm_secret_key"  # Cần thiết để dùng flash message (thông báo)

#123
# --- GIỮ NGUYÊN LOGIC KẾT NỐI DB CỦA KEN ---
def connect_db(sheet_name):
    try:
        # Nếu đang chạy trên Render (có biến môi trường)
        if "GOOGLE_SHEETS_JSON" in os.environ:
            creds_data = json.loads(os.environ.get("GOOGLE_SHEETS_JSON"))
            gc = gspread.service_account_from_dict(creds_data)
        else:
            # Nếu chạy trên máy Ken thì vẫn dùng file cũ
            gc = gspread.service_account(filename='credentials.json')

        sh = gc.open("KHO_DATA_2026")
        return sh.worksheet(sheet_name)
    except Exception as e:
        print(f"Lỗi: {e}")
        return None


# --- TRANG CHỦ (DASHBOARD) ---
@app.route('/')
def index():
    ws = connect_db("Inventory")
    ws_po = connect_db("Production")  # Giả sử ní có sheet này

    # Dữ liệu mặc định nếu chưa load được
    records = []
    po_records = []
    total_stock = 0
    sku_count = 0
    pending_po = 0

    if ws:
        data = ws.get_all_records()
        df = pd.DataFrame(data)

        if not df.empty:
            # 1. Xử lý hiển thị bảng Nhật ký
            records = data[-10:]
            records.reverse()

            # 2. Tính toán Metric
            df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
            total_stock = int(df['Qty'].sum())

            # 3. Đếm số lượng SKU (SỬA ĐOẠN NÀY)
            if 'FullCode' in df.columns:
                # ===> DÒNG NÀY QUAN TRỌNG: Ép kiểu sang chuỗi để tránh lỗi 'int is not iterable'
                df['FullCode'] = df['FullCode'].astype(str)

                # Sau khi ép sang chuỗi rồi mới tách
                df['SKU'] = df['FullCode'].apply(lambda x: x.split('|')[0] if '|' in x else x)
                sku_count = df['SKU'].nunique()

    if ws_po:
        po_data = ws_po.get_all_records()
        po_records = po_data
        # Đếm PO nào chưa xong
        df_po = pd.DataFrame(po_data)
        if not df_po.empty and 'Status' in df_po.columns:
            pending_po = len(df_po[df_po['Status'] != 'Done'])

    # Gửi hết dữ liệu ra ngoài HTML
    return render_template('index.html',
                           records=records,
                           po_records=po_records,
                           total_stock=total_stock,
                           sku_count=sku_count,
                           pending_po=pending_po)


@app.route('/nhap-kho', methods=['GET', 'POST'])
def nhap_kho():
    barcode_base64 = None
    data_preview = {}  # Tạo dictionary để chứa thông tin preview

    if request.method == 'POST':
        sku = request.form.get('sku')
        qty = request.form.get('qty')
        batch = request.form.get('batch')

        # ... (đoạn code ws.append_row giữ nguyên) ...

        # Tạo barcode
        full_code = f"{sku}|{batch}"
        rv = BytesIO()
        barcode.get_barcode_class('code128')(full_code, writer=ImageWriter()).write(rv)
        barcode_base64 = base64.b64encode(rv.getvalue()).decode('utf-8')

        # Gửi dữ liệu ra ngoài template để hiện ở cột phải
        return render_template('nhap_kho.html',
                               barcode_img=barcode_base64,
                               last_sku=sku,
                               last_qty=qty,
                               last_batch=batch)

    return render_template('nhap_kho.html')


@app.route('/xuat-kho', methods=['GET', 'POST'])
def xuat_kho():
    if request.method == 'POST':
        raw_scan = request.form.get('barcode_scan')
        user_name = "Ken (Technical)"

        if "|" in raw_scan:
            ws = connect_db("Inventory")
            if ws:
                # Ghi nhận hành động EXPORT với số lượng âm để trừ kho
                ws.append_row([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    user_name, raw_scan, "EXPORT", "", "", "Cổng Xuất", -1
                ])
                flash(f"✅ Đã xuất kho thành công mã: {raw_scan}", "success")
            else:
                flash("❌ Lỗi kết nối Database!", "danger")
        else:
            flash("⚠️ Barcode không hợp lệ! Cần định dạng SKU|Batch", "warning")

    return render_template('xuat_kho.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)