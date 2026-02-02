import streamlit as st
import pandas as pd
import barcode
from barcode.writer import ImageWriter
from io import BytesIO
import zipfile
import random
import string
from fpdf import FPDF
from pyzbar.pyzbar import decode
import cv2
import numpy as np


# --- CẤU HÌNH & KHỞI ĐỘNG ---
st.set_page_config(page_title="Ken Automation - Barcode Master", layout="wide")




# --- CÁC HÀM XỬ LÝ LOGIC (CORE) ---

# 1. Hàm tạo mã SKU an toàn (Không chứa ký tự dễ nhầm)
def generate_sku(length=8):
    safe_chars = "ABCDEFGHJKMNPRSTUVWXY" + "3456789"
    return ''.join(random.choices(safe_chars, k=length))


# 2. Hàm tạo dữ liệu giả lập
def generate_demo_data(quantity):
    data = []
    for i in range(quantity):
        data.append({
            "Product_ID": f"PROD-{generate_sku(6)}",
            "Batch_No": f"LOT-{random.randint(202400, 202499)}",
            "Quantity": random.randint(10, 100)
        })
    return pd.DataFrame(data)


# 3. Hàm tạo ảnh Barcode (Lưu vào RAM)
def create_barcode_image(code_text, code_type='code128'):
    try:
        rv = BytesIO()
        BARCODE_CLASS = barcode.get_barcode_class(code_type)
        options = {
            "module_width": 0.3,
            "module_height": 10.0,
            "font_size": 8,
            "text_distance": 3.0,
            "quiet_zone": 1.0
        }
        my_barcode = BARCODE_CLASS(code_text, writer=ImageWriter())
        my_barcode.write(rv, options=options)
        return rv
    except Exception:
        return None


# 4. Hàm tạo file PDF A4 (Layout 3x8)
def create_pdf_a4(dataframe, target_col):
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=False, margin=0)
    pdf.add_page()

    margin_x = 10
    margin_y = 10
    col_width = 65
    row_height = 35
    cols_per_page = 3
    rows_per_page = 8

    x, y = margin_x, margin_y
    col_counter = 0
    row_counter = 0

    for index, row in dataframe.iterrows():
        code_val = str(row[target_col])
        img_buffer = create_barcode_image(code_val)

        if img_buffer:
            # Vẽ khung (Optional)
            pdf.set_line_width(0.1)
            pdf.rect(x, y, col_width, row_height)

            # Chèn ảnh
            pdf.image(img_buffer, x=x + 2, y=y + 2, w=col_width - 4, h=row_height - 10)

            # Ghi thông tin text
            pdf.set_font("Arial", size=8)
            pdf.set_xy(x, y + row_height - 8)
            info_text = f"Batch: {row.get('Batch_No', 'N/A')} | Qty: {row.get('Quantity', '0')}"
            pdf.cell(col_width, 5, txt=info_text, align='C')

            # Tính tọa độ kế tiếp
            col_counter += 1
            if col_counter < cols_per_page:
                x += col_width
            else:
                col_counter = 0
                x = margin_x
                row_counter += 1
                y += row_height
                if row_counter >= rows_per_page:
                    pdf.add_page()
                    row_counter = 0
                    y = margin_y
                    x = margin_x

    pdf_buffer = BytesIO()
    pdf_output = pdf.output()
    pdf_buffer.write(pdf_output)
    return pdf_buffer


# 5. Hàm Xử lý ảnh & Decode Barcode (Dùng chung cho cả Webcam và Upload)
def process_and_decode(image_bytes):
    # Convert bytes -> OpenCV Image
    cv_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Decode
    decoded_objects = decode(cv_image)
    results = []

    if decoded_objects:
        for obj in decoded_objects:
            barcode_data = obj.data.decode("utf-8")
            barcode_type = obj.type
            results.append((barcode_data, barcode_type))

            # Vẽ khung
            points = obj.polygon
            if len(points) == 4:
                pts = np.array(points, np.int32).reshape((-1, 1, 2))
                cv2.polylines(cv_image, [pts], True, (0, 255, 0), 3)
                # Vẽ text lên ảnh luôn để dễ nhìn
                cv2.putText(cv_image, barcode_data, (pts[0][0][0], pts[0][0][1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return cv_image, results
# --- GIAO DIỆN NGƯỜI DÙNG (UI) ---
st.title("🏭 AUTOMATION BARCODE CENTER PRO")
st.caption("Developed by Ken | Tech Stack: Python, Streamlit, OpenCV, Ngrok")

tab1, tab2, tab3 = st.tabs(["🖨️ Tạo Đơn (Manual)", "🏭 Tạo Hàng Loạt (Batch)", "📷 Quét Kiểm Tra (Scanner)"])

# === TAB 1: MANUAL ===
with tab1:
    col1, col2 = st.columns([1, 2])
    with col1:
        input_code = st.text_input("Nhập mã sản phẩm:", "VINA-MILK-001")
        if st.button("Generate Preview"):
            st.session_state['preview_img'] = create_barcode_image(input_code)
    with col2:
        if 'preview_img' in st.session_state:
            st.image(st.session_state['preview_img'], width=300)

# === TAB 2: BATCH PROCESSING ===
with tab2:
    st.subheader("Xử lý dữ liệu lớn & Đóng gói")

    # Khu vực Data
    col_demo1, col_demo2 = st.columns([1, 3])
    with col_demo1:
        qty_demo = st.number_input("Số lượng mã cần tạo:", value=24, step=24)
        if st.button("Tạo dữ liệu mẫu"):
            st.session_state['batch_df'] = generate_demo_data(qty_demo)

    with col_demo2:
        if 'batch_df' in st.session_state:
            st.dataframe(st.session_state['batch_df'], height=150, use_container_width=True)

    st.divider()

    # Khu vực Xuất File
    if 'batch_df' in st.session_state:
        target_col = st.selectbox("Chọn cột làm Barcode:", st.session_state['batch_df'].columns)

        c1, c2 = st.columns(2)

        # Nút 1: Xuất ZIP ảnh rời
        with c1:
            if st.button("📦 Xuất ảnh rời (.ZIP)"):
                with st.spinner("Đang nén file..."):
                    zip_buffer = BytesIO()
                    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                        for idx, row in st.session_state['batch_df'].iterrows():
                            code = str(row[target_col])
                            img = create_barcode_image(code)
                            if img:
                                zip_file.writestr(f"{code}.png", img.getvalue())

                    st.success("Đã nén xong!")
                    st.download_button("⬇️ Tải ZIP", zip_buffer.getvalue(), "barcodes.zip", "application/zip")

        # Nút 2: Xuất PDF A4
        with c2:
            if st.button("📄 Xuất file in A4 (.PDF)"):
                with st.spinner("Đang dàn trang PDF..."):
                    pdf_data = create_pdf_a4(st.session_state['batch_df'], target_col)
                    st.success("Dàn trang hoàn tất!")
                    st.download_button("⬇️ Tải PDF", pdf_data.getvalue(), "layout_a4.pdf", "application/pdf")

# === TAB 3: SCANNER ĐA NĂNG ===
with tab3:
    st.subheader("Trạm kiểm soát Barcode")

    # Chọn chế độ nhập liệu
    scan_mode = st.radio("Chọn phương thức quét:",
                         ["📸 Live Webcam (Nhanh)", "📂 Upload File / Camera Gốc (Nét hơn)"],
                         horizontal=True)

    input_image = None

    if scan_mode == "📸 Live Webcam (Nhanh)":
        st.caption("Dùng webcam mặc định của trình duyệt.")
        input_image = st.camera_input("Bấm nút để chụp")

    else:
        st.caption(
            "💡 Trên điện thoại/iPad: Chọn mục này -> Bấm 'Browse files' -> Chọn 'Take Photo' để dùng Camera gốc (có zoom, flash, đổi cam trước sau).")
        input_image = st.file_uploader("Tải ảnh lên hoặc Chụp mới", type=['png', 'jpg', 'jpeg'])

    # Xử lý khi có ảnh đầu vào
    if input_image is not None:
        # Gọi hàm xử lý chung
        processed_img, decoded_info = process_and_decode(input_image.getvalue())

        # Chia cột hiển thị kết quả
        col_res1, col_res2 = st.columns([1, 1])

        with col_res1:
            st.image(processed_img, channels="BGR", caption="Ảnh đã xử lý", use_container_width=True)

        with col_res2:
            if decoded_info:
                st.success(f"✅ ĐÃ TÌM THẤY {len(decoded_info)} MÃ!")
                for code, b_type in decoded_info:
                    st.info(f"📦 Code: **{code}**\n\n🏷️ Loại: {b_type}")
            else:
                st.warning("⚠️ Không tìm thấy Barcode nào trong ảnh này.")
                st.markdown("""
                **Gợi ý nếu không quét được:**
                - Ảnh bị mờ hoặc rung -> *Dùng chế độ 'Upload/Camera Gốc' để lấy nét tốt hơn.*
                - Barcode quá nhỏ -> *Zoom ảnh lại gần.*
                - Thiếu sáng -> *Bật đèn flash.*
                """)