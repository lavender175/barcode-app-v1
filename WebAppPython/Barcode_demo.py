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
import streamlit_authenticator as stauth
import gspread  # Thư viện Google Sheet
import json
from datetime import datetime

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Hệ Thống Kho Vận - KenAdmin", layout="wide", page_icon="🔒")

#123
# --- 2. CẤU HÌNH KẾT NỐI GOOGLE SHEET ---
def connect_to_gsheet():
    try:
        # Lấy thông tin từ Secrets
        if "gcp_service_account" in st.secrets:
            # Cách 1: Nếu cấu hình dạng TOML chuẩn
            creds_dict = dict(st.secrets["gcp_service_account"])

            # Cách 2: Nếu cấu hình dạng JSON string (Mẹo nhanh)
            if "json_content" in creds_dict:
                creds_dict = json.loads(creds_dict["json_content"])

            gc = gspread.service_account_from_dict(creds_dict)

            # --- QUAN TRỌNG: THAY TÊN FILE GOOGLE SHEET CỦA ÔNG VÀO ĐÂY ---
            sh = gc.open("KHO_DATA_2026")  # <--- TÊN FILE TRÊN GOOGLE DRIVE

            # Chọn sheet đầu tiên hoặc sheet tên 'Logs'
            try:
                worksheet = sh.worksheet("Logs")
            except:
                # Nếu chưa có thì tạo mới
                worksheet = sh.add_worksheet(title="Logs", rows=1000, cols=5)
                worksheet.append_row(["Timestamp", "User", "Barcode", "Type", "Action"])

            return worksheet
        else:
            return None
    except Exception as e:
        st.error(f"Lỗi kết nối Google Sheet: {e}")
        return None


# --- 3. CẤU HÌNH USER ---
config_user = {
    'credentials': {
        'usernames': {
            'kenadmin': {
                'name': 'Ken (Admin)',
                'password': '$2b$12$fhhd6mGI7DbdB8YwRhVb3u2rzOSusBOzXm5ZVIw9Ywj4LzN4Y/zsO'
            },
            'kho': {
                'name': 'Nhân Viên Kho',
                'password': '$2b$12$oX5vi/EBJtEyK.D7j7UOMe4o65VmFlFRXdVtdfCfhzz67atZjJ3H2'
            }
        }
    },
    'cookie': {'expiry_days': 30, 'key': 'random_key', 'name': 'auth_cookie'}
}

# --- 4. LOGIN FLOW ---
authenticator = stauth.Authenticate(
    config_user['credentials'],
    config_user['cookie']['name'],
    config_user['cookie']['key'],
    config_user['cookie']['expiry_days']
)

authenticator.login()

if st.session_state["authentication_status"] is False:
    st.error('❌ Sai mật khẩu!')
elif st.session_state["authentication_status"] is None:
    st.warning('🔒 Vui lòng đăng nhập.')
elif st.session_state["authentication_status"] is True:

    # Lấy thông tin user hiện tại
    user_real_name = st.session_state["name"]
    username_id = st.session_state["username"]  # 'kenadmin' hoặc 'kho'

    with st.sidebar:
        st.write(f"User: **{user_real_name}**")
        authenticator.logout('Đăng xuất', 'sidebar')
        st.divider()

        # Kiểm tra kết nối Database
        if st.button("Kiểm tra kết nối Sheet"):
            ws = connect_to_gsheet()
            if ws: st.success("✅ Đã kết nối Google Sheet!")

    # --- MAIN APP ---
    st.title(f"🏭 KHO VẬN THÔNG MINH ({user_real_name})")


    # --- CÁC HÀM LOGIC (Giữ nguyên) ---
    def create_barcode_image(code_text, code_type='code128'):
        try:
            rv = BytesIO()
            BARCODE_CLASS = barcode.get_barcode_class(code_type)
            options = {"module_width": 0.3, "module_height": 10.0, "font_size": 8, "quiet_zone": 1.0}
            my_barcode = BARCODE_CLASS(code_text, writer=ImageWriter())
            my_barcode.write(rv, options=options)
            return rv
        except:
            return None


    def process_and_decode(image_bytes):
        cv_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        decoded_objects = decode(cv_image)
        results = []
        if decoded_objects:
            for obj in decoded_objects:
                data = obj.data.decode("utf-8")
                results.append((data, obj.type))
                pts = np.array(obj.polygon, np.int32).reshape((-1, 1, 2))
                cv2.polylines(cv_image, [pts], True, (0, 255, 0), 3)
                cv2.putText(cv_image, data, (pts[0][0][0], pts[0][0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)
        return cv_image, results


    # --- PHÂN QUYỀN GIAO DIỆN (UI) ---

    # Nếu là ADMIN: Thấy hết 3 tab
    if username_id == 'kenadmin':
        tab1, tab2, tab3 = st.tabs(["🖨️ Tạo Đơn Lẻ", "🏭 Tạo Hàng Loạt", "📷 Quét Kho"])

        with tab1:
            st.info("Chức năng dành riêng cho Admin tạo mã.")
            code = st.text_input("Mã:", "VN-123")
            if st.button("Tạo mã"):
                img = create_barcode_image(code)
                st.image(img)

        with tab2:
            st.info("Module xử lý Batch (Đã ẩn chi tiết cho gọn code demo).")

    # Nếu là KHO: Chỉ thấy 1 tab Quét (Nhưng Admin cũng thấy tab này ở vị trí số 3)
    else:
        st.info("👋 Chào nhân viên kho! Hãy bắt đầu ca làm việc.")
        tab3 = st.container()  # Chỉ hiện container này

    # --- NỘI DUNG TAB 3 (SCANNER) - Dùng chung cho cả 2 ---
    # Lưu ý: Với Admin thì nó nằm trong tab3, với User Kho thì nó nằm ngay ngoài
    with tab3:
        st.subheader("📡 TRẠM QUÉT MÃ (LIVE DATA)")

        scan_mode = st.radio("Chế độ:", ["Webcam", "Upload Ảnh"], horizontal=True)
        img_file = st.camera_input("Chụp ảnh") if scan_mode == "Webcam" else st.file_uploader("Tải ảnh")

        if img_file:
            processed_img, data = process_and_decode(img_file.getvalue())
            col1, col2 = st.columns(2)
            with col1:
                st.image(processed_img, caption="Kết quả xử lý")
            with col2:
                if data:
                    st.success(f"✅ Phát hiện {len(data)} mã!")

                    # LOGIC LƯU VÀO GOOGLE SHEET
                    ws = connect_to_gsheet()
                    for code, btype in data:
                        st.code(f"{code} ({btype})")

                        if ws:
                            # Ghi log: Thời gian - User - Mã - Loại - Hành động
                            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            ws.append_row([now, user_real_name, code, btype, "SCAN_IN"])
                            st.toast(f"💾 Đã lưu {code} vào Google Sheet!", icon="☁️")
                        else:
                            st.warning("Chưa kết nối Database!")
                else:
                    st.error("Không tìm thấy mã nào.")