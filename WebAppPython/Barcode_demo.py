import streamlit as st
import pandas as pd
import barcode
from barcode.writer import ImageWriter
from io import BytesIO
import zipfile
import random
from fpdf import FPDF
from pyzbar.pyzbar import decode
import cv2
import numpy as np
import streamlit_authenticator as stauth
import gspread
import json
from datetime import datetime, date, timedelta
import altair as alt  # Thư viện vẽ biểu đồ đẹp

# --- 1. CẤU HÌNH HỆ THỐNG ---
st.set_page_config(page_title="Vinamilk Inventory System", layout="wide", page_icon="🥛")


# --- 2. KẾT NỐI DATABASE (GOOGLE SHEET) ---
def connect_db(sheet_name):
    try:
        if "gcp_service_account" in st.secrets:
            creds = dict(st.secrets["gcp_service_account"])
            if "json_content" in creds: creds = json.loads(creds["json_content"])
            gc = gspread.service_account_from_dict(creds)
            sh = gc.open("KHO_DATA_2026")  # <--- TÊN FILE CỦA ÔNG
            try:
                ws = sh.worksheet(sheet_name)
            except:
                ws = sh.add_worksheet(title=sheet_name, rows=1000, cols=10)
            return ws
    except:
        return None


# --- 3. CẤU HÌNH USER ---
config_user = {
    'credentials': {
        'usernames': {
            'manager': {'name': 'Quản Lý Kho (Admin)',
                        'password': '$2b$12$MWFqC9gNSU93.GfxSUSqnOn4duvXwOrW2WX6Kq6QkL2f6ZgYypkY.'},  # 123456
            'staff': {'name': 'Nhân Viên Vận Hành',
                      'password': '$2b$12$ZCxqkVJBagfsWJBoFntXSedewNTSBYbcKJHYbXdVP0k4jErvVYRVq'}  # admin123
        }
    },
    'cookie': {'expiry_days': 1, 'key': 'vina_key', 'name': 'vina_cookie'}
}

authenticator = stauth.Authenticate(config_user['credentials'], config_user['cookie']['name'],
                                    config_user['cookie']['key'], config_user['cookie']['expiry_days'])
authenticator.login()

# --- 4. LOGIC CHÍNH ---
if st.session_state["authentication_status"] is True:

    user_name = st.session_state["name"]
    user_role = st.session_state["username"]  # manager / staff

    # SIDEBAR
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2554/2554045.png", width=80)
        st.title("KHO VẬN THÔNG MINH")
        st.write(f"Hello, **{user_name}**")
        st.info(f"Vai trò: {user_role.upper()}")
        authenticator.logout('Đăng xuất', 'sidebar')
        st.divider()
        st.caption("Version: 4.0 (Vinamilk Standard)")


    # --- HÀM HỖ TRỢ ---
    def create_barcode(code):
        rv = BytesIO();
        barcode.get_barcode_class('code128')(code, writer=ImageWriter()).write(rv,
                                                                               {"module_height": 8.0, "font_size": 6});
        return rv


    def decode_img(img_bytes):
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        decoded = decode(img)
        res = []
        if decoded:
            for obj in decoded:
                txt = obj.data.decode("utf-8")
                res.append(txt)
                cv2.rectangle(img, (obj.rect.left, obj.rect.top),
                              (obj.rect.left + obj.rect.width, obj.rect.top + obj.rect.height), (0, 255, 0), 3)
                cv2.putText(img, txt, (obj.rect.left, obj.rect.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return img, res


    # --- GIAO DIỆN CHÍNH ---
    st.header(f"🥛 HỆ THỐNG QUẢN LÝ KHO ({datetime.now().strftime('%d/%m/%Y')})")

    # TAB ĐIỀU KHIỂN
    tabs = ["📊 Dashboard (Báo Cáo)", "📥 Nhập Kho (Inbound)", "📤 Xuất Kho (Outbound)"]
    if user_role == 'staff': tabs = ["📥 Nhập Kho (Inbound)", "📤 Xuất Kho (Outbound)"]  # Nhân viên ko xem báo cáo

    current_tab = st.radio("Chọn chức năng:", tabs, horizontal=True, label_visibility="collapsed")
    st.divider()

    # === MODULE 1: NHẬP KHO (TẠO MÃ & GHI DATA) ===
    if "Nhập Kho" in current_tab:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("1. Thông tin Lô Hàng")
            sku = st.selectbox("Sản phẩm:", ["VNM-SUATUOI-1L", "VNM-SUACHUA-ALOE", "VNM-ONGTHO-RED"])
            batch = st.text_input("Số Lô (Batch):", f"LOT-{random.randint(1000, 9999)}")
            nsx = st.date_input("Ngày SX:", date.today())
            hsd = st.date_input("Hạn SD:", date.today() + timedelta(days=180))  # Mặc định 6 tháng
            loc = st.selectbox("Vị trí kho:", ["Kho Lạnh A", "Kho Mát B", "Kệ Pallet C1"])

            # Tự động tạo mã Barcode chứa thông tin Lô
            full_code = f"{sku}|{batch}"
            st.info(f"Mã định danh: {full_code}")

            if st.button("🖨️ Tạo & Nhập Kho", type="primary"):
                ws = connect_db("Inventory")
                if ws:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Ghi vào Google Sheet
                    ws.append_row([now, user_name, full_code, "IMPORT", str(nsx), str(hsd), loc, 100])
                    st.toast("Đã nhập kho thành công!", icon="✅")
                    st.session_state['last_barcode'] = full_code
                else:
                    st.error("Lỗi kết nối Server!")

        with c2:
            st.subheader("2. Tem Mã Vạch")
            if 'last_barcode' in st.session_state:
                img = create_barcode(st.session_state['last_barcode'])
                st.image(img, caption="Tem dán thùng (Chuẩn GS1-128 Simulation)", width=400)
                st.success(f"HSD: {hsd.strftime('%d/%m/%Y')} | Kho: {loc}")

    # === MODULE 2: XUẤT KHO & KIỂM TRA (SCANNER) ===
    elif "Xuất Kho" in current_tab:
        st.subheader("🔍 Quét kiểm tra & Xuất hàng")
        mode = st.radio("Input:", ["Webcam Live", "Upload Ảnh"], horizontal=True)
        img_in = st.camera_input("Quét mã") if mode == "Webcam Live" else st.file_uploader("Tải ảnh")

        if img_in:
            p_img, codes = decode_img(img_in.getvalue())
            col_L, col_R = st.columns(2)
            with col_L:
                st.image(p_img, caption="Camera Feed")

            with col_R:
                if codes:
                    for code in codes:
                        st.markdown(f"### 📦 Phát hiện: `{code}`")

                        # LOGIC KIỂM TRA HẠN SỬ DỤNG (Mock Data demo)
                        # Thực tế sẽ query từ Google Sheet về để check
                        if "LOT" in code:
                            parts = code.split("|")
                            sku_code = parts[0]
                            st.success(f"✅ Mã hợp lệ: {sku_code}")

                            # Giả lập check HSD (Demo logic)
                            # Nếu muốn xịn, phải query ws.get_all_values() để tìm dòng có mã này
                            st.warning("⚠️ Lưu ý: Kiểm tra kỹ HSD trên bao bì trước khi xuất!")

                            ws = connect_db("Inventory")
                            if ws:
                                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                ws.append_row([now, user_name, code, "EXPORT", "", "", "Cổng Xuất 1", -1])
                                st.toast(f"Đã xuất kho: {code}")
                else:
                    st.error("Không tìm thấy mã vạch!")

    # === MODULE 3: DASHBOARD (CHỈ MANAGER THẤY) ===
    elif "Dashboard" in current_tab:
        st.subheader("📈 Báo Cáo Tồn Kho & Hạn Sử Dụng")

        ws = connect_db("Inventory")
        if ws:
            data = ws.get_all_records()
            if len(data) > 0:
                df = pd.DataFrame(data)

                # Metric tổng quan
                m1, m2, m3 = st.columns(3)
                m1.metric("Tổng Lượt Nhập", len(df[df['Action'] == 'IMPORT']))
                m2.metric("Tổng Lượt Xuất", len(df[df['Action'] == 'EXPORT']))
                m3.metric("Cảnh Báo Hết Hạn", "2 Lô", delta="-1 Lô", delta_color="inverse")

                st.divider()

                # Biểu đồ 1: Hoạt động theo nhân viên
                chart = alt.Chart(df).mark_bar().encode(
                    x='User',
                    y='count()',
                    color='Action'
                ).properties(title="Hiệu suất nhân viên")
                st.altair_chart(chart, use_container_width=True)

                # Bảng dữ liệu chi tiết
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Chưa có dữ liệu. Hãy nhập kho vài đơn hàng!")

elif st.session_state["authentication_status"] is False:
    st.error('Sai mật khẩu!')
elif st.session_state["authentication_status"] is None:
    st.warning('Vui lòng đăng nhập hệ thống.')