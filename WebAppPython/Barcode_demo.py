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
import unicodedata


def remove_accents(input_str):
    if not input_str: return ""
    # Chuyển đổi Tiếng Việt có dấu -> Không dấu (để in PDF không bị lỗi font)
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])
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

    # --- HÀM XỬ LÝ ẢNH NÂNG CAO (SMART DECODE) ---
    def decode_img(img_bytes):
        # 1. Đọc ảnh từ bytes sang format OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Chiến thuật 1: Đọc ngay ảnh gốc
        decoded_objects = decode(img)

        # 3. Chiến thuật 2: Nếu thất bại, chuyển sang Đen Trắng (Grayscale)
        if not decoded_objects:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            decoded_objects = decode(gray)

            # 4. Chiến thuật 3: Nếu vẫn thất bại, dùng Threshold (Nhị phân hóa)
            # Giúp làm rõ các vạch đen trên nền trắng, loại bỏ nhiễu màu
            if not decoded_objects:
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                decoded_objects = decode(thresh)

        # 5. Vẽ kết quả lên ảnh (nếu tìm thấy)
        results = []
        if decoded_objects:
            for obj in decoded_objects:
                txt = obj.data.decode("utf-8")
                results.append(txt)

                # Vẽ khung xanh lá
                points = obj.polygon
                if len(points) == 4:
                    pts = np.array(points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], True, (0, 255, 0), 3)
                else:
                    # Trường hợp khung hình chữ nhật đơn giản
                    x, y, w, h = obj.rect
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

                # Vẽ chữ lên ảnh
                x, y, w, h = obj.rect
                cv2.putText(img, txt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img, results


    # --- GIAO DIỆN CHÍNH ---
    st.header(f"🥛 HỆ THỐNG QUẢN LÝ KHO ({datetime.now().strftime('%d/%m/%Y')})")

    # TAB ĐIỀU KHIỂN
    tabs = ["📊 Dashboard (Báo Cáo)", "📥 Nhập Kho (Inbound)", "📤 Xuất Kho (Outbound)"]
    if user_role == 'staff': tabs = ["📥 Nhập Kho (Inbound)", "📤 Xuất Kho (Outbound)"]  # Nhân viên ko xem báo cáo

    current_tab = st.radio("Chọn chức năng:", tabs, horizontal=True, label_visibility="collapsed")
    st.divider()


    # === MODULE 1: NHẬP KHO (INBOUND) & IN TEM ===
    if "Nhập Kho" in current_tab:
        st.subheader("📥 Nhập Kho & In Tem")

        c1, c2 = st.columns([1, 1.5])

        with c1:
            st.markdown("#### 1. Thông tin Lô Hàng")
            sku = st.selectbox("Sản phẩm:", ["VNM-SUATUOI-1L", "VNM-SUACHUA-ALOE", "VNM-ONGTHO-RED"])
            qty = st.number_input("Số lượng nhập (Qty):", min_value=1, value=100, step=10)
            batch = st.text_input("Số Lô (Batch):", f"LOT-{random.randint(1000, 9999)}")
            nsx = st.date_input("Ngày SX:", date.today())
            hsd = st.date_input("Hạn SD:", date.today() + timedelta(days=180))
            loc = st.selectbox("Vị trí lưu kho:", ["Kho Lạnh A", "Kho Mát B", "Kệ Pallet C1"])

            full_code = f"{sku}|{batch}"
            st.info(f"🆔 Mã lô: {full_code}")

            if st.button("💾 Lưu Phiếu Nhập Kho", type="primary"):
                ws = connect_db("Inventory")
                if ws:
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ws.append_row([now, user_name, full_code, "IMPORT", str(nsx), str(hsd), loc, qty])
                    st.toast(f"Đã nhập {qty} sản phẩm!", icon="✅")
                    st.session_state['last_import'] = {'code': full_code, 'qty': qty, 'batch': batch, 'hsd': str(hsd),
                                                       'sku': sku}
                else:
                    st.error("Lỗi kết nối Google Sheet!")

        with c2:
            st.markdown("#### 2. Tùy chọn In Tem")

            if 'last_import' in st.session_state:
                info = st.session_state['last_import']
                st.success(f"✅ Đã nhập lô: {info['batch']}")

                img = create_barcode(info['code'])
                st.image(img, caption=f"Mã: {info['code']}", width=350)
                st.divider()

                col_print1, col_print2 = st.columns(2)

                # --- FIX LỖI IN 1 TEM ---
                with col_print1:
                    if st.button("📦 In 1 Tem Thùng"):
                        pdf = FPDF(orientation='L', unit='mm', format=(100, 150))
                        pdf.add_page()
                        pdf.set_font("Helvetica", 'B', 20)  # Dùng font Helvetica chuẩn

                        # Dùng hàm remove_accents để tránh lỗi font
                        title = remove_accents("TEM LUU KHO")
                        pdf.cell(0, 20, txt=title, ln=True, align='C')

                        import tempfile

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            img.seek(0)
                            tmp.write(img.getvalue())
                            pdf.image(tmp.name, x=10, y=30, w=130)

                        pdf.set_xy(10, 80)
                        pdf.set_font("Helvetica", size=12)

                        # Chuẩn bị nội dung text (bỏ dấu tiếng Việt)
                        content = (
                            f"SP: {remove_accents(info['sku'])}\n"
                            f"Lo: {info['batch']}\n"
                            f"SL: {info['qty']}\n"
                            f"HSD: {info['hsd']}"
                        )
                        pdf.multi_cell(0, 10, txt=content)

                        # SỬA LỖI OUTPUT: Không dùng .encode('latin-1') nữa
                        try:
                            pdf_data = pdf.output(dest='S').encode('latin-1')  # Cho bản cũ
                        except:
                            pdf_data = pdf.output()  # Cho bản mới (bytearray)

                        st.download_button("⬇️ Tải Tem (PDF)", pdf_data, f"Pallet_{info['batch']}.pdf",
                                           "application/pdf")

                # --- FIX LỖI IN NHIỀU TEM ---
                with col_print2:
                    if st.button(f"🏷️ In {info['qty']} Tem Lẻ"):
                        with st.spinner("Đang tạo file PDF..."):
                            pdf_bulk = FPDF(orientation='P', unit='mm', format='A4')
                            pdf_bulk.set_auto_page_break(auto=False, margin=0)
                            pdf_bulk.add_page()

                            margin_x, margin_y = 10, 10
                            col_width, row_height = 65, 35
                            cols, rows = 3, 8
                            x, y = margin_x, margin_y
                            count_x, count_y = 0

                            import tempfile

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_bulk:
                                img.seek(0)
                                tmp_bulk.write(img.getvalue())
                                tmp_path = tmp_bulk.name

                            for i in range(int(info['qty'])):
                                pdf_bulk.rect(x, y, col_width, row_height)
                                pdf_bulk.image(tmp_path, x=x + 2, y=y + 2, w=col_width - 4, h=row_height - 10)
                                pdf_bulk.set_font("Helvetica", size=7)
                                pdf_bulk.set_xy(x, y + row_height - 6)

                                # Text dưới barcode (Bỏ dấu)
                                txt_lbl = remove_accents(f"{info['sku']} | Exp: {info['hsd']}")
                                pdf_bulk.cell(col_width, 5, txt=txt_lbl, align='C')

                                count_x += 1
                                if count_x < cols:
                                    x += col_width
                                else:
                                    count_x = 0;
                                    x = margin_x
                                    count_y += 1;
                                    y += row_height
                                    if count_y >= rows:
                                        pdf_bulk.add_page();
                                        count_y = 0;
                                        y = margin_y;
                                        x = margin_x

                            # SỬA LỖI OUTPUT
                            try:
                                bulk_data = pdf_bulk.output(dest='S').encode('latin-1')
                            except:
                                bulk_data = pdf_bulk.output()

                            st.download_button("⬇️ Tải A4 (PDF)", bulk_data, f"Bulk_{info['batch']}.pdf",
                                               "application/pdf")

    # === MODULE 2: XUẤT KHO & KIỂM TRA (SCANNER) ===
    elif "Xuất Kho" in current_tab:
        st.subheader("📤 Xuất Kho & Kiểm Tra")

        # Tạo nút chuyển đổi chế độ nhập liệu
        input_method = st.radio(
            "Chọn thiết bị nhập liệu:",
            ["🔫 Súng Quét (PC/Kho)", "📱 Camera Điện Thoại (Mobile)"],
            horizontal=True
        )

        st.divider()

        final_code = None

        # --- MODE A: DÙNG SÚNG QUÉT (PC) ---
        if "Súng Quét" in input_method:
            c1, c2 = st.columns([2, 1])
            with c1:
                st.info("💡 Đặt trỏ chuột vào ô bên dưới và bắn mã.")
                # Ô này sẽ nhận tín hiệu từ máy quét (hoặc paste thủ công để test)
                scan_input = st.text_input("INPUT:", placeholder="Đang chờ tín hiệu...", key="scanner_in")
                if scan_input:
                    final_code = scan_input
            with c2:
                st.image("https://cdn-icons-png.flaticon.com/512/2830/2830305.png", caption="Hardware Scanner Mode",
                         width=100)

        # --- MODE B: DÙNG CAMERA (MOBILE) ---
        else:
            st.warning("💡 Lưu ý: Giữ camera ổn định, đủ ánh sáng.")
            # Camera Input của Streamlit chạy rất mượt trên Mobile
            img_file = st.camera_input("Chụp ảnh mã vạch")

            if img_file:
                # Gọi hàm xử lý ảnh "Vua Lì Đòn"
                p_img, codes = decode_img(img_file.getvalue())

                if codes:
                    final_code = codes[0]  # Lấy mã đầu tiên
                    st.success("✅ Đã đọc được mã!")
                else:
                    st.error("❌ Ảnh mờ hoặc không có mã. Hãy thử lại!")
                    st.image(p_img, caption="Ảnh vừa chụp (Không đọc được)", width=300)

        # --- XỬ LÝ KẾT QUẢ CHUNG (CHO CẢ 2 CHẾ ĐỘ) ---
        if final_code:
            st.divider()
            st.markdown(f"### 📦 MÃ ĐÃ QUÉT: `{final_code}`")

            # Logic phân tích mã
            sku = final_code
            batch = "N/A"

            if "|" in final_code:
                sku, batch = final_code.split("|")

            # Hiển thị thẻ thông tin đẹp
            m1, m2 = st.columns(2)
            m1.metric("Sản phẩm (SKU)", sku)
            m2.metric("Lô hàng (Batch)", batch, delta="Đang xuất kho", delta_color="inverse")

            # Ghi vào Database
            ws = connect_db("Inventory")
            if ws:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Ghi Log
                ws.append_row([now, user_name, final_code, "EXPORT", "", "", "Mobile/Scanner", -1])
                st.toast(f"Đã xuất kho: {sku}", icon="🚛")

                # Hiệu ứng thành công
                if "Súng Quét" in input_method:
                    st.balloons()  # PC thì thả bóng
                else:
                    st.snow()  # Mobile thì thả tuyết (cho nhẹ máy)

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