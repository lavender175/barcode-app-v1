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
def check_duplicate_batch(sku, batch):
    """
    Kiểm tra xem SKU và Batch đã tồn tại trong sheet Inventory chưa.
    Trả về True nếu đã tồn tại, False nếu chưa.
    """
    ws = connect_db("Inventory")
    if ws:
        # Lấy cột SKU|Batch (Giả sử là cột C - index 2)
        records = ws.get_all_records()
        if not records:
            return False

        full_code = f"{sku}|{batch}"
        # Kiểm tra nhanh trong danh sách hiện tại
        exists = any(item.get('FullCode') == full_code for item in records)
        return exists
    return False

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
        try:
            rv = BytesIO()
            BARCODE_CLASS = barcode.get_barcode_class('code128')

            # Cấu hình tối ưu cho máy in văn phòng + Máy quét Zebra
            options = {
                "module_width": 0.5,  # Độ dày vạch (0.5 là "điểm ngọt" cho giấy A4)
                "module_height": 18.0,  # Tăng nhẹ chiều cao để quét nhanh hơn
                "font_size": 10,
                "text_distance": 4.0,
                "quiet_zone": 6.5,  # Tăng vùng trắng hai đầu để Zebra dễ định vị
                "write_text": True
            }

            my_barcode = BARCODE_CLASS(code, writer=ImageWriter())
            my_barcode.write(rv, options=options)
            return rv
        except Exception:
            return None

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
    # === CẬP NHẬT MODULE 1: NHẬP KHO & IN TEM (BẢN VÁ LỖI STREAMLIT) ===
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
            st.info(f"🆔 Mã lô dự kiến: {full_code}")

            if st.button("💾 Lưu Phiếu Nhập Kho", type="primary"):
                # BƯỚC 1: KIỂM TRA TRÙNG
                if check_duplicate_batch(sku, batch):
                    st.error(f"❌ Lô hàng '{batch}' của sản phẩm này đã có trong hệ thống!")
                else:
                    ws = connect_db("Inventory")
                    if ws:
                        try:
                            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            ws.append_row([now, user_name, full_code, "IMPORT", str(nsx), str(hsd), loc, qty])
                            st.toast(f"Đã nhập {qty} sản phẩm thành công!", icon="✅")
                            st.session_state['last_import'] = {
                                'code': full_code, 'qty': qty, 'batch': batch,
                                'hsd': str(hsd), 'sku': sku
                            }
                        except Exception as e:
                            st.error(f"Lỗi khi ghi vào Google Sheets: {e}")
                    else:
                        st.error("Không thể kết nối Database!")

        with c2:
            st.markdown("#### 2. Tùy chọn In Tem")
            if 'last_import' in st.session_state:
                info = st.session_state['last_import']
                st.success(f"✅ Sẵn sàng in tem cho lô: {info['batch']}")

                # Tạo barcode preview
                img = create_barcode(info['code'])
                if img:
                    st.image(img, caption=f"Mã QR/Barcode: {info['code']}", width=300)

                st.divider()
                cp1, cp2 = st.columns(2)

                with cp1:
                    if st.button("📦 In 1 Tem Thùng"):
                        try:
                            pdf = FPDF(orientation='L', unit='mm', format=(100, 150))
                            pdf.add_page()
                            pdf.set_font("Helvetica", 'B', 16)
                            pdf.cell(0, 10, txt=remove_accents("PHIEU LUU KHO (PALLET)"), ln=True, align='C')

                            import tempfile

                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                img.seek(0)
                                tmp.write(img.getvalue())
                                pdf.image(tmp.name, x=10, y=20, w=130)

                            pdf.set_xy(10, 75)
                            pdf.set_font("Helvetica", size=11)
                            content = f"SKU: {info['sku']}\nLot: {info['batch']}\nQty: {info['qty']}\nExp: {info['hsd']}"
                            pdf.multi_cell(0, 8, txt=remove_accents(content))

                            pdf_data = bytes(pdf.output())
                            st.download_button("⬇️ Tải Tem Thùng", pdf_data, f"Pallet_{info['batch']}.pdf")
                        except Exception as e:
                            st.error(f"Lỗi in tem thùng: {e}")

                with cp2:
                    if st.button(f"🏷️ In {info['qty']} Tem Lẻ"):
                        try:  # BẮT ĐẦU KHỐI TRY ĐỂ SỬA LỖI TRONG ẢNH CỦA ÔNG
                            with st.spinner("Đang tính toán layout A4..."):
                                pdf_bulk = FPDF(orientation='P', unit='mm', format='A4')
                                pdf_bulk.set_auto_page_break(auto=False, margin=0)
                                pdf_bulk.add_page()

                                # Layout tối ưu cho Zebra: 3 cột x 7 hàng
                                mx, my = 12, 12
                                cw, rh = 62, 40
                                cols, rows = 3, 7
                                x, y = mx, my
                                cx, cy = 0, 0

                                import tempfile

                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_b:
                                    img.seek(0)
                                    tmp_b.write(img.getvalue())
                                    t_path = tmp_b.name

                                for i in range(int(info['qty'])):
                                    # KHÔNG DÙNG pdf_bulk.rect ĐỂ BỎ KHUNG
                                    pdf_bulk.image(t_path, x=x + 2, y=y + 5, w=cw - 4)

                                    pdf_bulk.set_font("Helvetica", size=7)
                                    pdf_bulk.set_xy(x, y + rh - 8)
                                    label = f"{info['sku']} | Exp: {info['hsd']}"
                                    pdf_bulk.cell(cw, 5, txt=remove_accents(label), align='C')

                                    cx += 1
                                    if cx < cols:
                                        x += cw
                                    else:
                                        cx = 0;
                                        x = mx;
                                        cy += 1;
                                        y += rh
                                        if cy >= rows:
                                            pdf_bulk.add_page();
                                            cy = 0;
                                            y = my;
                                            x = mx

                                bulk_bytes = bytes(pdf_bulk.output())
                                st.download_button("⬇️ Tải A4 PDF", bulk_bytes, f"Bulk_{info['batch']}.pdf")

                        except Exception as e:  # KHỐI EXCEPT BẮT BUỘC PHẢI CÓ
                            st.error(f"Lỗi xử lý PDF hàng loạt: {e}")
                        finally:  # KHỐI FINALLY (Tùy chọn nhưng nên có để code sạch)
                            pass

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