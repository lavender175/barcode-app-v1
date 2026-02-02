import streamlit as st
import pandas as pd
import barcode
from barcode.writer import ImageWriter
from io import BytesIO
import random
from fpdf import FPDF
from pyzbar.pyzbar import decode
import cv2
import numpy as np
import streamlit_authenticator as stauth
import gspread
import json
from datetime import datetime, date, timedelta
import altair as alt
import unicodedata
import tempfile

# --- 1. CẤU HÌNH HỆ THỐNG & HÀM BỔ TRỢ ---
st.set_page_config(page_title="WMS Pro - Vinamilk Standard", layout="wide", page_icon="🏭")


def remove_accents(input_str):
    if not input_str: return ""
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


# --- GIẢ LẬP DỮ LIỆU LỆNH SẢN XUẤT (PO) - THEO YÊU CẦU JD ---
# Thực tế sẽ lấy từ ERP hoặc Sheet "Production_Orders"
MOCK_DB_PO = {
    "PO-2026-001": {
        "Product": "Sữa Tươi 100% (Lô Sáng)",
        "BOM": {"VNM-SUATUOI-RAW": 100, "VNM-DUONG-TINH-LUYEN": 5}  # SKU: Số lượng cần
    },
    "PO-2026-002": {
        "Product": "Sữa Chua Nha Đam",
        "BOM": {"VNM-MEN-PROBI": 2, "VNM-NHADAM-CUBES": 20}
    }
}


# --- 2. KẾT NỐI DATABASE ---
def connect_db(sheet_name):
    try:
        if "gcp_service_account" in st.secrets:
            creds = dict(st.secrets["gcp_service_account"])
            if "json_content" in creds: creds = json.loads(creds["json_content"])
            gc = gspread.service_account_from_dict(creds)
            sh = gc.open("KHO_DATA_2026")
            try:
                ws = sh.worksheet(sheet_name)
            except:
                ws = sh.add_worksheet(title=sheet_name, rows=1000, cols=10)
            return ws
    except Exception as e:
        st.error(f"Lỗi kết nối GSheets: {e}")
        return None
    return None


def check_duplicate_batch(sku, batch):
    """Kiểm tra trùng lặp để đảm bảo tính duy nhất của Batch"""
    ws = connect_db("Inventory")
    if ws:
        try:
            # Lấy toàn bộ cột C (giả sử cột chứa Mã Full) để check nhanh
            all_codes = ws.col_values(3)
            full_code = f"{sku}|{batch}"
            return full_code in all_codes
        except:
            return False
    return False


# --- 3. CẤU HÌNH USER ---
config_user = {
    'credentials': {
        'usernames': {
            'manager': {'name': 'Quản Lý Kho',
                        'password': '$2b$12$MWFqC9gNSU93.GfxSUSqnOn4duvXwOrW2WX6Kq6QkL2f6ZgYypkY.'},
            'staff': {'name': 'Nhân Viên', 'password': '$2b$12$ZCxqkVJBagfsWJBoFntXSedewNTSBYbcKJHYbXdVP0k4jErvVYRVq'}
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
    user_role = st.session_state["username"]

    with st.sidebar:
        st.title("🏭 WMS PRO")
        st.write(f"User: **{user_name}**")
        authenticator.logout('Đăng xuất', 'sidebar')
        st.divider()
        st.markdown("### 📌 Menu")


    # --- HÀM XỬ LÝ ẢNH & BARCODE ---
    def create_barcode(code):
        try:
            rv = BytesIO()
            BARCODE_CLASS = barcode.get_barcode_class('code128')
            options = {
                "module_width": 0.5, "module_height": 18.0,
                "font_size": 10, "text_distance": 4.0,
                "quiet_zone": 6.5, "write_text": True
            }
            my_barcode = BARCODE_CLASS(code, writer=ImageWriter())
            my_barcode.write(rv, options=options)
            return rv
        except:
            return None


    def decode_img(img_bytes):
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        decoded_objects = decode(img)
        if not decoded_objects:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            decoded_objects = decode(gray)
            if not decoded_objects:
                _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                decoded_objects = decode(thresh)
        results = []
        if decoded_objects:
            for obj in decoded_objects:
                txt = obj.data.decode("utf-8")
                results.append(txt)
                x, y, w, h = obj.rect
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img, txt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return img, results


    # --- GIAO DIỆN CHÍNH ---
    st.title("HỆ THỐNG QUẢN LÝ KHO & SẢN XUẤT")
    tabs = ["📥 Nhập Kho (Inbound)", "🏭 Xuất Kho (Outbound)", "📊 Dashboard"]
    current_tab = st.radio("Chọn nghiệp vụ:", tabs, horizontal=True, label_visibility="collapsed")
    st.divider()

    # ================= MODULE 1: NHẬP KHO =================
    if "Nhập Kho" in current_tab:
        c1, c2 = st.columns([1, 1.5])
        with c1:
            st.subheader("1. Nhập Liệu")
            sku = st.selectbox("SKU/Nguyên Liệu:",
                               ["VNM-SUATUOI-RAW", "VNM-DUONG-TINH-LUYEN", "VNM-MEN-PROBI", "VNM-NHADAM-CUBES"])
            qty = st.number_input("Số lượng (Kg/Unit):", min_value=1, value=100)
            batch = st.text_input("Số Batch:", f"LOT-{random.randint(1000, 9999)}")
            nsx = st.date_input("NSX:", date.today())
            hsd = st.date_input("HSD:", date.today() + timedelta(days=30))
            loc = st.selectbox("Vị trí:", ["Kho A (Lạnh)", "Kho B (Thường)", "Kho C (Hóa Chất)"])

            full_code = f"{sku}|{batch}"
            st.info(f"Mã định danh: {full_code}")

            if st.button("💾 Lưu Kho", type="primary"):
                if check_duplicate_batch(sku, batch):
                    st.error("❌ Lỗi: Batch này đã tồn tại!")
                else:
                    ws = connect_db("Inventory")
                    if ws:
                        try:
                            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            ws.append_row([now, user_name, full_code, "IMPORT", str(nsx), str(hsd), loc, qty])
                            st.toast(f"Đã nhập {qty} {sku}", icon="✅")
                            st.session_state['last_import'] = {'code': full_code, 'qty': qty, 'batch': batch,
                                                               'hsd': str(hsd), 'sku': sku}
                        except Exception as e:
                            st.error(f"Lỗi GSheets: {e}")

        with c2:
            st.subheader("2. In Tem (Zebra Ready)")
            if 'last_import' in st.session_state:
                info = st.session_state['last_import']
                img = create_barcode(info['code'])
                if img: st.image(img, width=300)

                b1, b2 = st.columns(2)
                with b1:
                    if st.button("📦 Tem Thùng (Lẻ)"):
                        try:
                            pdf = FPDF(orientation='L', unit='mm', format=(100, 150))
                            pdf.add_page();
                            pdf.set_font("Helvetica", 'B', 16)
                            pdf.cell(0, 10, txt=remove_accents("TEM NGUYEN LIEU"), ln=True, align='C')
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                img.seek(0);
                                tmp.write(img.getvalue())
                                pdf.image(tmp.name, x=10, y=20, w=130)
                            pdf_data = bytes(pdf.output())
                            st.download_button("⬇️ Tải PDF", pdf_data, f"Pallet_{info['batch']}.pdf")
                        except Exception as e:
                            st.error(str(e))

                with b2:
                    if st.button(f"🏷️ Tem Loạt ({info['qty']})"):
                        try:  # --- KHỐI TRY/EXCEPT ĐÃ ĐƯỢC FIX ---
                            with st.spinner("Đang render..."):
                                pdf_bulk = FPDF(orientation='P', unit='mm', format='A4')
                                pdf_bulk.set_auto_page_break(auto=False, margin=0)
                                pdf_bulk.add_page()
                                mx, my, cw, rh = 12, 12, 62, 40
                                cols, rows = 3, 7
                                x, y, cx, cy = mx, my, 0, 0

                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_b:
                                    img.seek(0);
                                    tmp_b.write(img.getvalue())
                                    t_path = tmp_b.name

                                for i in range(int(info['qty'])):
                                    pdf_bulk.image(t_path, x=x + 2, y=y + 5, w=cw - 4)  # Không vẽ Rect
                                    pdf_bulk.set_font("Helvetica", size=8)
                                    pdf_bulk.set_xy(x, y + rh - 8)
                                    pdf_bulk.cell(cw, 5, txt=remove_accents(f"{info['sku']} | {info['batch']}"),
                                                  align='C')

                                    cx += 1
                                    if cx < cols:
                                        x += cw
                                    else:
                                        cx = 0;
                                        x = mx;
                                        cy += 1;
                                        y += rh
                                        if cy >= rows: pdf_bulk.add_page(); cy = 0; y = my; x = mx

                                bulk_bytes = bytes(pdf_bulk.output())
                                st.download_button("⬇️ Tải A4 Bulk", bulk_bytes, f"Bulk_{info['batch']}.pdf")
                        except Exception as e:
                            st.error(f"Lỗi in loạt: {e}")
                        finally:
                            pass

    # ================= MODULE 2: XUẤT KHO (NÂNG CẤP PO) =================
    elif "Xuất Kho" in current_tab:
        mode = st.radio("Chế độ xuất:", ["🚀 Xuất Lẻ (Thông thường)", "🏭 Xuất Cho Sản Xuất (Theo PO)"], horizontal=True)
        st.divider()

        # --- MODE A: XUẤT SẢN XUẤT (NEW FEATURE) ---
        if "Theo PO" in mode:
            c_po, c_scan = st.columns([1, 2])
            with c_po:
                po_sel = st.selectbox("Chọn Lệnh SX:", list(MOCK_DB_PO.keys()))
                po_data = MOCK_DB_PO[po_sel]
                st.info(f"SP: {po_data['Product']}")
                st.write("**Công thức (BOM):**")
                st.dataframe(pd.DataFrame(list(po_data['BOM'].items()), columns=['SKU', 'Cần (Kg)']), hide_index=True)

            with c_scan:
                st.write("👇 **QUÉT MÃ NGUYÊN LIỆU ĐỐI CHIẾU:**")
                scan_in = st.text_input("Scanner Input:", key="po_scan", placeholder="Click vào đây và bắn súng...")

                if scan_in:
                    s_sku = scan_in.split("|")[0] if "|" in scan_in else scan_in
                    s_batch = scan_in.split("|")[1] if "|" in scan_in else "N/A"

                    # VALIDATION LOGIC
                    if s_sku in po_data['BOM']:
                        st.success(f"✅ ĐÚNG NGUYÊN LIỆU: {s_sku}")
                        st.caption(f"Batch: {s_batch}")

                        confirm_qty = st.number_input(f"Số lượng xuất thực tế ({s_sku}):", value=po_data['BOM'][s_sku])
                        if st.button("Xác nhận xuất PO"):
                            ws = connect_db("Inventory")
                            if ws:
                                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                ws.append_row(
                                    [now, user_name, scan_in, "EXPORT_PO", "", "", f"To: {po_sel}", -confirm_qty])
                                st.toast("Đã xuất kho thành công!", icon="🏭")
                    else:
                        st.error(f"⛔ SAI NGUYÊN LIỆU! '{s_sku}' KHÔNG CÓ TRONG LỆNH {po_sel}")

        # --- MODE B: XUẤT LẺ (CŨ) ---
        else:
            st.write("📱 **Quét mã vạch (Camera/Súng):**")
            scan_method = st.radio("Input:", ["Súng Quét", "Camera"], horizontal=True)
            final_code = None

            if "Súng" in scan_method:
                final_code = st.text_input("Nhập/Quét mã:", key="manual_scan")
            else:
                img_file = st.camera_input("Chụp mã")
                if img_file:
                    _, codes = decode_img(img_file.getvalue())
                    if codes: final_code = codes[0]

            if final_code:
                st.success(f"Đã quét: {final_code}")
                if st.button("Xác nhận xuất lẻ"):
                    ws = connect_db("Inventory")
                    if ws:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ws.append_row([now, user_name, final_code, "EXPORT", "", "", "Retail", -1])
                        st.toast("Đã xuất kho!", icon="🚛")

    # ================= MODULE 3: DASHBOARD =================
    elif "Dashboard" in current_tab:
        st.subheader("Báo Cáo Tồn Kho")
        ws = connect_db("Inventory")
        if ws:
            df = pd.DataFrame(ws.get_all_records())
            if not df.empty:
                m1, m2 = st.columns(2)
                m1.metric("Tổng Phiếu Nhập", len(df[df['Action'] == 'IMPORT']))
                m2.metric("Tổng Phiếu Xuất", len(df[df['Action'].str.contains('EXPORT')]))
                st.dataframe(df, use_container_width=True)
            else:
                st.warning("Chưa có dữ liệu!")

elif st.session_state["authentication_status"] is False:
    st.error('Sai mật khẩu!')
elif st.session_state["authentication_status"] is None:
    st.warning('Vui lòng đăng nhập.')