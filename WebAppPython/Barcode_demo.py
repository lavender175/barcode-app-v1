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

# --- 1. CẤU HÌNH HỆ THỐNG ---
st.set_page_config(page_title="WMS Demo - Vinamilk", layout="wide", page_icon="🥛")

# --- CSS TÙY CHỈNH CHO MENU NGANG (MOBILE FRIENDLY) ---
# Biến radio button thành dạng thanh điều hướng (Navbar)
st.markdown("""
<style>
    /* 1. Ẩn nút 3 gạch và Sidebar mặc định để rộng chỗ */
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}

    /* 2. Style cho Menu ngang (Radio) thành dạng Thẻ (Pills) */
    div[class*="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {
        font-size: 16px;
        font-weight: 600;
    }
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        justify-content: center; /* Căn giữa menu */
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
        overflow-x: auto; /* Cho phép cuộn ngang nếu màn hình quá bé */
    }
    /* 3. Chỉnh lại Header cho gọn */
    .main-header {
        font-size: 24px !important; 
        font-weight: 700; 
        color: #154360; 
        text-align: center;
        margin-top: -50px; /* Đẩy lên trên cùng */
    }
    .block-container {
        padding-top: 1rem; /* Giảm khoảng trắng đầu trang */
    }
</style>
""", unsafe_allow_html=True)


def remove_accents(input_str):
    if not input_str: return ""
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


# --- DATA GIẢ LẬP ---
MOCK_DB_PO = {
    "PO-2026-001": {"Product": "Sữa Tươi 100% (Lô Sáng)", "BOM": {"VNM-SUATUOI-RAW": 100, "VNM-DUONG-TINH-LUYEN": 5}},
    "PO-2026-002": {"Product": "Sữa Chua Nha Đam", "BOM": {"VNM-MEN-PROBI": 2, "VNM-NHADAM-CUBES": 20}}
}


# --- KẾT NỐI DB ---
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
    except:
        return None
    return None


def check_duplicate_batch(sku, batch):
    ws = connect_db("Inventory")
    if ws:
        try:
            all_codes = ws.col_values(3)
            return f"{sku}|{batch}" in all_codes
        except:
            return False
    return False


def get_available_batches(target_sku):
    ws = connect_db("Inventory")
    if not ws: return []
    df = pd.DataFrame(ws.get_all_records())
    if df.empty: return []

    df['SKU'] = df['FullCode'].apply(lambda x: x.split('|')[0] if '|' in x else x)
    df['Batch'] = df['FullCode'].apply(lambda x: x.split('|')[1] if '|' in x else 'Unknown')

    df_sku = df[df['SKU'] == target_sku].copy()
    df_sku['Qty'] = pd.to_numeric(df_sku['Qty'], errors='coerce').fillna(0)
    df_sku['Real'] = df_sku.apply(lambda x: -x['Qty'] if 'EXPORT' in str(x['Action']) else x['Qty'], axis=1)

    summary = df_sku.groupby('Batch')['Real'].sum().reset_index()
    avail = summary[summary['Real'] > 0]['Batch'].tolist()

    valid = []
    for b in avail:
        row = df[(df['Batch'] == b) & (df['Action'] == 'IMPORT')].head(1)
        if not row.empty: valid.append((b, row.iloc[0]['HSD']))
    valid.sort(key=lambda x: x[1])
    return [f"{b} (HSD: {h})" for b, h in valid]


def create_barcode(code):
    try:
        rv = BytesIO()
        opts = {"module_width": 0.5, "module_height": 15.0, "font_size": 10, "quiet_zone": 6.5, "write_text": True}
        barcode.get_barcode_class('code128')(code, writer=ImageWriter()).write(rv, options=opts)
        return rv
    except:
        return None


def decode_img(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    decoded = decode(img)
    if not decoded:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        decoded = decode(gray)
        if not decoded:
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            decoded = decode(thresh)
    res = []
    if decoded:
        for obj in decoded:
            res.append(obj.data.decode("utf-8"))
    return img, res


# --- USER AUTH ---
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

# --- MAIN APP ---
if st.session_state["authentication_status"] is True:
    user_name = st.session_state["name"]

    # === MENU NGANG (NAVBAR) ===
    # Đặt ở đầu trang, không dùng Sidebar nữa
    c_logo, c_menu, c_logout = st.columns([1, 6, 1], vertical_alignment="center")

    with c_logo:
        st.image("https://cdn-icons-png.flaticon.com/512/2554/2554045.png", width=50)

    with c_menu:
        # Menu chính nằm ở giữa
        current_tab = st.radio(
            "Menu",
            ["Dashboard", "Nhập Kho", "Xuất Kho", "Truy Xuất"],
            horizontal=True,
            label_visibility="collapsed"  # Ẩn nhãn "Menu" đi cho gọn
        )

    with c_logout:
        authenticator.logout('Log out', 'main')

    st.divider()

    # ================= MODULE 1: NHẬP KHO =================
    if current_tab == "Nhập Kho":
        st.markdown(f'<p class="main-header">📥 {current_tab} (Inbound)</p>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.5], gap="large")

        with c1:
            st.caption("📝 Thông tin lô hàng")
            sku = st.selectbox("SKU:", ["VNM-SUATUOI-RAW", "VNM-DUONG-TINH-LUYEN", "VNM-MEN-PROBI", "VNM-NHADAM-CUBES"])
            qty = st.number_input("Qty:", min_value=1, value=100, step=10)
            batch = st.text_input("Batch:", f"LOT-{random.randint(1000, 9999)}")
            nsx = st.date_input("NSX:", date.today())
            hsd = st.date_input("HSD:", date.today() + timedelta(days=180))
            loc = st.selectbox("Vị trí:", ["Kho A (Lạnh)", "Kho B (Thường)", "Kho C (Hóa Chất)"])

            if st.button("💾 LƯU KHO", type="primary", use_container_width=True):
                if check_duplicate_batch(sku, batch):
                    st.error("Lỗi: Batch đã tồn tại!")
                else:
                    ws = connect_db("Inventory")
                    if ws:
                        full = f"{sku}|{batch}"
                        ws.append_row(
                            [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_name, full, "IMPORT", str(nsx),
                             str(hsd), loc, qty])
                        st.session_state['last_import'] = {'code': full, 'qty': qty, 'batch': batch, 'sku': sku}
                        st.toast("Đã nhập kho!", icon="✅")

        with c2:
            st.caption("🖨️ Kết quả & In Tem")
            # --- KHÔI PHỤC LOGIC HIỂN THỊ CHỜ (WAITING) ---
            if 'last_import' in st.session_state:
                info = st.session_state['last_import']
                st.success(f"Lô mới: {info['batch']}")
                m1, m2 = st.columns(2)
                m1.metric("SKU", info['sku']);
                m2.metric("Qty", info['qty'])

                img = create_barcode(info['code'])
                if img:
                    ic1, ic2 = st.columns([2, 1], vertical_alignment="center")
                    with ic1:
                        st.image(img, use_column_width=True)
                    with ic2:
                        if st.button("📦 Tem Lẻ", key="btn1", use_container_width=True):
                            pdf = FPDF(orientation='L', unit='mm', format=(100, 150));
                            pdf.add_page();
                            pdf.set_font("Helvetica", 'B', 16)
                            pdf.cell(0, 10, "TEM LUU KHO", ln=True, align='C')
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                img.seek(0);
                                tmp.write(img.getvalue());
                                pdf.image(tmp.name, 10, 20, 130)
                            pdf.set_xy(10, 80);
                            pdf.set_font("Helvetica", size=12);
                            pdf.multi_cell(0, 8, f"SKU: {info['sku']}\nBatch: {info['batch']}")
                            st.download_button("⬇ PDF", bytes(pdf.output()), f"P_{info['batch']}.pdf")

                        st.write("")  # Spacer

                        if st.button("🏷️ Tem Loạt", key="btn2", use_container_width=True):
                            pdf = FPDF('P', 'mm', 'A4');
                            pdf.set_auto_page_break(False);
                            pdf.add_page()
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                img.seek(0);
                                tmp.write(img.getvalue());
                                tp = tmp.name
                            x, y, cx, cy = 12, 12, 0, 0
                            for _ in range(info['qty']):
                                pdf.image(tp, x + 2, y + 5, 58);
                                pdf.set_xy(x, y + 32);
                                pdf.set_font("Helvetica", size=8)
                                pdf.cell(62, 5, f"{info['sku']}|{info['batch']}", align='C')
                                cx += 1;
                                x += 62
                                if cx > 2: cx = 0; x = 12; cy += 1; y += 40
                                if cy > 6: pdf.add_page(); cy = 0; y = 12
                            st.download_button("⬇ A4", bytes(pdf.output()), f"B_{info['batch']}.pdf")
            else:
                # HIỂN THỊ ICON CHỜ ĐẸP ĐẼ (CŨ)
                st.info("👈 Vui lòng nhập thông tin bên trái.")
                st.image("https://cdn-icons-png.flaticon.com/512/1466/1466668.png", width=150,
                         caption="Waiting for data...")

    # ================= MODULE 2: XUẤT KHO =================
    elif current_tab == "Xuất Kho":
        st.markdown(f'<p class="main-header">📤 {current_tab} (Outbound)</p>', unsafe_allow_html=True)
        mode = st.radio("Chế độ:", ["🚀 Xuất Lẻ", "🏭 Xuất PO"], horizontal=True)
        st.divider()

        if mode == "🏭 Xuất PO":
            c1, c2 = st.columns([1, 2])
            with c1:
                po = st.selectbox("Chọn PO:", list(MOCK_DB_PO.keys()))
                st.write(MOCK_DB_PO[po]['BOM'])
            with c2:
                with st.form("po"):
                    raw = st.text_input("Scan Barcode:")
                    st.form_submit_button("Check")
                if raw:
                    sku = raw.split("|")[0] if "|" in raw else raw
                    if sku in MOCK_DB_PO[po]['BOM']:
                        st.success(f"✅ Đúng: {sku}")
                        if "|" not in raw:
                            st.warning("⚠️ Thiếu Batch -> Chọn lô (FEFO):")
                            sug = get_available_batches(sku)
                            if sug:
                                batch = st.selectbox("Lô:", sug).split(" (")[0]
                                full = f"{sku}|{batch}"
                                if st.button("🚀 XUẤT"):
                                    connect_db("Inventory").append_row(
                                        [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_name, full, "EXPORT_PO", "",
                                         "", f"To: {po}", -100])  # Demo -100
                                    st.toast("Done!", icon="✅")
                            else:
                                st.error("Hết hàng!")
                        else:
                            if st.button("🚀 XUẤT"):
                                connect_db("Inventory").append_row(
                                    [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_name, raw, "EXPORT_PO", "", "",
                                     f"To: {po}", -100])
                                st.toast("Done!", icon="✅")
                    else:
                        st.error("⛔ Sai vật tư!")
        else:
            scan_type = st.radio("Input:", ["Súng", "Cam"], horizontal=True)
            raw = st.text_input("Mã:") if scan_type == "Súng" else (
                lambda x: decode_img(x.getvalue())[1][0] if x else None)(st.camera_input("Cam"))

            if raw:
                st.markdown(f"### 🔎 `{raw}`")
                qty = st.number_input("SL:", 1, value=1)
                full = raw
                if "|" not in raw:
                    st.warning("Chọn lô (FEFO):")
                    sug = get_available_batches(raw)
                    if sug:
                        full = f"{raw}|{st.selectbox('Lô:', sug).split(' (')[0]}"
                    else:
                        full = None; st.error("Hết hàng")

                if full and st.button("🚀 XÁC NHẬN"):
                    connect_db("Inventory").append_row(
                        [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_name, full, "EXPORT", "", "", "Retail",
                         -qty])
                    st.toast("Đã xuất!", icon="🚛")

    # ================= MODULE 3: DASHBOARD (KHÔI PHỤC TAB) =================
    elif current_tab == "Dashboard":
        st.markdown(f'<p class="main-header">📊 {current_tab} (Analytics)</p>', unsafe_allow_html=True)
        ws_inv = connect_db("Inventory");
        ws_po = connect_db("Production")

        if ws_inv:
            df = pd.DataFrame(ws_inv.get_all_records())
            if not df.empty:
                df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
                df['Real'] = df.apply(lambda x: -x['Qty'] if 'EXPORT' in str(x['Action']) else x['Qty'], axis=1)
                df['SKU'] = df['FullCode'].apply(lambda x: x.split('|')[0] if '|' in x else x)

                total = df.groupby('SKU')['Real'].sum();
                total = total[total > 0]

                c1, c2, c3 = st.columns(3)
                c1.metric("📦 Tổng Tồn", f"{int(total.sum()):,}")
                c2.metric("🔖 Loại SKU", len(total))
                c3.metric("🏭 PO Pending", 2)  # Demo số liệu

                st.divider()

                # --- KHÔI PHỤC TABS (NHẬT KÝ & TIẾN ĐỘ SX) ---
                t1, t2 = st.tabs(["📝 Nhật Ký Kho", "🏭 Tiến Độ Sản Xuất"])

                with t1:
                    st.dataframe(df.sort_values('Timestamp', ascending=False).head(10)[
                                     ['Timestamp', 'FullCode', 'Action', 'Qty', 'User']], use_container_width=True,
                                 hide_index=True)

                with t2:
                    # Tab này đã được khôi phục lại
                    if ws_po:
                        df_p = pd.DataFrame(ws_po.get_all_records())


                        def color_status(val):
                            color = '#d4edda' if val == 'Done' else '#fff3cd' if val == 'Pending' else '#cce5ff'
                            return f'background-color: {color}'


                        st.dataframe(df_p.style.applymap(color_status, subset=['Status']), use_container_width=True)
                    else:
                        st.info("Chưa có dữ liệu sản xuất.")

    # ================= MODULE 4: TRUY XUẤT =================
    elif current_tab == "Truy Xuất":
        st.markdown(f'<p class="main-header">🔍 {current_tab} (Traceability)</p>', unsafe_allow_html=True)
        q = st.text_input("Nhập Batch:", placeholder="VD: LOT-1234")
        if q:
            ws = connect_db("Inventory")
            if ws:
                df = pd.DataFrame(ws.get_all_records())
                sub = df[df['FullCode'].str.contains(q, na=False)].copy()
                if not sub.empty:
                    sub['Qty'] = pd.to_numeric(sub['Qty'], errors='coerce')
                    sub['Real'] = sub.apply(lambda x: -x['Qty'] if 'EXPORT' in x['Action'] else x['Qty'], axis=1)
                    sub['Time'] = pd.to_datetime(sub['Timestamp'])

                    st.success(f"Tìm thấy: {len(sub)} giao dịch")
                    bal = sub['Real'].sum()
                    st.metric("Tồn hiện tại", bal)

                    chart = sub.sort_values("Time").copy()
                    chart['Run'] = chart['Real'].cumsum()
                    st.altair_chart(alt.Chart(chart).mark_line(point=True).encode(x='Time:T', y='Run:Q').interactive(),
                                    use_container_width=True)
                else:
                    st.warning("Không tìm thấy!")

elif st.session_state["authentication_status"] is False:
    st.error('Sai mật khẩu!')
elif st.session_state["authentication_status"] is None:
    st.warning('Vui lòng đăng nhập.')