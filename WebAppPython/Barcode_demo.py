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

# --- CSS TÙY CHỈNH: MENU NGANG & GIAO DIỆN MOBILE ---
st.markdown("""
<style>
    /* 1. Ẩn Sidebar mặc định */
    [data-testid="stSidebar"] {display: none;}
    [data-testid="collapsedControl"] {display: none;}

    /* 2. Menu Ngang (Navbar) dạng thẻ */
    div[data-testid="stRadio"] > label {display: none;} /* Ẩn nhãn 'Menu' */
    div[role="radiogroup"] {
        display: flex;
        flex-direction: row;
        gap: 10px;
        overflow-x: auto; /* Cuộn ngang trên điện thoại bé */
        padding-bottom: 5px;
    }
    div[role="radiogroup"] > label {
        background-color: #f0f2f6;
        padding: 5px 15px;
        border-radius: 20px;
        border: 1px solid #e0e0e0;
        transition: all 0.3s;
    }
    div[role="radiogroup"] > label:hover {
        background-color: #e3f2fd;
        border-color: #2196f3;
    }
    /* Highlight tab đang chọn */
    div[role="radiogroup"] label[data-checked="true"] {
        background-color: #154360 !important;
        color: white !important;
        font-weight: bold;
    }

    /* 3. Tinh chỉnh Header */
    .main-header {
        font-size: 22px !important; 
        font-weight: 700; 
        color: #154360; 
        margin-top: -20px;
    }
    .block-container {
        padding-top: 1rem;
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


# HÀM LẤY BATCH VÀ TỒN KHO THỰC TẾ
def get_batch_stock_info(target_sku):
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
    # Lọc lô còn hàng (>0)
    avail_df = summary[summary['Real'] > 0].copy()

    valid = []
    for index, row in avail_df.iterrows():
        b = row['Batch']
        q = row['Real']
        # Lấy HSD
        row_imp = df[(df['Batch'] == b) & (df['Action'] == 'IMPORT')].head(1)
        hsd = row_imp.iloc[0]['HSD'] if not row_imp.empty else "N/A"
        valid.append({'batch': b, 'qty': q, 'hsd': hsd})

    # Sắp xếp theo HSD (FEFO)
    valid.sort(key=lambda x: x['hsd'])
    return valid  # Trả về list các dict [{'batch':..., 'qty':..., 'hsd':...}]


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

    # === HEADER NAVIGATOR (MENU NGANG) ===
    c_logo, c_menu, c_logout = st.columns([1, 6, 1], vertical_alignment="center")
    with c_logo:
        st.image("https://cdn-icons-png.flaticon.com/512/2554/2554045.png", width=45)
    with c_menu:
        current_tab = st.radio("M", ["Dashboard", "Nhập Kho", "Xuất Kho", "Truy Xuất"], horizontal=True)
    with c_logout:
        authenticator.logout('Exit', 'main')
    st.divider()

    # ================= MODULE 1: NHẬP KHO =================
    if current_tab == "Nhập Kho":
        st.markdown(f'<p class="main-header">📥 NHẬP KHO (INBOUND)</p>', unsafe_allow_html=True)
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
                        if st.button("📦 Tem Lẻ", use_container_width=True):
                            try:
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
                            except:
                                st.error("Lỗi tạo PDF")
                        st.write("")
                        if st.button("🏷️ Tem Loạt", use_container_width=True):
                            try:
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
                            except:
                                st.error("Lỗi tạo PDF")
            else:
                # --- KHÔI PHỤC ICON WAITING ---
                st.info("👈 Vui lòng nhập thông tin bên trái.")
                c_wait1, c_wait2, c_wait3 = st.columns([1, 2, 1])
                with c_wait2:
                    st.image("https://cdn-icons-png.flaticon.com/512/1466/1466668.png", caption="Waiting for data...",
                             width=150)

    # ================= MODULE 2: XUẤT KHO =================
    elif current_tab == "Xuất Kho":
        st.markdown(f'<p class="main-header">📤 XUẤT KHO (OUTBOUND)</p>', unsafe_allow_html=True)
        mode = st.radio("Chế độ:", ["🚀 Xuất Lẻ", "🏭 Xuất PO"], horizontal=True)
        st.divider()

        if mode == "🏭 Xuất PO":
            c1, c2 = st.columns([1, 2])
            with c1:
                po = st.selectbox("Chọn PO:", list(MOCK_DB_PO.keys()))
                st.write("**Công thức (BOM):**")
                st.dataframe(pd.DataFrame(list(MOCK_DB_PO[po]['BOM'].items()), columns=['SKU', 'Định Mức']),
                             hide_index=True)
            with c2:
                with st.form("po"):
                    raw = st.text_input("Scan Barcode:")
                    st.form_submit_button("Check")

                if raw:
                    sku = raw.split("|")[0] if "|" in raw else raw
                    if sku in MOCK_DB_PO[po]['BOM']:
                        st.success(f"✅ ĐÚNG VẬT TƯ: {sku}")

                        # LOGIC CHỌN BATCH & KIỂM TRA TỒN KHO
                        final_code = None
                        max_qty = 0  # Tồn kho tối đa của lô được chọn

                        if "|" in raw:
                            # Quét full code -> Check tồn của lô này
                            batch_in_code = raw.split("|")[1]
                            stock_data = get_batch_stock_info(sku)  # Lấy list tồn
                            # Tìm xem lô này có tồn tại và còn hàng ko
                            found_batch = next((item for item in stock_data if item['batch'] == batch_in_code), None)
                            if found_batch:
                                final_code = raw
                                max_qty = found_batch['qty']
                                st.caption(f"Lô: {batch_in_code} - Tồn: {max_qty}")
                            else:
                                st.error(f"❌ Lô {batch_in_code} đã hết hàng hoặc không tồn tại!")
                        else:
                            # Quét thiếu -> Chọn lô FEFO
                            st.warning("⚠️ Thiếu Batch -> Chọn lô (FEFO):")
                            stock_data = get_batch_stock_info(sku)
                            if stock_data:
                                # Tạo list hiển thị có kèm số lượng tồn
                                options = [f"{i['batch']} (Tồn: {i['qty']} - HSD: {i['hsd']})" for i in stock_data]
                                sel = st.selectbox("Chọn lô:", options)

                                # Parse lại dữ liệu đã chọn
                                sel_batch = sel.split(" (")[0]
                                sel_qty = int(sel.split("Tồn: ")[1].split(" -")[0])

                                final_code = f"{sku}|{sel_batch}"
                                max_qty = sel_qty
                            else:
                                st.error("❌ Hết hàng tồn kho!")

                        # INPUT SỐ LƯỢNG & NÚT XUẤT
                        if final_code and max_qty > 0:
                            st.divider()
                            c_q, c_b = st.columns([1, 1])
                            with c_q:
                                # Max value chặn không cho nhập lố
                                q_out = st.number_input("Thực xuất (Kg):", min_value=1, max_value=int(max_qty), value=1)
                                st.caption(f"Tối đa: {max_qty}")
                            with c_b:
                                st.write("");
                                st.write("")
                                if st.button("🚀 XUẤT NGAY"):
                                    connect_db("Inventory").append_row(
                                        [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_name, final_code,
                                         "EXPORT_PO", "", "", f"To: {po}", -q_out])
                                    st.toast("Thành công!", icon="✅");
                                    st.success(f"Đã xuất: {final_code}")
                    else:
                        st.error("⛔ Sai vật tư!")
        else:
            # --- XUẤT LẺ ---
            scan_type = st.radio("Input:", ["Súng", "Cam"], horizontal=True)
            raw = st.text_input("Mã:") if scan_type == "Súng" else (
                lambda x: decode_img(x.getvalue())[1][0] if x else None)(st.camera_input("Cam"))

            if raw:
                st.markdown(f"### 🔎 `{raw}`")
                final_code = None
                max_qty = 0

                sku_check = raw.split("|")[0] if "|" in raw else raw
                stock_data = get_batch_stock_info(sku_check)

                if "|" in raw:
                    b_check = raw.split("|")[1]
                    f_b = next((i for i in stock_data if i['batch'] == b_check), None)
                    if f_b:
                        final_code = raw; max_qty = f_b['qty']
                    else:
                        st.error("Lô này hết hàng!")
                else:
                    if stock_data:
                        opts = [f"{i['batch']} (Tồn: {i['qty']} - HSD: {i['hsd']})" for i in stock_data]
                        sel = st.selectbox("Chọn lô:", opts)
                        final_code = f"{sku_check}|{sel.split(' (')[0]}"
                        max_qty = int(sel.split("Tồn: ")[1].split(" -")[0])
                    else:
                        st.error("Hết hàng!")

                if final_code and max_qty > 0:
                    q = st.number_input("SL Xuất:", 1, max_value=int(max_qty), value=1)
                    st.caption(f"Max: {max_qty}")
                    if st.button("🚀 XUẤT"):
                        connect_db("Inventory").append_row(
                            [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_name, final_code, "EXPORT", "", "",
                             "Retail", -q])
                        st.toast("Đã xuất!", icon="🚛")

    # ================= MODULE 3: DASHBOARD (KHÔI PHỤC TAB) =================
    elif current_tab == "Dashboard":
        st.markdown(f'<p class="main-header">📊 DASHBOARD</p>', unsafe_allow_html=True)
        ws_inv = connect_db("Inventory");
        ws_po = connect_db("Production")

        if ws_inv:
            df = pd.DataFrame(ws_inv.get_all_records())
            if not df.empty:
                # --- FIX LỖI TYPE ERROR TẠI ĐÂY ---
                # Ép toàn bộ cột FullCode sang dạng chuỗi (string) để tránh lỗi với mã số
                df['FullCode'] = df['FullCode'].astype(str)

                df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
                df['Real'] = df.apply(lambda x: -x['Qty'] if 'EXPORT' in str(x['Action']) else x['Qty'], axis=1)

                # Logic tách chuỗi an toàn hơn
                df['SKU'] = df['FullCode'].apply(lambda x: x.split('|')[0] if '|' in x else x)

                total = df.groupby('SKU')['Real'].sum();
                total = total[total > 0]

                c1, c2, c3 = st.columns(3)
                c1.metric("📦 Tổng Tồn", f"{int(total.sum()):,}")
                c2.metric("🔖 Loại SKU", len(total))
                c3.metric("🏭 PO Pending", 2)

                st.divider()

                # --- TABS ---
                t1, t2 = st.tabs(["📝 Nhật Ký Kho", "🏭 Tiến Độ Sản Xuất"])

                with t1:
                    st.dataframe(df.sort_values('Timestamp', ascending=False).head(15)[
                                     ['Timestamp', 'FullCode', 'Action', 'Qty', 'User']], use_container_width=True,
                                 hide_index=True)

                with t2:
                    if ws_po:
                        df_p = pd.DataFrame(ws_po.get_all_records())


                        def color_status(val):
                            c = '#d4edda' if val == 'Done' else '#fff3cd' if val == 'Pending' else '#cce5ff'
                            return f'background-color: {c}'


                        st.dataframe(df_p.style.applymap(color_status, subset=['Status']), use_container_width=True)
                    else:
                        st.info("Chưa có dữ liệu PO.")

    # ================= MODULE 4: TRUY XUẤT =================
    elif current_tab == "Truy Xuất":
        st.markdown(f'<p class="main-header">🔍 TRACEABILITY</p>', unsafe_allow_html=True)
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

                    with st.expander("Chi tiết"):
                        st.dataframe(sub[['Timestamp', 'Action', 'Qty', 'User', 'Location']], use_container_width=True)
                else:
                    st.warning("Không tìm thấy!")

elif st.session_state["authentication_status"] is False:
    st.error('Sai mật khẩu!')
elif st.session_state["authentication_status"] is None:
    st.warning('Vui lòng đăng nhập.')