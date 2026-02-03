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

# --- CSS TỐI ƯU ---
st.markdown("""
<style>
    [data-testid="stHorizontalBlock"] {flex-wrap: nowrap !important; overflow-x: auto !important; padding-bottom: 5px;}
    .stAppDeployButton {display: none;}
    .main-header {font-size: 24px !important; font-weight: 700; color: #154360; margin-bottom: 10px;}
    .block-container {padding-top: 2rem;}
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


def get_batch_stock_info(target_sku):
    ws = connect_db("Inventory")
    if not ws: return []
    df = pd.DataFrame(ws.get_all_records())
    if df.empty: return []

    df['FullCode'] = df['FullCode'].astype(str)
    df['SKU'] = df['FullCode'].apply(lambda x: x.split('|')[0] if '|' in x else x)
    df['Batch'] = df['FullCode'].apply(lambda x: x.split('|')[1] if '|' in x else 'Unknown')

    df_sku = df[df['SKU'] == target_sku].copy()
    df_sku['Qty'] = pd.to_numeric(df_sku['Qty'], errors='coerce').fillna(0)
    df_sku['Real'] = df_sku.apply(lambda x: -x['Qty'] if 'EXPORT' in str(x['Action']) else x['Qty'], axis=1)

    summary = df_sku.groupby('Batch')['Real'].sum().reset_index()
    avail_df = summary[summary['Real'] > 0].copy()

    valid = []
    for index, row in avail_df.iterrows():
        b = row['Batch']
        q = row['Real']
        row_imp = df[(df['Batch'] == b) & (df['Action'] == 'IMPORT')].head(1)
        hsd = row_imp.iloc[0]['HSD'] if not row_imp.empty else "N/A"
        valid.append({'batch': b, 'qty': q, 'hsd': hsd})

    valid.sort(key=lambda x: x['hsd'])
    return valid


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

    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2554/2554045.png", width=80)
        st.markdown(f"👤 **{user_name}**")
        current_tab = st.radio("CHỨC NĂNG:", ["Dashboard", "Nhập Kho", "Xuất Kho", "Truy Xuất"], index=0)
        st.divider()
        authenticator.logout('Đăng xuất', 'sidebar')

    # ================= NHẬP KHO =================
    if current_tab == "Nhập Kho":
        st.markdown(f'<p class="main-header">📥 NHẬP KHO</p>', unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1.5], gap="large")
        with c1:
            sku = st.selectbox("SKU:", ["VNM-SUATUOI-RAW", "VNM-DUONG-TINH-LUYEN", "VNM-MEN-PROBI", "VNM-NHADAM-CUBES"])
            qty = st.number_input("Qty:", min_value=1, value=100)
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
            if 'last_import' in st.session_state:
                info = st.session_state['last_import']
                st.success(f"Lô mới: {info['batch']}")
                m1, m2 = st.columns(2);
                m1.metric("SKU", info['sku']);
                m2.metric("Qty", info['qty'])
                img = create_barcode(info['code'])
                if img: st.image(img, width=250)

    # ================= XUẤT KHO (ĐÃ KHÓA NÚT) =================
    elif current_tab == "Xuất Kho":
        st.markdown(f'<p class="main-header">📤 XUẤT KHO</p>', unsafe_allow_html=True)
        mode = st.radio("Chế độ:", ["🚀 Xuất Lẻ", "🏭 Xuất PO"], horizontal=True)
        st.divider()

        if mode == "🏭 Xuất PO":
            c1, c2 = st.columns([1, 2])
            with c1:
                po = st.selectbox("Chọn PO:", list(MOCK_DB_PO.keys()))
                st.write("**BOM:**")
                st.dataframe(pd.DataFrame(list(MOCK_DB_PO[po]['BOM'].items()), columns=['SKU', 'Định Mức']),
                             hide_index=True)
            with c2:
                with st.form("po"):
                    raw = st.text_input("Scan Barcode:")
                    st.form_submit_button("Check")

                if raw:
                    sku = raw.split("|")[0] if "|" in raw else raw
                    if sku in MOCK_DB_PO[po]['BOM']:
                        st.success(f"✅ ĐÚNG: {sku}")
                        final_code = None;
                        max_qty = 0

                        if "|" in raw:
                            batch_in_code = raw.split("|")[1]
                            stock_data = get_batch_stock_info(sku)
                            found_batch = next((item for item in stock_data if item['batch'] == batch_in_code), None)
                            if found_batch:
                                final_code = raw;
                                max_qty = found_batch['qty']
                                st.caption(f"Lô: {batch_in_code} - Tồn: {max_qty}")
                            else:
                                st.error(f"❌ Lô {batch_in_code} hết hàng!")
                        else:
                            st.warning("⚠️ Chọn lô (FEFO):")
                            stock_data = get_batch_stock_info(sku)
                            if stock_data:
                                opts = [f"{i['batch']} (Tồn: {i['qty']} - HSD: {i['hsd']})" for i in stock_data]
                                sel = st.selectbox("Chọn lô:", opts)
                                final_code = f"{sku}|{sel.split(' (')[0]}"
                                max_qty = int(sel.split("Tồn: ")[1].split(" -")[0])
                            else:
                                st.error("❌ Hết hàng!")

                        if final_code and max_qty > 0:
                            st.divider()
                            q_out = st.number_input("Thực xuất (Kg):", min_value=1, value=1)

                            # --- LOGIC KHÓA NÚT TẠI ĐÂY ---
                            is_invalid = q_out > max_qty
                            if is_invalid:
                                st.error(f"⛔ Vượt tồn kho! Chỉ còn {max_qty} kg.")

                            # Dùng tham số disabled=is_invalid để khóa nút cứng
                            if st.button("🚀 XUẤT NGAY", type="primary", disabled=is_invalid):
                                ws = connect_db("Inventory")
                                if ws:
                                    ws.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_name, final_code,
                                                   "EXPORT_PO", "", "", f"To: {po}", -q_out])
                                    st.toast("Thành công!", icon="✅");
                                    st.success(f"Đã xuất: {final_code}")
                    else:
                        st.error("⛔ Sai vật tư!")

        else:  # XUẤT LẺ
            raw = (lambda x: decode_img(x.getvalue())[1][0] if x else None)(st.camera_input("Cam"))
            if not raw: raw = st.text_input("Hoặc nhập tay mã:")
            if raw:
                st.markdown(f"### 🔎 `{raw}`")
                sku_check = raw.split("|")[0] if "|" in raw else raw
                stk = get_batch_stock_info(sku_check)
                final_code = None;
                max_qty = 0
                if stk:
                    sel = st.selectbox("Chọn lô:", [f"{i['batch']} (Tồn: {i['qty']})" for i in stk])
                    final_code = f"{sku_check}|{sel.split(' (')[0]}"
                    max_qty = int(sel.split("Tồn: ")[1].split(")")[0])

                    q = st.number_input("SL Xuất:", 1, value=1)
                    invalid_retail = q > max_qty
                    if invalid_retail: st.error("Quá tồn kho!")

                    if st.button("🚀 XUẤT", disabled=invalid_retail):
                        connect_db("Inventory").append_row(
                            [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_name, final_code, "EXPORT", "", "",
                             "Retail", -q])
                        st.toast("Đã xuất!", icon="🚛")

    # ================= DASHBOARD =================
    elif current_tab == "Dashboard":
        st.markdown(f'<p class="main-header">📊 DASHBOARD</p>', unsafe_allow_html=True)
        ws_inv = connect_db("Inventory")
        if ws_inv:
            df = pd.DataFrame(ws_inv.get_all_records())
            if not df.empty:
                df['FullCode'] = df['FullCode'].astype(str)
                df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
                df['Real'] = df.apply(lambda x: -x['Qty'] if 'EXPORT' in str(x['Action']) else x['Qty'], axis=1)
                total = df.groupby('FullCode')['Real'].sum().sum()
                st.metric("📦 Tổng Tồn Kho", f"{int(total):,}")
                st.dataframe(df.sort_values('Timestamp', ascending=False).head(10), use_container_width=True)

    # ================= TRUY XUẤT =================
    elif current_tab == "Truy Xuất":
        st.markdown(f'<p class="main-header">🔍 TRACEABILITY</p>', unsafe_allow_html=True)
        q = st.text_input("Nhập Batch:", placeholder="VD: LOT-1234")
        if q:
            ws = connect_db("Inventory")
            if ws:
                df = pd.DataFrame(ws.get_all_records())
                sub = df[df['FullCode'].astype(str).str.contains(q)].copy()
                if not sub.empty:
                    st.success(f"Tìm thấy: {len(sub)} giao dịch")
                    st.dataframe(sub, use_container_width=True)

elif st.session_state["authentication_status"] is False:
    st.error('Sai mật khẩu!')
elif st.session_state["authentication_status"] is None:
    st.warning('Vui lòng đăng nhập.')