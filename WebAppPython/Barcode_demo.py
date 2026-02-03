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

# --- CSS TẠO GIAO DIỆN APP (LOGO TRÁI - MENU PHẢI) ---
st.markdown("""
<style>
    /* 1. Tùy chỉnh Header mặc định của Streamlit */
    header[data-testid="stHeader"] {
        /* Màu nền Header (Trắng hoặc Xám tùy theme) */
        background-color: var(--background-color);
        /* Đường viền dưới cho giống App */
        border-bottom: 1px solid #f0f2f6;
        /* Đảm bảo nó luôn nằm trên cùng */
        z-index: 999999;
    }

    /* 2. CHÈN LOGO VÀO GÓC TRÁI HEADER (QUAN TRỌNG NHẤT) */
    header[data-testid="stHeader"]::before {
        content: "";
        /* Link Logo (Thay bằng link logo thật của ông nếu cần) */
        background-image: url('https://cdn-icons-png.flaticon.com/512/2554/2554045.png');
        background-size: contain;
        background-repeat: no-repeat;
        position: absolute;
        left: 20px;       /* Cách lề trái 20px */
        top: 50%;         /* Căn giữa chiều dọc */
        transform: translateY(-50%);
        width: 40px;      /* Kích thước logo */
        height: 40px;
        z-index: 999;
    }

    /* 3. Dời nút 3 gạch (Hamburger) qua phải và đổi màu cho đẹp (nếu cần) */
    [data-testid="stSidebarCollapsedControl"] {
        color: #154360 !important; /* Màu xanh Vinamilk */
    }

    /* 4. Dashboard 1 hàng ngang trên Mobile */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        padding-bottom: 5px;
    }
    
    /* 5. Tinh chỉnh lại khoảng cách nội dung để không bị Header che */
    .block-container {
        padding-top: 3rem !important; 
    }
    
    /* Ẩn nút Manage App / Deploy thừa thãi ở góc phải (nếu muốn sạch sẽ) */
    .stAppDeployButton {
        display: none;
    }
    
    /* Tiêu đề trang con */
    .main-header {
        font-size: 22px !important; 
        font-weight: 700; 
        color: var(--text-color);
        margin-bottom: 15px;
        margin-top: 10px;
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
            try: ws = sh.worksheet(sheet_name)
            except: ws = sh.add_worksheet(title=sheet_name, rows=1000, cols=10)
            return ws
    except: return None
    return None

def check_duplicate_batch(sku, batch):
    ws = connect_db("Inventory")
    if ws:
        try:
            all_codes = ws.col_values(3)
            return f"{sku}|{batch}" in all_codes
        except: return False
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
        opts = {"module_width":0.5, "module_height":15.0, "font_size":10, "quiet_zone":6.5, "write_text":True}
        barcode.get_barcode_class('code128')(code, writer=ImageWriter()).write(rv, options=opts)
        return rv
    except: return None

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
            'manager': {'name': 'Quản Lý Kho', 'password': '$2b$12$MWFqC9gNSU93.GfxSUSqnOn4duvXwOrW2WX6Kq6QkL2f6ZgYypkY.'},
            'staff': {'name': 'Nhân Viên', 'password': '$2b$12$ZCxqkVJBagfsWJBoFntXSedewNTSBYbcKJHYbXdVP0k4jErvVYRVq'}
        }
    },
    'cookie': {'expiry_days': 1, 'key': 'vina_key', 'name': 'vina_cookie'}
}
authenticator = stauth.Authenticate(config_user['credentials'], config_user['cookie']['name'], config_user['cookie']['key'], config_user['cookie']['expiry_days'])
authenticator.login()

# --- MAIN APP ---
if st.session_state["authentication_status"] is True:
    user_name = st.session_state["name"]
    
    # === MENU SIDEBAR (NƠI CHỨA NÚT BẤM) ===
    # Khi bấm nút 3 gạch, cái này sẽ trượt ra
    with st.sidebar:
        # Không cần st.image ở đây nữa vì đã có trên Header dính rồi
        st.markdown(f"👤 **{user_name}**")
        
        current_tab = st.radio(
            "CHỨC NĂNG:",
            ["Dashboard", "Nhập Kho", "Xuất Kho", "Truy Xuất"],
            index=0
        )
        
        st.divider()
        authenticator.logout('Đăng xuất', 'sidebar')
        st.caption("v8.0 Mobile Native")

    # ================= MODULE 1: NHẬP KHO =================
    if current_tab == "Nhập Kho":
        st.markdown(f'<p class="main-header">📥 NHẬP KHO</p>', unsafe_allow_html=True)
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
                if check_duplicate_batch(sku, batch): st.error("Lỗi: Batch đã tồn tại!")
                else:
                    ws = connect_db("Inventory")
                    if ws:
                        full = f"{sku}|{batch}"
                        ws.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_name, full, "IMPORT", str(nsx), str(hsd), loc, qty])
                        st.session_state['last_import'] = {'code':full, 'qty':qty, 'batch':batch, 'sku':sku}
                        st.toast("Đã nhập kho!", icon="✅")

        with c2:
            st.caption("🖨️ Kết quả & In Tem")
            if 'last_import' in st.session_state:
                info = st.session_state['last_import']
                st.success(f"Lô mới: {info['batch']}")
                m1, m2 = st.columns(2)
                m1.metric("SKU", info['sku']); m2.metric("Qty", info['qty'])
                
                img = create_barcode(info['code'])
                if img:
                    ic1, ic2 = st.columns([2, 1], vertical_alignment="center")
                    with ic1: st.image(img, use_column_width=True)
                    with ic2:
                        if st.button("📦 Tem Lẻ", use_container_width=True):
                            try:
                                pdf = FPDF(orientation='L', unit='mm', format=(100, 150)); pdf.add_page(); pdf.set_font("Helvetica", 'B', 16)
                                pdf.cell(0, 10, "TEM LUU KHO", ln=True, align='C')
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                    img.seek(0); tmp.write(img.getvalue()); pdf.image(tmp.name, 10, 20, 130)
                                pdf.set_xy(10,80); pdf.set_font("Helvetica", size=12); pdf.multi_cell(0,8,f"SKU: {info['sku']}\nBatch: {info['batch']}")
                                st.download_button("⬇ PDF", bytes(pdf.output()), f"P_{info['batch']}.pdf")
                            except: st.error("Lỗi tạo PDF")
                        st.write("") 
                        if st.button("🏷️ Tem Loạt", use_container_width=True):
                            try:
                                pdf = FPDF('P','mm','A4'); pdf.set_auto_page_break(False); pdf.add_page()
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                    img.seek(0); tmp.write(img.getvalue()); tp = tmp.name
                                x,y,cx,cy = 12,12,0,0
                                for _ in range(info['qty']):
                                    pdf.image(tp, x+2, y+5, 58); pdf.set_xy(x, y+32); pdf.set_font("Helvetica",size=8)
                                    pdf.cell(62,5, f"{info['sku']}|{info['batch']}", align='C')
                                    cx+=1; x+=62
                                    if cx>2: cx=0; x=12; cy+=1; y+=40
                                    if cy>6: pdf.add_page(); cy=0; y=12
                                st.download_button("⬇ A4", bytes(pdf.output()), f"B_{info['batch']}.pdf")
                            except: st.error("Lỗi tạo PDF")
            else:
                st.info("👈 Vui lòng nhập thông tin.")
                c_wait1, c_wait2, c_wait3 = st.columns([1,2,1])
                with c_wait2:
                    st.image("https://cdn-icons-png.flaticon.com/512/1466/1466668.png", caption="Waiting...", width=100)

    # ================= MODULE 2: XUẤT KHO =================
    elif current_tab == "Xuất Kho":
        st.markdown(f'<p class="main-header">📤 XUẤT KHO</p>', unsafe_allow_html=True)
        mode = st.radio("Chế độ:", ["🚀 Xuất Lẻ", "🏭 Xuất PO"], horizontal=True)
        st.divider()

        if mode == "🏭 Xuất PO":
            c1, c2 = st.columns([1, 2])
            with c1:
                po = st.selectbox("Chọn PO:", list(MOCK_DB_PO.keys()))
                st.write("**Công thức (BOM):**")
                st.dataframe(pd.DataFrame(list(MOCK_DB_PO[po]['BOM'].items()), columns=['SKU', 'Định Mức']), hide_index=True)
            with c2:
                with st.form("po"):
                    raw = st.text_input("Scan Barcode:")
                    st.form_submit_button("Check")
                
                if raw:
                    sku = raw.split("|")[0] if "|" in raw else raw
                    if sku in MOCK_DB_PO[po]['BOM']:
                        st.success(f"✅ ĐÚNG: {sku}")
                        final_code = None; max_qty = 0
                        
                        if "|" in raw:
                            batch_in_code = raw.split("|")[1]
                            stock_data = get_batch_stock_info(sku)
                            found_batch = next((item for item in stock_data if item['batch'] == batch_in_code), None)
                            if found_batch:
                                final_code = raw; max_qty = found_batch['qty']
                                st.caption(f"Lô: {batch_in_code} - Tồn: {max_qty}")
                            else: st.error(f"❌ Lô {batch_in_code} hết hàng!")
                        else:
                            st.warning("⚠️ Chọn lô (FEFO):")
                            stock_data = get_batch_stock_info(sku)
                            if stock_data:
                                opts = [f"{i['batch']} (Tồn: {i['qty']} - HSD: {i['hsd']})" for i in stock_data]
                                sel = st.selectbox("Chọn lô:", opts)
                                final_code = f"{sku}|{sel.split(' (')[0]}"
                                max_qty = int(sel.split("Tồn: ")[1].split(" -")[0])
                            else: st.error("❌ Hết hàng!")
                        
                        if final_code and max_qty > 0:
                            st.divider()
                            c_q, c_b = st.columns([1,1])
                            with c_q: q_out = st.number_input("Thực xuất (Kg):", min_value=1, max_value=int(max_qty), value=1)
                            with c_b: 
                                st.write(""); st.write("")
                                if st.button("🚀 XUẤT NGAY"):
                                    connect_db("Inventory").append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_name, final_code, "EXPORT_PO", "", "", f"To: {po}", -q_out])
                                    st.toast("Thành công!", icon="✅"); st.success(f"Đã xuất: {final_code}")
                    else: st.error("⛔ Sai vật tư!")
        else:
            scan_type = st.radio("Input:", ["Súng", "Cam"], horizontal=True)
            raw = st.text_input("Mã:") if scan_type == "Súng" else (lambda x: decode_img(x.getvalue())[1][0] if x else None)(st.camera_input("Cam"))
            
            if raw:
                st.markdown(f"### 🔎 `{raw}`")
                final_code = None; max_qty = 0
                sku_check = raw.split("|")[0] if "|" in raw else raw
                stock_data = get_batch_stock_info(sku_check)

                if "|" in raw:
                     b_check = raw.split("|")[1]
                     f_b = next((i for i in stock_data if i['batch'] == b_check), None)
                     if f_b: final_code = raw; max_qty = f_b['qty']
                     else: st.error("Lô này hết hàng!")
                else:
                    if stock_data:
                        opts = [f"{i['batch']} (Tồn: {i['qty']} - HSD: {i['hsd']})" for i in stock_data]
                        sel = st.selectbox("Chọn lô:", opts)
                        final_code = f"{sku_check}|{sel.split(' (')[0]}"
                        max_qty = int(sel.split("Tồn: ")[1].split(" -")[0])
                    else: st.error("Hết hàng!")

                if final_code and max_qty > 0:
                    q = st.number_input("SL Xuất:", 1, max_value=int(max_qty), value=1)
                    if st.button("🚀 XUẤT"):
                        connect_db("Inventory").append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_name, final_code, "EXPORT", "", "", "Retail", -q])
                        st.toast("Đã xuất!", icon="🚛")

    # ================= MODULE 3: DASHBOARD =================
    elif current_tab == "Dashboard":
        st.markdown(f'<p class="main-header">📊 DASHBOARD</p>', unsafe_allow_html=True)
        ws_inv = connect_db("Inventory"); ws_po = connect_db("Production")
        
        if ws_inv:
            df = pd.DataFrame(ws_inv.get_all_records())
            if not df.empty:
                df['FullCode'] = df['FullCode'].astype(str)
                df['Qty'] = pd.to_numeric(df['Qty'], errors='coerce').fillna(0)
                df['Real'] = df.apply(lambda x: -x['Qty'] if 'EXPORT' in str(x['Action']) else x['Qty'], axis=1)
                df['SKU'] = df['FullCode'].apply(lambda x: x.split('|')[0] if '|' in x else x)
                
                total = df.groupby('SKU')['Real'].sum(); total = total[total>0]
                
                # --- DASHBOARD METRICS: 1 HÀNG NGANG ---
                c1, c2, c3 = st.columns(3)
                c1.metric("📦 Tổng Tồn", f"{int(total.sum()):,}")
                c2.metric("🔖 Loại SKU", len(total))
                c3.metric("🏭 Pending", 2)
                
                st.divider()
                
                t1, t2 = st.tabs(["📝 Nhật Ký Kho", "🏭 Tiến Độ SX"])
                with t1:
                    st.dataframe(df.sort_values('Timestamp', ascending=False).head(10)[['Timestamp','FullCode','Action','Qty','User']], use_container_width=True, hide_index=True)
                    
                with t2:
                    if ws_po:
                        df_p = pd.DataFrame(ws_po.get_all_records())
                        def color_status(val):
                            if val == 'Done': return 'background-color: #d4edda; color: black;'
                            elif val == 'Pending': return 'background-color: #fff3cd; color: black;'
                            elif val == 'Running': return 'background-color: #cce5ff; color: black;'
                            return ''
                        st.dataframe(df_p.style.applymap(color_status, subset=['Status']), use_container_width=True)
                    else: st.info("Chưa có dữ liệu PO.")

    # ================= MODULE 4: TRUY XUẤT =================
    elif current_tab == "Truy Xuất":
        st.markdown(f'<p class="main-header">🔍 TRACEABILITY</p>', unsafe_allow_html=True)
        q = st.text_input("Nhập Batch:", placeholder="VD: LOT-1234")
        if q:
            ws = connect_db("Inventory")
            if ws:
                df = pd.DataFrame(ws.get_all_records())
                df['FullCode'] = df['FullCode'].astype(str)
                sub = df[df['FullCode'].str.contains(q, case=False, na=False)].copy()
                
                if not sub.empty:
                    sub['Qty'] = pd.to_numeric(sub['Qty'], errors='coerce')
                    sub['Real'] = sub.apply(lambda x: -x['Qty'] if 'EXPORT' in str(x['Action']) else x['Qty'], axis=1)
                    sub['Time'] = pd.to_datetime(sub['Timestamp'])
                    
                    st.success(f"Tìm thấy: {len(sub)} giao dịch")
                    bal = sub['Real'].sum()
                    st.metric("Tồn hiện tại", bal)
                    
                    chart = sub.sort_values("Time").copy()
                    chart['Run'] = chart['Real'].cumsum()
                    st.altair_chart(alt.Chart(chart).mark_line(point=True).encode(x='Time:T', y='Run:Q').interactive(), use_container_width=True)
                    
                    with st.expander("Chi tiết"):
                        st.dataframe(sub[['Timestamp','Action','Qty','User','Location']], use_container_width=True)
                else: st.warning("Không tìm thấy!")

elif st.session_state["authentication_status"] is False:
    st.error('Sai mật khẩu!')
elif st.session_state["authentication_status"] is None:
    st.warning('Vui lòng đăng nhập.')
