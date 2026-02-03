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
st.set_page_config(page_title="WMS Demo - Vinamilk Standard", layout="wide", page_icon="🏭")


def remove_accents(input_str):
    if not input_str: return ""
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


# --- GIẢ LẬP DỮ LIỆU LỆNH SẢN XUẤT (PO) ---
MOCK_DB_PO = {
    "PO-2026-001": {
        "Product": "Sữa Tươi 100% (Lô Sáng)",
        "BOM": {"VNM-SUATUOI-RAW": 100, "VNM-DUONG-TINH-LUYEN": 5}
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
        st.error(f"⚠️ Lỗi kết nối GSheets: {e}")
        return None
    return None


def check_duplicate_batch(sku, batch):
    ws = connect_db("Inventory")
    if ws:
        try:
            all_codes = ws.col_values(3)
            full_code = f"{sku}|{batch}"
            return full_code in all_codes
        except:
            return False
    return False


def get_available_batches(target_sku):
    ws = connect_db("Inventory")
    if not ws: return []
    df = pd.DataFrame(ws.get_all_records())
    if df.empty: return []

    df['SKU_Only'] = df['FullCode'].apply(lambda x: str(x).split('|')[0] if '|' in str(x) else str(x))
    df['Batch_Only'] = df['FullCode'].apply(lambda x: str(x).split('|')[1] if '|' in str(x) else 'Unknown')

    df_sku = df[df['SKU_Only'] == target_sku].copy()
    df_sku['Qty'] = pd.to_numeric(df_sku['Qty'], errors='coerce').fillna(0)
    df_sku['Real_Qty'] = df_sku.apply(lambda x: -x['Qty'] if 'EXPORT' in str(x['Action']).upper() else x['Qty'], axis=1)

    batch_summary = df_sku.groupby('Batch_Only')['Real_Qty'].sum().reset_index()
    available_batches = batch_summary[batch_summary['Real_Qty'] > 0]['Batch_Only'].tolist()

    valid_batches_info = []
    for b in available_batches:
        row_info = df[(df['Batch_Only'] == b) & (df['Action'] == 'IMPORT')].head(1)
        if not row_info.empty:
            hsd = row_info.iloc[0]['HSD']
            valid_batches_info.append((b, hsd))

    valid_batches_info.sort(key=lambda x: x[1])
    return [f"{b} (HSD: {hsd})" for b, hsd in valid_batches_info]


# --- HÀM XỬ LÝ ẢNH & BARCODE ---
def create_barcode(code):
    try:
        rv = BytesIO()
        BARCODE_CLASS = barcode.get_barcode_class('code128')
        options = {"module_width": 0.5, "module_height": 18.0, "font_size": 10, "text_distance": 4.0, "quiet_zone": 6.5,
                   "write_text": True}
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

    # === SIDEBAR (MENU) ===
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2554/2554045.png", width=70)
        st.title("WMS PRO")
        st.caption(f"User: {user_name}")

        current_tab = st.radio(
            "CHỨC NĂNG:",
            ["📊 Dashboard", "📥 Nhập Kho (Inbound)", "📤 Xuất Kho (Outbound)", "🔍 Truy Xuất (Traceability)"],
            index=0
        )

        st.divider()

        # --- KỊCH BẢN DEMO (HELP GUIDE) ---
        with st.expander("❓ Hướng dẫn Demo (Kịch bản)"):
            st.markdown("""
            **1. Nhập Kho:**
            - Tạo mã `VNM-SUATUOI`.
            - Bấm **Lưu Kho**.
            - Bấm **In Tem** (Demo in PDF).

            **2. Xuất Kho (Bình thường):**
            - Chọn chế độ **Súng Quét**.
            - Nhập mã thiếu `VNM-SUATUOI`.
            - Hệ thống sẽ **Cảnh báo** & Gợi ý Lô (FEFO).
            - Chọn Lô -> Xuất.

            **3. Xuất Kho (Theo PO):**
            - Chọn `PO-2026-001`.
            - Quét mã sai -> Báo đỏ.
            - Quét mã đúng -> Báo xanh.

            **4. Truy Xuất:**
            - Nhập số Batch vừa tạo.
            - Xem biểu đồ dòng chảy.
            """)

        st.divider()
        authenticator.logout('Đăng xuất', 'sidebar')

    # === MAIN HEADER ===
    st.markdown("""
        <style>
        .main-header {font-size: 26px !important; font-weight: 700; color: #154360; margin-bottom: 10px;}
        .block-container {padding-top: 2rem;}
        </style>
    """, unsafe_allow_html=True)

    st.markdown(f'<p class="main-header">{current_tab}</p>', unsafe_allow_html=True)

    # ================= MODULE 1: NHẬP KHO =================
    if "Nhập Kho" in current_tab:
        c1, c2 = st.columns([1, 1.5], gap="medium")
        with c1:
            st.markdown("#### 1. Thông tin nhập liệu")
            sku = st.selectbox("SKU/Nguyên Liệu:",
                               ["VNM-SUATUOI-RAW", "VNM-DUONG-TINH-LUYEN", "VNM-MEN-PROBI", "VNM-NHADAM-CUBES"])
            qty = st.number_input("Số lượng (Kg/Unit):", min_value=1, value=100, step=10)
            batch = st.text_input("Số Batch:", f"LOT-{random.randint(1000, 9999)}")
            nsx = st.date_input("NSX:", date.today())
            hsd = st.date_input("HSD:", date.today() + timedelta(days=180))
            loc = st.selectbox("Vị trí:", ["Kho A (Lạnh)", "Kho B (Thường)", "Kho C (Hóa Chất)"])

            full_code = f"{sku}|{batch}"
            st.info(f"🆔 Mã định danh: {full_code}")

            if st.button("💾 Lưu Kho", type="primary", use_container_width=True):
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
            st.markdown("#### 2. Kết quả & In Tem")
            if 'last_import' in st.session_state:
                info = st.session_state['last_import']
                st.success(f"✅ Đã lưu lô: {info['batch']}")

                m1, m2, m3 = st.columns(3)
                m1.metric("SKU", info['sku'])
                m2.metric("Batch", info['batch'])
                m3.metric("Qty", info['qty'])

                st.divider()
                st.markdown("🖨️ **Tùy chọn In Ấn:**")

                img = create_barcode(info['code'])
                if img:
                    ic1, ic2, ic3 = st.columns([2, 1, 1], vertical_alignment="center")
                    with ic1:
                        st.image(img, use_column_width=True)
                    with ic2:
                        if st.button("📦 Tem Thùng", use_container_width=True):
                            try:
                                pdf = FPDF(orientation='L', unit='mm', format=(100, 150))
                                pdf.add_page();
                                pdf.set_font("Helvetica", 'B', 16)
                                pdf.cell(0, 10, txt=remove_accents("TEM LUU KHO"), ln=True, align='C')
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                    img.seek(0);
                                    tmp.write(img.getvalue());
                                    pdf.image(tmp.name, x=10, y=20, w=130)
                                pdf.set_xy(10, 80);
                                pdf.set_font("Helvetica", size=12)
                                pdf.multi_cell(0, 8,
                                               txt=f"SKU: {info['sku']}\nBatch: {info['batch']}\nQty: {info['qty']}")
                                pdf_data = bytes(pdf.output());
                                st.download_button("⬇️ PDF", pdf_data, f"Pallet_{info['batch']}.pdf")
                            except Exception as e:
                                st.error(str(e))
                    with ic3:
                        if st.button("🏷️ Tem Loạt", use_container_width=True):
                            try:
                                pdf_bulk = FPDF(orientation='P', unit='mm', format='A4')
                                pdf_bulk.set_auto_page_break(auto=False, margin=0);
                                pdf_bulk.add_page()
                                mx, my, cw, rh = 12, 12, 62, 40;
                                cols, rows = 3, 7;
                                x, y, cx, cy = mx, my, 0, 0
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_b:
                                    img.seek(0);
                                    tmp_b.write(img.getvalue());
                                    t_path = tmp_b.name
                                for i in range(int(info['qty'])):
                                    pdf_bulk.image(t_path, x=x + 2, y=y + 5, w=cw - 4)
                                    pdf_bulk.set_font("Helvetica", size=8);
                                    pdf_bulk.set_xy(x, y + rh - 8)
                                    pdf_bulk.cell(cw, 5, txt=f"{info['sku']} | {info['batch']}", align='C')
                                    cx += 1;
                                    if cx < cols:
                                        x += cw
                                    else:
                                        cx = 0; x = mx; cy += 1; y += rh
                                    if cy >= rows: pdf_bulk.add_page(); cy = 0; y = my; x = mx
                                try:
                                    bulk_bytes = bytes(pdf_bulk.output())
                                except:
                                    bulk_bytes = pdf_bulk.output(dest='S').encode('latin-1')
                                st.download_button("⬇️ A4", bulk_bytes, f"Bulk_{info['batch']}.pdf")
                            except Exception as e:
                                st.error(str(e))
            else:
                st.info("👈 Vui lòng nhập thông tin và bấm Lưu Kho.")

    # ================= MODULE 2: XUẤT KHO =================
    elif "Xuất Kho" in current_tab:
        st.caption("Hỗ trợ FEFO (First Expired - First Out) & Kiểm soát theo Lệnh sản xuất.")
        mode = st.radio("Chế độ:", ["🚀 Xuất Lẻ (Thông thường)", "🏭 Xuất Cho Sản Xuất (Theo PO)"], horizontal=True)
        st.divider()

        if "Theo PO" in mode:
            cp1, cp2 = st.columns([1, 2])
            with cp1:
                st.markdown("##### 1. Chọn Lệnh SX")
                po_sel = st.selectbox("PO:", list(MOCK_DB_PO.keys()))
                po_data = MOCK_DB_PO[po_sel]
                st.info(f"SP: **{po_data['Product']}**")
                st.write("**BOM (Định mức):**")
                st.dataframe(pd.DataFrame(list(po_data['BOM'].items()), columns=['SKU', 'Cần (Kg)']), hide_index=True)
            with cp2:
                st.markdown("##### 2. Quét Đối Chiếu")
                with st.form("po_form"):
                    raw_scan = st.text_input("Scan Barcode:", placeholder="Quét mã...")
                    btn_po = st.form_submit_button("🔍 Kiểm tra")

                if raw_scan:
                    if "|" in raw_scan:
                        s_sku, s_batch = raw_scan.split("|"); full = raw_scan; is_sel = True
                    else:
                        s_sku = raw_scan; s_batch = None; full = None; is_sel = False

                    if s_sku in po_data['BOM']:
                        req_qty = po_data['BOM'][s_sku]
                        st.success(f"✅ ĐÚNG VẬT TƯ: {s_sku}")
                        st.progress(0, text=f"Target: {req_qty} Kg")

                        final_code = None
                        if is_sel:
                            final_code = full; st.caption(f"Lô: {s_batch}")
                        else:
                            st.warning(f"⚠️ Mã `{s_sku}` thiếu Batch! Chọn lô (FEFO):")
                            sug = get_available_batches(s_sku)
                            if sug:
                                sb = st.selectbox("Chọn lô:", sug)
                                rb = sb.split(" (")[0]
                                final_code = f"{s_sku}|{rb}"
                            else:
                                st.error("❌ Hết tồn kho!")

                        if final_code:
                            st.divider()
                            c_q, c_b = st.columns([1, 1])
                            with c_q:
                                q_out = st.number_input("Thực xuất (Kg):", min_value=1, value=int(req_qty))
                            with c_b:
                                st.write("");
                                st.write("")
                                if st.button("🚀 Xuất PO", type="primary"):
                                    ws = connect_db("Inventory")
                                    if ws:
                                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        ws.append_row(
                                            [now, user_name, final_code, "EXPORT_PO", "", "", f"To: {po_sel}", -q_out])
                                        st.toast("Thành công!", icon="✅");
                                        st.success(f"Đã xuất: {final_code}")
                    else:
                        st.error(f"⛔ SAI VẬT TƯ! '{s_sku}' không thuộc lệnh {po_sel}")

        else:  # Xuất Lẻ
            st.markdown("##### 📱 Quét mã vạch")
            scan_type = st.radio("Input:", ["Súng Quét", "Camera"], horizontal=True, label_visibility="collapsed")
            raw_code = None
            if "Súng" in scan_type:
                with st.form("scan_retail"):
                    raw_code = st.text_input("Nhập mã:", key="ret_scan")
                    st.form_submit_button("🔍 Kiểm tra")
            else:
                img_file = st.camera_input("Chụp ảnh")
                if img_file:
                    _, codes = decode_img(img_file.getvalue())
                    if codes: raw_code = codes[0]

            if raw_code:
                st.divider()
                st.markdown(f"### 🔎 Mã: `{raw_code}`")
                qty_out = st.number_input("Số lượng xuất:", min_value=1, value=1)
                final_code = None

                if "|" in raw_code:
                    st.success("✅ Mã hợp lệ!");
                    final_code = raw_code
                else:
                    st.warning("⚠️ Thiếu Batch! Đang tìm lô tồn kho (FEFO)...")
                    sug = get_available_batches(raw_code)
                    if sug:
                        sb = st.selectbox("👉 Chọn lô (Ưu tiên Date cũ):", sug)
                        rb = sb.split(" (")[0]
                        final_code = f"{raw_code}|{rb}"
                        st.info(f"Mã sẽ ghi: {final_code}")
                    else:
                        st.error("❌ Không tìm thấy tồn kho!")

                if final_code:
                    if st.button("🚀 Xác nhận xuất", type="primary"):
                        ws = connect_db("Inventory")
                        if ws:
                            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            ws.append_row([now, user_name, final_code, "EXPORT", "", "", "Xuất Bán Hàng", -qty_out])
                            st.toast("Đã xuất kho!", icon="🚛");
                            st.success("Giao dịch thành công.")

    # ================= MODULE 3: DASHBOARD =================
    elif "Dashboard" in current_tab:
        ws_inv = connect_db("Inventory");
        ws_po = connect_db("Production")
        if ws_inv and ws_po:
            df_inv = pd.DataFrame(ws_inv.get_all_records())
            df_po = pd.DataFrame(ws_po.get_all_records())

            if not df_inv.empty:
                df_inv['Qty'] = pd.to_numeric(df_inv['Qty'], errors='coerce').fillna(0)
                df_inv['Real_Qty'] = df_inv.apply(
                    lambda x: -x['Qty'] if 'EXPORT' in str(x['Action']).upper() else x['Qty'], axis=1)
                df_inv['SKU_Only'] = df_inv['FullCode'].apply(lambda x: x.split('|')[0] if '|' in str(x) else str(x))

                stock_df = df_inv.groupby('SKU_Only')['Real_Qty'].sum().reset_index()
                stock_df.columns = ['SKU', 'Stock_Qty'];
                stock_df = stock_df[stock_df['Stock_Qty'] > 0]

                # KPIs
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📦 Tổng Tồn", f"{int(stock_df['Stock_Qty'].sum()):,}")
                c2.metric("🔖 Loại SKU", len(stock_df))
                c3.metric("🏭 PO Pending", len(df_po[df_po['Status'] == 'Pending']))

                # Cảnh báo Date
                try:
                    df_imp = df_inv[df_inv['Action'] == 'IMPORT'].copy()
                    df_imp['HSD'] = pd.to_datetime(df_imp['HSD'], errors='coerce')
                    near_exp = df_imp[(df_imp['HSD'] > pd.to_datetime(date.today())) & (
                                df_imp['HSD'] <= pd.to_datetime(date.today()) + timedelta(days=30))]
                    c4.metric("⚠️ Cảnh Báo Date", len(near_exp), delta="Ưu tiên xuất", delta_color="inverse")
                except:
                    c4.metric("⚠️ Cảnh Báo Date", 0)

                st.divider()

                # CHARTS
                cc1, cc2 = st.columns([2, 1])
                with cc1:
                    st.markdown("##### 📈 Tồn Kho Theo SKU")
                    if not stock_df.empty:
                        ch = alt.Chart(stock_df).mark_bar().encode(x=alt.X('SKU', sort='-y'), y='Stock_Qty',
                                                                   color='SKU',
                                                                   tooltip=['SKU', 'Stock_Qty']).properties(height=300)
                        st.altair_chart(ch, use_container_width=True)
                with cc2:
                    st.markdown("##### 🍩 Trạng Thái PO")
                    if not df_po.empty:
                        stats = df_po['Status'].value_counts().reset_index();
                        stats.columns = ['Status', 'Count']
                        don = alt.Chart(stats).mark_arc(innerRadius=50).encode(theta='Count', color='Status',
                                                                               tooltip=['Status', 'Count']).properties(
                            height=300)
                        st.altair_chart(don, use_container_width=True)

                # TABLE
                st.markdown("#### 📝 Nhật Ký Hoạt Động")
                df_disp = df_inv.copy()
                df_disp['SKU_V'] = df_disp['FullCode'].apply(
                    lambda x: str(x).split('|')[0] if '|' in str(x) else str(x))
                df_disp['Batch_V'] = df_disp['FullCode'].apply(
                    lambda x: str(x).split('|')[1] if '|' in str(x) else '---')
                st.dataframe(df_disp.sort_values('Timestamp', ascending=False).head(15)[
                                 ['Timestamp', 'SKU_V', 'Batch_V', 'Qty', 'Action', 'User']], use_container_width=True,
                             hide_index=True)

    # ================= MODULE 4: TRUY XUẤT =================
    elif "Truy Xuất" in current_tab:
        st.subheader("🔍 Truy Xuất Nguồn Gốc (Traceability)")
        bq = st.text_input("Nhập Batch cần tra:", placeholder="VD: LOT-1234")
        if bq:
            ws = connect_db("Inventory")
            if ws:
                df = pd.DataFrame(ws.get_all_records())
                td = df[df['FullCode'].astype(str).str.contains(bq, case=False, na=False)].copy()
                if not td.empty:
                    td['Qty'] = pd.to_numeric(td['Qty'], errors='coerce').fillna(0)
                    td['Real_Qty'] = td.apply(lambda x: -x['Qty'] if 'EXPORT' in str(x['Action']).upper() else x['Qty'],
                                              axis=1)
                    td['Timestamp'] = pd.to_datetime(td['Timestamp'], errors='coerce')
                    bal = td['Real_Qty'].sum()

                    st.success(f"🔎 Tìm thấy {len(td)} giao dịch: **{bq}**")
                    col1, col2, col3 = st.columns(3)

                    imps = td[td['Action'] == 'IMPORT'].sort_values('Timestamp')
                    if not imps.empty:
                        col1.metric("Ngày Nhập", imps.iloc[0]['Timestamp'].strftime("%d/%m/%Y"))
                    else:
                        col1.metric("Ngày Nhập", "N/A")

                    col2.metric("HSD", td.iloc[0]['HSD'])
                    col3.metric("Tồn Hiện Tại", f"{bal}", delta="Available" if bal > 0 else "Hết hàng")

                    st.divider()
                    st.subheader("📈 Dòng Chảy Vật Tư")
                    cdata = td.sort_values("Timestamp").copy()
                    cdata['Run_Bal'] = cdata['Real_Qty'].cumsum()

                    base = alt.Chart(cdata).encode(x=alt.X('Timestamp:T', axis=alt.Axis(format='%H:%M %d/%m')))
                    line = base.mark_line(point=True).encode(y='Run_Bal',
                                                             tooltip=['Timestamp', 'Action', 'Qty', 'Run_Bal'])
                    area = base.mark_area(opacity=0.3).encode(y='Run_Bal')
                    st.altair_chart(line + area, use_container_width=True)

                    with st.expander("Chi tiết"):
                        st.dataframe(td[['Timestamp', 'Action', 'Qty', 'User', 'Location']], use_container_width=True)
                else:
                    st.warning("Không tìm thấy dữ liệu!")

elif st.session_state["authentication_status"] is False:
    st.error('Sai mật khẩu!')
elif st.session_state["authentication_status"] is None:
    st.warning('Vui lòng đăng nhập.')