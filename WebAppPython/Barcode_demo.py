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


def get_available_batches(target_sku):
    """
    Tìm các Batch còn tồn kho của SKU này.
    Sắp xếp theo HSD tăng dần (FEFO - Hết hạn trước xuất trước).
    """
    ws = connect_db("Inventory")
    if not ws: return []

    df = pd.DataFrame(ws.get_all_records())
    if df.empty: return []

    # 1. Tách SKU và Batch từ FullCode
    df['SKU_Only'] = df['FullCode'].apply(lambda x: str(x).split('|')[0] if '|' in str(x) else str(x))
    df['Batch_Only'] = df['FullCode'].apply(lambda x: str(x).split('|')[1] if '|' in str(x) else 'Unknown')

    # 2. Lọc đúng SKU đang cần xuất
    df_sku = df[df['SKU_Only'] == target_sku].copy()

    # 3. Tính tồn kho cho từng Batch
    df_sku['Qty'] = pd.to_numeric(df_sku['Qty'], errors='coerce').fillna(0)
    df_sku['Real_Qty'] = df_sku.apply(lambda x: -x['Qty'] if 'EXPORT' in str(x['Action']).upper() else x['Qty'], axis=1)

    batch_summary = df_sku.groupby('Batch_Only')['Real_Qty'].sum().reset_index()

    # 4. Chỉ lấy Batch nào còn hàng (>0)
    available_batches = batch_summary[batch_summary['Real_Qty'] > 0]['Batch_Only'].tolist()

    # 5. (Nâng cao) Map lại với HSD để sort FEFO
    # Lấy HSD của từng batch từ lệnh IMPORT đầu tiên
    valid_batches_info = []
    for b in available_batches:
        # Tìm dòng nhập của batch này để lấy HSD
        row_info = df[(df['Batch_Only'] == b) & (df['Action'] == 'IMPORT')].head(1)
        if not row_info.empty:
            hsd = row_info.iloc[0]['HSD']
            valid_batches_info.append((b, hsd))

    # Sắp xếp theo HSD (Date nhỏ/gần nhất lên đầu)
    valid_batches_info.sort(key=lambda x: x[1])

    return [f"{b} (HSD: {hsd})" for b, hsd in valid_batches_info]

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

    # === CẤU HÌNH MENU BÊN TRÁI (SIDEBAR) ===
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2554/2554045.png", width=80)
        st.title("WMS PRO")
        st.caption(f"Hello, {user_name} 👋")

        # --- MENU CHÍNH (NGUỒN DUY NHẤT) ---
        # Chỉ khai báo current_tab MỘT LẦN ở đây
        current_tab = st.radio(
            "Chọn Nghiệp Vụ:",
            ["📊 Dashboard", "📥 Nhập Kho (Inbound)", "📤 Xuất Kho (Outbound)", "🔍 Truy Xuất (Traceability)"],
            index=0
        )

        st.divider()
        authenticator.logout('Đăng xuất', 'sidebar')

    # === PHẦN HEADER CỦA TRANG CHÍNH ===
    # Ở ngoài này KHÔNG ĐƯỢC CÓ st.radio nữa
    # Chỉ hiển thị tiêu đề dựa trên cái ông đã chọn trong Sidebar
    st.divider()
    st.title(f"{current_tab}")

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
    st.markdown("""
            <style>
            .main-header {
                font-size: 24px !important; 
                font-weight: 700; 
                color: #2E86C1; /* Màu xanh Vinamilk */
                margin-bottom: 20px;
            }
            /* Ẩn bớt khoảng trắng thừa ở đầu trang Streamlit */
            .block-container {
                padding-top: 2rem; 
            }
            </style>
        """, unsafe_allow_html=True)

    # Hiển thị tiêu đề gọn gàng (Chỉ 1 dòng duy nhất)
    # Ví dụ: "📦 Xuất Kho (Outbound)"
    st.markdown(f'<p class="main-header">{current_tab}</p>', unsafe_allow_html=True)

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
            st.subheader("2. Kết Quả & In Tem")

            # Kiểm tra xem trong phiên làm việc có dữ liệu vừa nhập không
            if 'last_import' in st.session_state:
                info = st.session_state['last_import']

                # --- 1. HIỂN THỊ THÔNG TIN TO RÕ (MỚI) ---
                # Dùng st.success để báo trạng thái và st.metric để hiện số to
                st.success(f"✅ Đã lưu thành công lô: {info['batch']}")

                m1, m2, m3 = st.columns([1.5, 1, 1])
                m1.metric("Sản Phẩm (SKU)", info['sku'])
                m2.metric("Số Batch", info['batch'], delta="Mới nhất")
                m3.metric("Số Lượng", info['qty'])

                st.divider()

                # --- 2. HIỂN THỊ BARCODE ---
                st.markdown("##### 🖨️ Tùy chọn In Ấn:")

                # Tạo barcode từ thông tin session
                img = create_barcode(info['code'])

                if img:
                    # Chia làm 3 cột: [Ảnh Barcode (To)] - [Nút 1] - [Nút 2]
                    # vertical_alignment="center" giúp nút canh giữa theo chiều dọc với ảnh (Streamlit bản mới)
                    col_img, col_btn1, col_btn2 = st.columns([2, 1, 1], gap="small", vertical_alignment="center")

                    with col_img:
                        # Hiển thị ảnh barcode
                        st.image(img, use_column_width=True)
                        st.caption(f"Code: `{info['code']}`")

                    with col_btn1:
                        # Nút In Tem Lẻ
                        if st.button("📦 Tem Thùng", use_container_width=True):
                            try:
                                # ... (Giữ nguyên logic tạo PDF Tem Thùng cũ của ông) ...
                                pdf = FPDF(orientation='L', unit='mm', format=(100, 150))
                                pdf.add_page();
                                pdf.set_font("Helvetica", 'B', 16)
                                pdf.cell(0, 10, txt=remove_accents("TEM LUU KHO"), ln=True, align='C')

                                import tempfile

                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                    img.seek(0);
                                    tmp.write(img.getvalue());
                                    pdf.image(tmp.name, x=10, y=20, w=130)

                                pdf.set_xy(10, 80);
                                pdf.set_font("Helvetica", size=12)
                                pdf.multi_cell(0, 8,
                                               txt=f"SKU: {info['sku']}\nBatch: {info['batch']}\nQty: {info['qty']}")

                                pdf_data = bytes(pdf.output())
                                st.download_button("⬇️ Tải PDF", pdf_data, f"Pallet_{info['batch']}.pdf")
                            except Exception as e:
                                st.error(str(e))

                    with col_btn2:
                        # Nút In Tem Loạt
                        if st.button(f"🏷️ In {info['qty']} Tem", use_container_width=True):
                            try:
                                # ... (Giữ nguyên logic In Loạt cũ của ông) ...
                                with st.spinner("Rendering..."):
                                    pdf_bulk = FPDF(orientation='P', unit='mm', format='A4')
                                    pdf_bulk.set_auto_page_break(auto=False, margin=0);
                                    pdf_bulk.add_page()
                                    mx, my, cw, rh = 12, 12, 62, 40
                                    cols, rows = 3, 7
                                    x, y, cx, cy = mx, my, 0, 0

                                    import tempfile

                                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_b:
                                        img.seek(0);
                                        tmp_b.write(img.getvalue());
                                        t_path = tmp_b.name

                                    for i in range(int(info['qty'])):
                                        pdf_bulk.image(t_path, x=x + 2, y=y + 5, w=cw - 4)
                                        pdf_bulk.set_font("Helvetica", size=8)
                                        pdf_bulk.set_xy(x, y + rh - 8)
                                        pdf_bulk.cell(cw, 5, txt=f"{info['sku']} | {info['batch']}", align='C')
                                        cx += 1
                                        if cx < cols:
                                            x += cw
                                        else:
                                            cx = 0; x = mx; cy += 1; y += rh;
                                        if cy >= rows: pdf_bulk.add_page(); cy = 0; y = my; x = mx

                                    try:
                                        bulk_bytes = bytes(pdf_bulk.output())
                                    except:
                                        bulk_bytes = pdf_bulk.output(dest='S').encode('latin-1')

                                    st.download_button("⬇️ Tải A4", bulk_bytes, f"Bulk_{info['batch']}.pdf")
                            except Exception as e:
                                st.error(str(e))
                            finally:
                                pass
            else:
                # Khi chưa nhập gì thì hiện thông báo chờ
                st.info("👈 Vui lòng nhập thông tin lô hàng và bấm 'Lưu Kho' bên trái.")
                st.image("https://cdn-icons-png.flaticon.com/512/1466/1466668.png", width=100,
                         caption="Waiting for data...")

    # ================= MODULE 2: XUẤT KHO (NÂNG CẤP PO) =================
    elif "Xuất Kho" in current_tab:
        st.caption("Chế độ xuất kho thông minh hỗ trợ FEFO & Kiểm soát theo Lệnh sản xuất.")
        mode = st.radio("Chế độ:", ["🚀 Xuất Lẻ (Thông thường)", "🏭 Xuất Cho Sản Xuất (Theo PO)"], horizontal=True)
        st.divider()

        # --- MODE A: XUẤT SẢN XUẤT (NEW FEATURE) ---
        if "Theo PO" in mode:
            col_po_info, col_po_scan = st.columns([1, 2])

            with col_po_info:
                st.markdown("##### 1. Chọn Lệnh Sản Xuất")
                po_sel = st.selectbox("Danh sách PO:", list(MOCK_DB_PO.keys()), label_visibility="collapsed")
                po_data = MOCK_DB_PO[po_sel]

                st.info(f"🏷️ Sản phẩm: **{po_data['Product']}**")

                # Hiển thị bảng định mức (BOM)
                st.write("**📋 Định mức vật tư (BOM):**")
                bom_df = pd.DataFrame(list(po_data['BOM'].items()), columns=['Nguyên Liệu', 'Định Mức (Kg)'])
                st.dataframe(bom_df, use_container_width=True, hide_index=True)

            with col_po_scan:
                st.markdown(f"##### 2. Quét Nguyên Liệu cho: `{po_sel}`")

                # Ô quét mã (Có form để enter tiện hơn)
                with st.form("po_scan_form"):
                    raw_scan = st.text_input("Quét mã nguyên liệu tại đây:", placeholder="Quét mã SKU hoặc FullCode...")
                    btn_check_po = st.form_submit_button("🔍 Kiểm tra & Đối chiếu")

                if raw_scan:
                    # 1. Xử lý tách mã (SKU và Batch)
                    if "|" in raw_scan:
                        scan_sku, scan_batch = raw_scan.split("|")
                        scan_full_code = raw_scan
                        is_batch_selected = True
                    else:
                        scan_sku = raw_scan
                        scan_batch = None
                        scan_full_code = None
                        is_batch_selected = False

                    # 2. KIỂM TRA BOM (QUAN TRỌNG NHẤT)
                    # Xem SKU vừa quét có nằm trong công thức của PO này không
                    if scan_sku in po_data['BOM']:
                        target_qty = po_data['BOM'][scan_sku]
                        st.success(f"✅ ĐÚNG NGUYÊN LIỆU: **{scan_sku}**")

                        # Hiển thị thanh tiến độ giả lập (Cho đẹp)
                        st.progress(0, text=f"Cần cấp: {target_qty} Kg")

                        # 3. KIỂM TRA BATCH (FEFO LOGIC)
                        final_code_to_export = None

                        if is_batch_selected:
                            # Nếu quét mã full -> Dùng luôn
                            final_code_to_export = scan_full_code
                            st.caption(f"🎯 Đã xác định lô: {scan_batch}")
                        else:
                            # Nếu quét mã thiếu -> Bắt chọn lô (Giống bên xuất lẻ)
                            st.warning(f"⚠️ Mã `{scan_sku}` chưa có thông tin Lô (Batch). Vui lòng chọn bên dưới:")
                            suggested = get_available_batches(scan_sku)

                            if suggested:
                                sel_batch = st.selectbox("👉 Chọn lô xuất (Ưu tiên Date cũ):", suggested)
                                real_batch = sel_batch.split(" (")[0]
                                final_code_to_export = f"{scan_sku}|{real_batch}"
                            else:
                                st.error("❌ Hết hàng tồn kho cho mã này!")

                        # 4. FORM XÁC NHẬN XUẤT
                        if final_code_to_export:
                            st.divider()
                            c_qty, c_btn = st.columns([1, 1])
                            with c_qty:
                                qty_out_po = st.number_input("Khối lượng thực xuất (Kg):",
                                                             min_value=1, value=int(target_qty))
                            with c_btn:
                                st.write("")  # Spacer
                                st.write("")
                                if st.button("🚀 Cấp Hàng Cho SX", type="primary"):
                                    ws = connect_db("Inventory")
                                    if ws:
                                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        # Ghi log Action là EXPORT_PO để sau này lọc báo cáo
                                        ws.append_row([
                                            now,
                                            user_name,
                                            final_code_to_export,
                                            "EXPORT_PO",
                                            "",
                                            "",
                                            f"To: {po_sel}",  # Location đích là mã PO
                                            -qty_out_po
                                        ])
                                        st.toast(f"Đã cấp {qty_out_po}kg cho {po_sel}", icon="🏭")
                                        st.success(f"Đã ghi nhận giao dịch: {final_code_to_export}")

                    else:
                        # NẾU QUÉT SAI HÀNG (KHÔNG CÓ TRONG BOM)
                        st.error(f"⛔ CẢNH BÁO SAI VẬT TƯ!")
                        st.markdown(f"""
                                Nguyên liệu **{scan_sku}** KHÔNG có trong công thức của **{po_sel}**.
                                \n👉 Vui lòng kiểm tra lại phiếu Lệnh Sản Xuất.
                                """)

        # --- MODE B: XUẤT LẺ (CẬP NHẬT LOGIC CHẶN LỖI) ---
        else:
            st.write("📱 **Quét mã vạch:**")
            scan_method = st.radio("Input:", ["Súng Quét", "Camera"], horizontal=True,
                                   label_visibility="collapsed")

            raw_code = None
            if "Súng" in scan_method:
                # Dùng form để Enter không bị reload trang mất dữ liệu
                with st.form("scan_form"):
                    raw_code = st.text_input("Nhập/Quét mã:", key="manual_scan")
                    submitted = st.form_submit_button("🔍 Kiểm tra")
            else:
                img_file = st.camera_input("Chụp mã")
                if img_file:
                    _, codes = decode_img(img_file.getvalue())
                    if codes: raw_code = codes[0]

            # --- LOGIC XỬ LÝ MÃ ---
            if raw_code:
                st.divider()
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown(f"### 🔎 Mã quét: `{raw_code}`")
                with c2:
                    # 1. THÊM Ô NHẬP SỐ LƯỢNG (Để không phải xuất từng cái)
                    qty_out = st.number_input("Số lượng xuất:", min_value=1, value=1, step=1)

                # Biến này sẽ hứng giá trị cuối cùng để ghi vào DB
                final_full_code = None

                # --- TRƯỜNG HỢP 1: MÃ CHUẨN (Đã có dấu |) ---
                if "|" in raw_code:
                    sku, batch = raw_code.split("|")
                    st.success(f"✅ Mã hợp lệ! Sản phẩm: {sku} - Lô: {batch}")
                    final_full_code = raw_code  # <--- ĐỊNH NGHĨA Ở ĐÂY

                # --- TRƯỜNG HỢP 2: MÃ THIẾU (Chỉ có SKU) ---
                else:
                    st.warning(f"⚠️ Mã `{raw_code}` thiếu thông tin Lô (Batch)!")

                    # Gọi hàm tìm batch gợi ý (FEFO)
                    suggested_batches = get_available_batches(raw_code)

                    if suggested_batches:
                        # Bắt buộc chọn lô
                        selected_batch_info = st.selectbox("👉 Chọn Lô cần xuất (Ưu tiên Date cũ):", suggested_batches)

                        # Tách lấy mã batch thật (Bỏ phần HSD trong ngoặc đơn đi)
                        real_batch = selected_batch_info.split(" (")[0]

                        # Ghép lại thành mã chuẩn
                        final_full_code = f"{raw_code}|{real_batch}"  # <--- ĐỊNH NGHĨA Ở ĐÂY
                        st.info(f"Mã sẽ ghi nhận: **{final_full_code}**")
                    else:
                        st.error(f"❌ Không tìm thấy tồn kho nào cho mã '{raw_code}'!")

                # --- NÚT XÁC NHẬN (Chỉ hiện khi đã có final_full_code) ---
                if final_full_code:
                    st.divider()
                    if st.button("🚀 Xác nhận xuất kho", type="primary"):
                        ws = connect_db("Inventory")
                        if ws:
                            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            # Ghi vào DB: dùng final_full_code và số lượng âm (-qty_out)
                            ws.append_row([
                                now,
                                user_name,
                                final_full_code,
                                "EXPORT",
                                "",
                                "",
                                "Xuất Bán Hàng",  # Đổi tên Label ở đây
                                -qty_out  # Trừ số lượng
                            ])
                            st.toast(f"Đã xuất {qty_out} sản phẩm!", icon="🚛")
                            st.success(f"Đã lưu giao dịch: {final_full_code} | SL: -{qty_out}")

    # ================= MODULE 3: DASHBOARD =================
    elif "Dashboard" in current_tab:
        st.subheader("📊 Dashboard Quản Trị Kho Vận")

        # 1. TẢI DỮ LIỆU TỪ 2 NGUỒN
        ws_inv = connect_db("Inventory")
        ws_po = connect_db("Production")

        if ws_inv and ws_po:
            with st.spinner("Đang tổng hợp dữ liệu kho..."):
                # Load Dataframes
                df_inv = pd.DataFrame(ws_inv.get_all_records())
                df_po = pd.DataFrame(ws_po.get_all_records())

                if df_inv.empty:
                    st.warning("Chưa có dữ liệu kho!")
                    st.stop()

                # --- XỬ LÝ SỐ LIỆU (AGGREGATION LOGIC) ---
                # Chuyển đổi cột Qty sang số (đề phòng lỗi string)
                df_inv['Qty'] = pd.to_numeric(df_inv['Qty'], errors='coerce').fillna(0)

                # Logic: Nếu Action là EXPORT hoặc EXPORT_PO thì nhân -1 để trừ kho
                # (Giả sử trong Sheet ông đang lưu số dương cho cả 2 hành động)
                df_inv['Real_Qty'] = df_inv.apply(
                    lambda x: -x['Qty'] if 'EXPORT' in str(x['Action']).upper() else x['Qty'], axis=1
                )

                # Tách SKU từ FullCode (VNM-A|LOT-1 -> VNM-A)
                df_inv['SKU_Only'] = df_inv['FullCode'].apply(lambda x: x.split('|')[0] if '|' in str(x) else str(x))

                # TÍNH TỒN KHO THỰC TẾ (Stock on Hand)
                stock_df = df_inv.groupby('SKU_Only')['Real_Qty'].sum().reset_index()
                stock_df.columns = ['SKU', 'Stock_Qty']
                stock_df = stock_df[stock_df['Stock_Qty'] > 0]  # Chỉ lấy hàng còn tồn

                # TÍNH CÁC CHỈ SỐ KPI
                total_items = stock_df['Stock_Qty'].sum()
                total_skus = len(stock_df)
                po_pending = len(df_po[df_po['Status'] == 'Pending'])

                # Cảnh báo Date (Giả lập check logic HSD từ df_inv)
                # Lấy các lô nhập (IMPORT) và check HSD so với hôm nay
                df_imports = df_inv[df_inv['Action'] == 'IMPORT'].copy()
                try:
                    df_imports['HSD'] = pd.to_datetime(df_imports['HSD'], errors='coerce')
                    today = pd.to_datetime(datetime.now().date())
                    # Lọc lô sắp hết hạn trong 30 ngày
                    near_exp = df_imports[
                        (df_imports['HSD'] > today) & (df_imports['HSD'] <= today + timedelta(days=30))]
                    warning_count = len(near_exp)
                except:
                    warning_count = 0

                # --- GIAO DIỆN HIỂN THỊ (UI/UX) ---

                # ROW 1: METRIC CARDS
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("📦 Tổng Tồn Kho (Unit)", f"{int(total_items):,}", delta="Real-time")
                c2.metric("🔖 Số loại SKU", total_skus, help="Số mã hàng đang quản lý")
                c3.metric("🏭 Lệnh SX Chờ (Pending)", po_pending, delta=f"-{len(df_po[df_po['Status'] == 'Done'])} Done",
                          delta_color="inverse")
                c4.metric("⚠️ Cảnh Báo Date (30d)", warning_count, delta="Ưu tiên xuất", delta_color="inverse")

                st.divider()

                # ROW 2: BIỂU ĐỒ PHÂN TÍCH (CHARTS)
                col_chart1, col_chart2 = st.columns([2, 1])

                with col_chart1:
                    st.markdown("##### 📈 Phân Bố Tồn Kho Theo SKU")
                    if not stock_df.empty:
                        # Biểu đồ cột dùng Altair
                        chart_bar = alt.Chart(stock_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                            x=alt.X('SKU', sort='-y', title=None),
                            y=alt.Y('Stock_Qty', title='Số lượng tồn'),
                            color=alt.Color('SKU', legend=None),
                            tooltip=['SKU', 'Stock_Qty']
                        ).properties(height=300)
                        st.altair_chart(chart_bar, use_container_width=True)
                    else:
                        st.info("Kho đang trống.")

                with col_chart2:
                    st.markdown("##### 🍩 Tỷ Lệ Trạng Thái PO")
                    if not df_po.empty:
                        # Biểu đồ tròn (Donut chart)
                        po_stats = df_po['Status'].value_counts().reset_index()
                        po_stats.columns = ['Status', 'Count']

                        chart_donut = alt.Chart(po_stats).mark_arc(innerRadius=50).encode(
                            theta=alt.Theta(field="Count", type="quantitative"),
                            color=alt.Color(field="Status", type="nominal"),
                            tooltip=['Status', 'Count']
                        ).properties(height=300)
                        st.altair_chart(chart_donut, use_container_width=True)

                st.divider()

                # ROW 3: CHI TIẾT GIAO DỊCH GẦN NHẤT & PO
                t1, t2 = st.tabs(["📝 Nhật Ký Kho (Gần nhất)", "🏭 Tiến Độ Sản Xuất"])

                with t1:
                    st.markdown("#### 📝 Nhật Ký Hoạt Động Chi Tiết")

                    # Tạo bản sao để xử lý hiển thị (không ảnh hưởng logic tính toán)
                    df_display = df_inv.copy()

                    # 1. Tách FullCode thành SKU và Batch riêng biệt cho dễ nhìn
                    # Logic: Nếu có dấu "|", tách ra. Nếu không, để Batch là trống
                    df_display['SKU_View'] = df_display['FullCode'].apply(
                        lambda x: str(x).split('|')[0] if '|' in str(x) else str(x))
                    df_display['Batch_View'] = df_display['FullCode'].apply(
                        lambda x: str(x).split('|')[1] if '|' in str(x) else '---')

                    # 2. Làm đẹp định dạng thời gian (Bỏ phần giây thừa thãi nếu muốn)
                    # df_display['Time_View'] = pd.to_datetime(df_display['Timestamp']).dt.strftime('%H:%M %d/%m/%Y')

                    # 3. Sắp xếp lại thứ tự cột cho thuận mắt Manager
                    cols_order = ['Timestamp', 'SKU_View', 'Batch_View', 'Qty', 'Location', 'Action', 'User']

                    # Lấy 15 dòng mới nhất
                    final_table = df_display.sort_values(by='Timestamp', ascending=False).head(15)[cols_order]

                    # 4. Hiển thị bảng với tên cột Tiếng Việt đẹp đẽ
                    st.dataframe(
                        final_table,
                        column_config={
                            "Timestamp": st.column_config.DatetimeColumn("Thời Gian", format="D/M/YYYY h:mm a"),
                            "SKU_View": "Sản Phẩm (SKU)",
                            "Batch_View": st.column_config.TextColumn("Số Lô (Batch)", help="Mã định danh lô hàng"),
                            "Qty": st.column_config.NumberColumn("Số Lượng", format="%d"),
                            "Location": "Vị Trí",
                            "Action": st.column_config.TextColumn("Hành Động", width="small"),
                            "User": "Người Nhập"
                        },
                        use_container_width=True,
                        hide_index=True
                    )

                with t2:
                    # Hiển thị bảng PO với định dạng màu sắc cho Status
                    def highlight_status(val):
                        color = '#d4edda' if val == 'Done' else '#fff3cd' if val == 'Pending' else '#cce5ff'
                        return f'background-color: {color}'


                    st.dataframe(
                        df_po.style.applymap(highlight_status, subset=['Status']),
                        use_container_width=True
                    )

        else:
            st.error("Mất kết nối với Google Sheets!")
        # ================= MODULE 4: TRUY XUẤT NGUỒN GỐC (ISO/HACCP) =================
    elif "Truy Xuất" in current_tab:
        st.subheader("🔍 Truy Xuất Nguồn Gốc (Traceability)")

        # 1. Input tìm kiếm
        batch_query = st.text_input("Nhập số Batch cần tra (VD: LOT-3854):", placeholder="Nhập mã lô...")

        if batch_query:
            ws = connect_db("Inventory")
            if ws:
                # Lấy dữ liệu
                df = pd.DataFrame(ws.get_all_records())

                # Lọc theo Batch (Tìm tương đối)
                # Chuyển FullCode thành string để tránh lỗi
                trace_data = df[df['FullCode'].astype(str).str.contains(batch_query, case=False, na=False)].copy()

                if not trace_data.empty:
                    # --- XỬ LÝ SỐ LIỆU AN TOÀN ---
                    # 1. Convert Qty sang số
                    trace_data['Qty'] = pd.to_numeric(trace_data['Qty'], errors='coerce').fillna(0)
                    # 2. Convert Timestamp sang dạng Thời gian chuẩn
                    trace_data['Timestamp'] = pd.to_datetime(trace_data['Timestamp'], errors='coerce')

                    # 3. Tính Real_Qty (Số âm dương)
                    trace_data['Real_Qty'] = trace_data.apply(
                        lambda x: -x['Qty'] if 'EXPORT' in str(x['Action']).upper() else x['Qty'], axis=1
                    )

                    # 4. Tính tồn kho hiện tại
                    balance = trace_data['Real_Qty'].sum()

                    # --- HIỂN THỊ THÔNG TIN (FIX LỖI CRASH KHI KHÔNG CÓ IMPORT) ---
                    st.success(f"🔎 Tìm thấy {len(trace_data)} giao dịch liên quan đến: **{batch_query}**")

                    # Tìm thông tin nhập kho (Nếu có)
                    imports = trace_data[trace_data['Action'] == 'IMPORT'].sort_values('Timestamp')

                    col_info1, col_info2, col_info3 = st.columns(3)

                    # Cột 1: Ngày nhập (Xử lý trường hợp không có Import)
                    if not imports.empty:
                        first_date = imports.iloc[0]['Timestamp'].strftime("%d/%m/%Y")
                        col_info1.metric("Ngày Nhập Kho", first_date)
                    else:
                        col_info1.metric("Ngày Nhập Kho", "N/A", "Chưa có dữ liệu nhập", delta_color="off")

                    # Cột 2: Hạn sử dụng (Lấy dòng mới nhất bất kỳ để check HSD)
                    any_hsd = trace_data.iloc[0]['HSD']
                    col_info2.metric("Hạn Sử Dụng", any_hsd)

                    # Cột 3: Tồn kho
                    col_info3.metric("Tồn Hiện Tại", f"{balance} Unit",
                                     delta="Available" if balance > 0 else "Negative/Sold",
                                     delta_color="normal" if balance >= 0 else "inverse")

                    st.divider()

                    # --- VẼ BIỂU ĐỒ (VISUALIZATION) ---
                    st.subheader("📈 Biểu Đồ Biến Động Số Dư")

                    # Sắp xếp theo thời gian cũ -> mới để vẽ đường đi
                    chart_data = trace_data.sort_values("Timestamp").copy()

                    # Tính lũy kế (Running Balance)
                    chart_data['Running_Balance'] = chart_data['Real_Qty'].cumsum()

                    # Vẽ biểu đồ
                    base = alt.Chart(chart_data).encode(
                        x=alt.X('Timestamp:T', title='Thời gian', axis=alt.Axis(format='%H:%M %d/%m')))

                    line = base.mark_line(point=True, strokeWidth=3).encode(
                        y=alt.Y('Running_Balance:Q', title='Tồn kho'),
                        tooltip=['Timestamp', 'Action', 'Qty', 'Running_Balance', 'User']
                    )

                    # Tô màu vùng dưới
                    area = base.mark_area(opacity=0.3, color='lightblue').encode(
                        y='Running_Balance:Q'
                    )

                    st.altair_chart(area + line, use_container_width=True)

                    # --- BẢNG CHI TIẾT ---
                    with st.expander("Xem chi tiết các dòng giao dịch"):
                        st.dataframe(trace_data[['Timestamp', 'Action', 'Qty', 'User', 'Location', 'FullCode']],
                                     use_container_width=True)

                else:
                    st.warning(f"⚠️ Không tìm thấy dữ liệu nào chứa mã: '{batch_query}'")
elif st.session_state["authentication_status"] is False:
    st.error('Sai mật khẩu!')
elif st.session_state["authentication_status"] is None:
    st.warning('Vui lòng đăng nhập.')