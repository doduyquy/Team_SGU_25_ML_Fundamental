"""
Script để xuất Jupyter Notebook ra PDF
Sử dụng: python export_to_pdf.py
"""

import os
import subprocess
import sys

def export_notebook_to_pdf(notebook_path, output_dir=None):
    """
    Xuất notebook ra PDF sử dụng nbconvert
    """
    if output_dir is None:
        output_dir = os.path.dirname(notebook_path)
    
    # Tên file output
    notebook_name = os.path.splitext(os.path.basename(notebook_path))[0]
    pdf_path = os.path.join(output_dir, f"{notebook_name}.pdf")
    
    print(f"🔄 Đang xuất notebook: {notebook_path}")
    print(f"📄 File PDF sẽ được lưu tại: {pdf_path}")
    
    # Sử dụng nbconvert để xuất ra PDF
    # Option 1: Xuất qua HTML trước (khuyến nghị)
    try:
        # Xuất ra HTML trước
        html_path = os.path.join(output_dir, f"{notebook_name}.html")
        print(f"\n📝 Bước 1: Xuất ra HTML...")
        subprocess.run([
            "jupyter", "nbconvert",
            "--to", "html",
            "--output", os.path.splitext(os.path.basename(html_path))[0],
            notebook_path
        ], check=True, cwd=os.path.dirname(notebook_path))
        print(f"✅ Đã tạo HTML: {html_path}")
        
        # Option: Có thể chuyển HTML sang PDF bằng wkhtmltopdf hoặc weasyprint
        print(f"\n💡 Để chuyển HTML sang PDF, bạn có thể:")
        print(f"   1. Mở file HTML trong trình duyệt")
        print(f"   2. Nhấn Ctrl+P (hoặc Cmd+P) và chọn 'Save as PDF'")
        print(f"   3. Hoặc sử dụng wkhtmltopdf/weasyprint để tự động hóa")
        
        return html_path
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Lỗi khi xuất notebook: {e}")
        return None
    except FileNotFoundError:
        print("❌ Không tìm thấy jupyter nbconvert. Vui lòng cài đặt:")
        print("   pip install jupyter nbconvert")
        return None

def export_to_pdf_direct(notebook_path, output_dir=None):
    """
    Thử xuất trực tiếp ra PDF (cần cài đặt LaTeX)
    """
    if output_dir is None:
        output_dir = os.path.dirname(notebook_path)
    
    notebook_name = os.path.splitext(os.path.basename(notebook_path))[0]
    pdf_path = os.path.join(output_dir, f"{notebook_name}.pdf")
    
    print(f"\n🔄 Thử xuất trực tiếp ra PDF (cần LaTeX)...")
    
    try:
        subprocess.run([
            "jupyter", "nbconvert",
            "--to", "pdf",
            "--output", os.path.splitext(os.path.basename(pdf_path))[0],
            notebook_path
        ], check=True, cwd=os.path.dirname(notebook_path))
        print(f"✅ Đã tạo PDF: {pdf_path}")
        return pdf_path
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Không thể xuất trực tiếp ra PDF (có thể thiếu LaTeX)")
        print(f"   Lỗi: {e}")
        return None
    except FileNotFoundError:
        print("❌ Không tìm thấy jupyter nbconvert")
        return None

if __name__ == "__main__":
    # Đường dẫn đến notebook
    script_dir = os.path.dirname(os.path.abspath(__file__))
    notebook_path = os.path.join(script_dir, "HousePrices_Full_Analysis_Report.ipynb")
    
    if not os.path.exists(notebook_path):
        print(f"❌ Không tìm thấy notebook: {notebook_path}")
        sys.exit(1)
    
    print("="*70)
    print("📊 XUẤT BÁO CÁO RA PDF")
    print("="*70)
    
    # Thử xuất trực tiếp ra PDF
    pdf_path = export_to_pdf_direct(notebook_path)
    
    # Nếu không được, xuất ra HTML
    if pdf_path is None:
        html_path = export_notebook_to_pdf(notebook_path)
        if html_path:
            print(f"\n✅ Đã tạo file HTML. Bạn có thể mở và in ra PDF thủ công.")
    
    print("\n" + "="*70)
    print("✅ Hoàn thành!")

