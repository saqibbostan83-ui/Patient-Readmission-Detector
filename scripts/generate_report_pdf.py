"""Generate PDF from markdown report using markdown + xhtml2pdf."""
import sys
from pathlib import Path
import markdown
from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "reports" / "FINAL_PROJECT_REPORT.md"
PDF_PATH = ROOT / "reports" / "FINAL_PROJECT_REPORT.pdf"

def convert_md_to_pdf():
    if not MD_PATH.exists():
        print(f"Report not found at {MD_PATH}")
        return False
    md_text = MD_PATH.read_text(encoding="utf-8")
    html_content = markdown.markdown(md_text, extensions=["tables", "fenced_code", "nl2br"])
    
    # Simple CSS styling suitable for academic research report
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        @page {{
            size: a4 portrait;
            margin: 2cm;
            @bottom-right {{
                content: counter(page);
            }}
        }}
        body {{
            font-family: Helvetica, Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.5;
            color: #222222;
        }}
        h1 {{
            font-size: 18pt;
            color: #1b365d;
            border-bottom: 2px solid #1b365d;
            padding-bottom: 6px;
            margin-top: 15px;
        }}
        h2 {{
            font-size: 14pt;
            color: #2b5c8f;
            border-bottom: 1px solid #ddd;
            padding-bottom: 4px;
            margin-top: 18px;
        }}
        h3 {{
            font-size: 11pt;
            color: #333333;
            margin-top: 12px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 8.5pt;
        }}
        th, td {{
            border: 1px solid #cccccc;
            padding: 5px 7px;
            text-align: left;
        }}
        th {{
            background-color: #f2f5f8;
            color: #1b365d;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #fafbfc;
        }}
        code {{
            font-family: "Courier New", Courier, monospace;
            background-color: #f4f4f4;
            padding: 1px 4px;
            font-size: 8.5pt;
        }}
        pre {{
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            padding: 8px;
            font-size: 8pt;
            white-space: pre-wrap;
        }}
        blockquote {{
            border-left: 3px solid #2b5c8f;
            margin: 10px 0;
            padding-left: 10px;
            color: #555555;
            font-style: italic;
        }}
    </style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """
    
    with open(PDF_PATH, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(styled_html, dest=pdf_file)
        
    if pisa_status.err:
        print(f"Error generating PDF: {pisa_status.err}")
        return False
    print(f"Successfully generated {PDF_PATH} ({PDF_PATH.stat().st_size:,} bytes)")
    return True

if __name__ == "__main__":
    convert_md_to_pdf()
