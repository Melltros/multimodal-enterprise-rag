"""
Utility: create_sample_pdf.py
Description: Generates a realistic 4-page enterprise quarterly financial report PDF
             using PyMuPDF for testing and evaluation of the RAG pipeline.
"""

from pathlib import Path
import pymupdf as fitz


def generate_sample_financial_report(output_path: str = "data/sample_report.pdf") -> Path:
    """
    Creates a realistic 4-page financial report PDF with text and tables.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    doc = fitz.open()

    # Define color palette
    primary_color = (0.08, 0.22, 0.40)     # Navy blue
    accent_color = (0.20, 0.45, 0.65)      # Slate blue
    text_color = (0.15, 0.15, 0.15)        # Dark grey
    table_header_bg = (0.90, 0.93, 0.97)   # Light blue fill
    border_color = (0.75, 0.80, 0.85)

    # ---------------------------------------------------------
    # PAGE 1: Executive Summary & Corporate Highlights
    # ---------------------------------------------------------
    page1 = doc.new_page(width=595, height=842)  # A4

    # Header Banner
    page1.draw_rect(fitz.Rect(40, 40, 555, 95), color=primary_color, fill=primary_color)
    page1.insert_text(fitz.Point(55, 68), "APEX GLOBAL TECHNOLOGIES INC.", fontsize=18, color=(1, 1, 1), fontname="hebo")
    page1.insert_text(fitz.Point(55, 85), "Q3 Fiscal Year 2025 Financial Results & Shareholder Report", fontsize=11, color=(0.85, 0.90, 0.95), fontname="helv")

    # Content
    y = 125
    page1.insert_text(fitz.Point(40, y), "1. Executive Summary", fontsize=14, color=primary_color, fontname="hebo")
    y += 20
    summary_text = (
        "Apex Global Technologies Inc. (NASDAQ: AGTX) today reported strong financial performance for the third "
        "quarter ended September 30, 2025. Total consolidated revenue reached record levels, driven by accelerated "
        "adoption of our Enterprise Cloud Platform and Next-Gen AI Infrastructure products. "
        "The company expanded gross margins by 210 basis points year-over-year while continuing targeted investments in "
        "research and development."
    )
    rect = fitz.Rect(40, y, 555, y + 65)
    page1.insert_textbox(rect, summary_text, fontsize=10, color=text_color, fontname="helv")

    y += 80
    page1.insert_text(fitz.Point(40, y), "Key Operational Highlights:", fontsize=11, color=accent_color, fontname="hebo")
    y += 18
    highlights = [
        "- Enterprise customer count grew to 4,820 accounts, up 22% year-over-year.",
        "- Cloud Annual Recurring Revenue (ARR) surpassed $3.10 billion, representing 64% of total revenues.",
        "- Successfully closed the acquisition of DataStream Analytics for $420 million in cash.",
        "- Operating cash flow for the quarter was $1.38 billion, compared to $1.02 billion in Q3 2024.",
        "- Repurchased 4.5 million shares of common stock for an aggregate value of $250 million."
    ]
    for h in highlights:
        page1.insert_text(fitz.Point(50, y), h, fontsize=9.5, color=text_color, fontname="helv")
        y += 16

    # Page footer
    page1.insert_text(fitz.Point(220, 810), "Page 1 of 4 | Apex Global Tech Q3 FY2025", fontsize=8, color=(0.5, 0.5, 0.5), fontname="helv")

    # ---------------------------------------------------------
    # PAGE 2: Segment Breakdown & Detailed Revenue Performance
    # ---------------------------------------------------------
    page2 = doc.new_page(width=595, height=842)
    page2.draw_rect(fitz.Rect(40, 40, 555, 70), color=primary_color, fill=primary_color)
    page2.insert_text(fitz.Point(55, 60), "Segment Performance & Revenue Breakdown", fontsize=14, color=(1, 1, 1), fontname="hebo")

    y = 95
    page2.insert_text(fitz.Point(40, y), "2. Revenue by Business Segment (in Millions USD)", fontsize=12, color=primary_color, fontname="hebo")
    y += 20

    # Draw Segment Table
    table_headers = ["Segment / Division", "Q3 2025", "Q3 2024", "YoY Growth", "% of Total"]
    table_rows = [
        ["Enterprise Cloud Services", "$2,150", "$1,780", "+20.8%", "44.6%"],
        ["AI Infrastructure & Hardware", "$1,420", "$1,120", "+26.8%", "29.5%"],
        ["Cybersecurity Solutions", "$780", "$710", "+9.9%", "16.2%"],
        ["Professional Advisory", "$470", "$610", "-23.0%", "9.7%"],
        ["Total Consolidated Revenue", "$4,820", "$4,220", "+14.2%", "100.0%"]
    ]

    col_widths = [180, 80, 80, 85, 90]
    row_height = 22

    # Draw Header
    x_curr = 40
    page2.draw_rect(fitz.Rect(40, y, 555, y + row_height), color=border_color, fill=table_header_bg)
    for col_idx, h_text in enumerate(table_headers):
        page2.insert_text(fitz.Point(x_curr + 5, y + 15), h_text, fontsize=9, color=primary_color, fontname="hebo")
        x_curr += col_widths[col_idx]
    y += row_height

    # Draw Rows
    for row_idx, row in enumerate(table_rows):
        x_curr = 40
        is_total = row_idx == len(table_rows) - 1
        row_bg = (0.96, 0.97, 0.99) if row_idx % 2 == 1 else (1, 1, 1)
        if is_total:
            row_bg = (0.92, 0.95, 0.98)
        page2.draw_rect(fitz.Rect(40, y, 555, y + row_height), color=border_color, fill=row_bg)
        for col_idx, cell_text in enumerate(row):
            fname = "hebo" if is_total else "helv"
            page2.insert_text(fitz.Point(x_curr + 5, y + 15), cell_text, fontsize=9, color=primary_color if is_total else text_color, fontname=fname)
            x_curr += col_widths[col_idx]
        y += row_height

    y += 30
    page2.insert_text(fitz.Point(40, y), "Segment Commentary:", fontsize=11, color=accent_color, fontname="hebo")
    y += 18
    commentary = (
        "Enterprise Cloud Services continued to be our primary revenue engine, fueled by expansion among Fortune 500 "
        "customers. AI Infrastructure demonstrated the fastest expansion (+26.8% YoY) due to hyper-scaler demand for "
        "GPU-accelerated compute clusters."
    )
    page2.insert_textbox(fitz.Rect(40, y, 555, y + 50), commentary, fontsize=9.5, color=text_color, fontname="helv")

    page2.insert_text(fitz.Point(220, 810), "Page 2 of 4 | Apex Global Tech Q3 FY2025", fontsize=8, color=(0.5, 0.5, 0.5), fontname="helv")

    # ---------------------------------------------------------
    # PAGE 3: Operating Expenses & Capital Allocation
    # ---------------------------------------------------------
    page3 = doc.new_page(width=595, height=842)
    page3.draw_rect(fitz.Rect(40, 40, 555, 70), color=primary_color, fill=primary_color)
    page3.insert_text(fitz.Point(55, 60), "Operating Expenses & Capital Allocation", fontsize=14, color=(1, 1, 1), fontname="hebo")

    y = 95
    page3.insert_text(fitz.Point(40, y), "3. Operating Expenses Breakdown (in Millions USD)", fontsize=12, color=primary_color, fontname="hebo")
    y += 20

    exp_headers = ["Expense Category", "Q3 2025", "Q3 2024", "% Change", "Strategic Rationale"]
    exp_rows = [
        ["Research & Development", "$980", "$820", "+19.5%", "Investment in frontier AI models"],
        ["Sales & Marketing", "$850", "$790", "+7.6%", "Direct enterprise sales force expansion"],
        ["General & Administrative", "$320", "$310", "+3.2%", "Process automation cost controls"],
        ["Restructuring & Other", "$45", "$15", "+200.0%", "DataStream consolidation costs"],
        ["Total Operating Expenses", "$2,195", "$1,935", "+13.4%", "Disciplined operational scaling"]
    ]
    exp_widths = [140, 70, 70, 70, 165]

    # Draw Header
    x_curr = 40
    page3.draw_rect(fitz.Rect(40, y, 555, y + row_height), color=border_color, fill=table_header_bg)
    for col_idx, h_text in enumerate(exp_headers):
        page3.insert_text(fitz.Point(x_curr + 5, y + 15), h_text, fontsize=8.5, color=primary_color, fontname="hebo")
        x_curr += exp_widths[col_idx]
    y += row_height

    # Draw Rows
    for row_idx, row in enumerate(exp_rows):
        x_curr = 40
        is_total = row_idx == len(exp_rows) - 1
        row_bg = (0.96, 0.97, 0.99) if row_idx % 2 == 1 else (1, 1, 1)
        if is_total:
            row_bg = (0.92, 0.95, 0.98)
        page3.draw_rect(fitz.Rect(40, y, 555, y + row_height), color=border_color, fill=row_bg)
        for col_idx, cell_text in enumerate(row):
            fname = "hebo" if is_total else "helv"
            page3.insert_text(fitz.Point(x_curr + 5, y + 15), cell_text, fontsize=8.5, color=primary_color if is_total else text_color, fontname=fname)
            x_curr += exp_widths[col_idx]
        y += row_height

    page3.insert_text(fitz.Point(220, 810), "Page 3 of 4 | Apex Global Tech Q3 FY2025", fontsize=8, color=(0.5, 0.5, 0.5), fontname="helv")

    # ---------------------------------------------------------
    # PAGE 4: Consolidated Statements & Q3 Net Revenue Metrics
    # ---------------------------------------------------------
    page4 = doc.new_page(width=595, height=842)
    page4.draw_rect(fitz.Rect(40, 40, 555, 70), color=primary_color, fill=primary_color)
    page4.insert_text(fitz.Point(55, 60), "Consolidated Financial Summary & Q3 Net Results", fontsize=14, color=(1, 1, 1), fontname="hebo")

    y = 95
    page4.insert_text(fitz.Point(40, y), "4. Consolidated Financial Summary & Net Results", fontsize=12, color=primary_color, fontname="hebo")
    y += 20

    net_results_text = (
        "According to the consolidated financial statements for the third quarter of FY2025 ended September 30, 2025:\n\n"
        "• Q3 Net Revenue: The company achieved Q3 Net Revenue of $4.82 billion (an increase of $600 million or 14.2% YoY from $4.22 billion in Q3 2024).\n"
        "• Gross Profit: Total gross profit reached $3.345 billion, with a gross margin of 69.4%.\n"
        "• Operating Income: Operating income for Q3 stood at $1.150 billion, representing an operating margin of 23.9%.\n"
        "• Net Income: GAAP net income was $940 million, or $1.42 per diluted share, compared to $780 million ($1.18 per share) in the prior-year period.\n"
        "• Cash & Marketable Securities: Total liquidity closed at $8.45 billion as of September 30, 2025."
    )
    rect4 = fitz.Rect(40, y, 555, y + 140)
    page4.insert_textbox(rect4, net_results_text, fontsize=9.5, color=text_color, fontname="helv")
    y += 150

    # Summary Table on Page 4
    p4_headers = ["Key Metric", "Q3 2025 ($B)", "Q3 2024 ($B)", "YoY Growth (%)"]
    p4_rows = [
        ["Net Revenue", "$4.820", "$4.220", "+14.2%"],
        ["Gross Profit", "$3.345", "$2.840", "+17.8%"],
        ["Operating Income", "$1.150", "$0.905", "+27.1%"],
        ["Net Income", "$0.940", "$0.780", "+20.5%"],
        ["Operating Cash Flow", "$1.380", "$1.020", "+35.3%"]
    ]
    p4_widths = [175, 110, 110, 120]

    page4.draw_rect(fitz.Rect(40, y, 555, y + row_height), color=border_color, fill=table_header_bg)
    x_curr = 40
    for col_idx, h_text in enumerate(p4_headers):
        page4.insert_text(fitz.Point(x_curr + 5, y + 15), h_text, fontsize=9, color=primary_color, fontname="hebo")
        x_curr += p4_widths[col_idx]
    y += row_height

    for row_idx, row in enumerate(p4_rows):
        x_curr = 40
        row_bg = (0.96, 0.97, 0.99) if row_idx % 2 == 1 else (1, 1, 1)
        page4.draw_rect(fitz.Rect(40, y, 555, y + row_height), color=border_color, fill=row_bg)
        for col_idx, cell_text in enumerate(row):
            page4.insert_text(fitz.Point(x_curr + 5, y + 15), cell_text, fontsize=9, color=text_color, fontname="helv")
            x_curr += p4_widths[col_idx]
        y += row_height

    page4.insert_text(fitz.Point(220, 810), "Page 4 of 4 | Apex Global Tech Q3 FY2025", fontsize=8, color=(0.5, 0.5, 0.5), fontname="helv")

    # Save to disk
    doc.save(str(out_file))
    doc.close()
    print(f"[+] Sample financial PDF successfully created at: {out_file.resolve()}")
    return out_file


if __name__ == "__main__":
    generate_sample_financial_report()
