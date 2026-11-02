"""PDF report generation (ReportLab). Text is kept ASCII-safe for the built-in fonts."""
from datetime import datetime
from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

RISK_COLORS = {"Low": "#2f9e44", "Moderate": "#f08c00", "High": "#e03131"}


def build_pdf(result, summary, factors, tips, disclaimer):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="Heart Disease Risk Report")
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=20, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=4)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10, leading=14)
    small = ParagraphStyle("small", parent=body, fontSize=8.5, textColor=colors.grey)

    def bullets(items):
        return [Paragraph("- " + escape(text), body) for text in items]

    risk_color = RISK_COLORS.get(result["risk"], "#555555")
    story = [
        Paragraph("Heart Disease Risk Report", h1),
        Paragraph("Generated on " + datetime.now().strftime("%d %b %Y, %H:%M"), small),
        Spacer(1, 8),
        Paragraph(f'<font color="{risk_color}" size="16"><b>{escape(result["risk"])} risk</b></font>', body),
        Paragraph(escape(result["label"]) + f" - model probability score: <b>{result['percent']}%</b>", body),
        Paragraph("Details entered", h2),
    ]

    table = Table([[label, str(value)] for label, value in summary],
                  colWidths=[85 * mm, 80 * mm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.whitesmoke, colors.white]),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(table)

    if factors:
        story.append(Paragraph("Why this result? (approximate)", h2))
        lines = []
        for item in factors:
            direction = "raised" if item["effect"] > 0 else "lowered"
            lines.append(f"{item['label']} ({item['value']}) {direction} the risk score by "
                         f"{abs(item['effect']):.1f} points compared with a typical patient.")
        story += bullets(lines)

    story.append(Paragraph("Health tips for you", h2))
    story += bullets(tips["personal"])
    story.append(Paragraph("General heart-healthy habits", h2))
    story += bullets(tips["general"])
    story += [Spacer(1, 6), Paragraph("<b>" + escape(tips["emergency"]) + "</b>", body),
              Spacer(1, 10), Paragraph(escape(disclaimer), small)]

    doc.build(story)
    buffer.seek(0)
    return buffer
