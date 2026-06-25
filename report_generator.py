"""
Zone 5 — PDF Report Generator
Clinical-grade dark-themed report with:
  - Header with report ID, date, institution
  - Original image + Grad-CAM++ heatmap side by side
  - Diagnosis & confidence
  - LIME feature weights bar chart
  - SHAP region values
  - NLG-generated explanation
  - Disclaimer
"""

import os, base64, re
from io import BytesIO
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                  Table, TableStyle, Image as RLImage,
                                  HRFlowable, KeepTogether)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Flowable
import numpy as np

# ── Colors ─────────────────────────────────────────────────────────────
BG       = colors.HexColor('#0a0f1e')
BG2      = colors.HexColor('#0f1729')
PANEL    = colors.HexColor('#141d35')
ACCENT   = colors.HexColor('#00d4ff')
ACCENT2  = colors.HexColor('#7c3aed')
GREEN    = colors.HexColor('#00ff88')
RED      = colors.HexColor('#ff4757')
ORANGE   = colors.HexColor('#ffa502')
YELLOW   = colors.HexColor('#ffd700')
WHITE    = colors.white
GREY     = colors.HexColor('#8892a4')
DGREY    = colors.HexColor('#1e2a45')

CLASS_COLORS = {
    'Normal':    GREEN,
    'Infection': ORANGE,
    'Fracture':  RED,
    'Tumor':     colors.HexColor('#ff6b81'),
}

# ── Canvas background ──────────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(BG)
    canvas.rect(0, 0, A4[0], A4[1], fill=1, stroke=0)

    # Top accent line
    canvas.setFillColor(ACCENT)
    canvas.rect(0, A4[1]-2, A4[0], 2, fill=1, stroke=0)

    # Footer
    canvas.setFillColor(DGREY)
    canvas.rect(0, 0, A4[0], 18*mm, fill=1, stroke=0)
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(GREY)
    canvas.drawString(15*mm, 7*mm,
        'MedAI Vision  |  Explainable AI for Medical Diagnosis  |  For clinical review only — not a substitute for professional diagnosis')
    canvas.drawRightString(A4[0]-15*mm, 7*mm,
        f'Page {doc.page}  |  {datetime.now().strftime("%d %b %Y")}')
    canvas.restoreState()


class ColorRect(Flowable):
    """A colored rectangle background for panels."""
    def __init__(self, width, height, color, radius=4):
        Flowable.__init__(self)
        self.width = width; self.height = height
        self.color = color; self.radius = radius
    def draw(self):
        self.canv.setFillColor(self.color)
        self.canv.roundRect(0, 0, self.width, self.height,
                             self.radius, fill=1, stroke=0)


def make_styles():
    return {
        'title': ParagraphStyle('title',
            fontName='Helvetica-Bold', fontSize=22, textColor=ACCENT,
            alignment=TA_CENTER, spaceAfter=2),
        'subtitle': ParagraphStyle('subtitle',
            fontName='Helvetica', fontSize=9, textColor=GREY,
            alignment=TA_CENTER, spaceAfter=6),
        'section': ParagraphStyle('section',
            fontName='Helvetica-Bold', fontSize=11, textColor=ACCENT,
            spaceBefore=8, spaceAfter=4,
            borderPad=2),
        'body': ParagraphStyle('body',
            fontName='Helvetica', fontSize=8.5, textColor=WHITE,
            leading=14, spaceAfter=4),
        'small': ParagraphStyle('small',
            fontName='Helvetica', fontSize=7.5, textColor=GREY, leading=11),
        'cls_name': ParagraphStyle('cls_name',
            fontName='Helvetica-Bold', fontSize=28, textColor=WHITE,
            alignment=TA_CENTER),
        'conf': ParagraphStyle('conf',
            fontName='Helvetica-Bold', fontSize=14, textColor=ACCENT,
            alignment=TA_CENTER),
        'label': ParagraphStyle('label',
            fontName='Helvetica-Bold', fontSize=7.5, textColor=GREY),
        'value': ParagraphStyle('value',
            fontName='Helvetica-Bold', fontSize=9, textColor=WHITE),
        'disclaimer': ParagraphStyle('disclaimer',
            fontName='Helvetica-Oblique', fontSize=7, textColor=GREY,
            leading=10, spaceAfter=4),
        'nlg': ParagraphStyle('nlg',
            fontName='Helvetica', fontSize=8.5, textColor=WHITE,
            leading=14, spaceAfter=5),
        'nlg_bold': ParagraphStyle('nlg_bold',
            fontName='Helvetica-Bold', fontSize=8.5, textColor=ACCENT,
            leading=14, spaceAfter=3),
    }


def b64_to_rl_image(b64str, width_mm, height_mm=None):
    """Convert base64 image to ReportLab Image."""
    data = base64.b64decode(b64str)
    buf  = BytesIO(data)
    w = width_mm * mm
    if height_mm:
        return RLImage(buf, width=w, height=height_mm*mm)
    return RLImage(buf, width=w, height=w * 0.75)


def strip_html(text):
    """Remove HTML tags for plain-text sections."""
    return re.sub(r'<[^>]+>', '', text)


def parse_nlg_to_paragraphs(explanation_html, styles):
    """Convert NLG HTML explanation to ReportLab paragraphs."""
    elems = []
    # Split on <strong> section headers
    sections = re.split(r'(?=<strong>)', explanation_html.strip())
    for sec in sections:
        sec = sec.strip()
        if not sec: continue
        # Check if this line starts with a bold header
        m = re.match(r'<strong>(.*?)</strong>(.*)', sec, re.DOTALL)
        if m:
            header = m.group(1).strip()
            body   = m.group(2).strip()
            elems.append(Paragraph(header, styles['nlg_bold']))
            if body:
                clean = re.sub(r'<[^>]+>', '', body).strip()
                if clean:
                    elems.append(Paragraph(clean, styles['nlg']))
        else:
            clean = re.sub(r'<[^>]+>', '', sec).strip()
            if clean:
                elems.append(Paragraph(clean, styles['nlg']))
    return elems


def build_pdf_report(report_id, img_array, predicted_class, confidence,
                     all_probs, class_names, heatmap_b64, lime_text,
                     output_path, modality='chest', backbone='DenseNet-121',
                     lime_regions=None, lime_weights=None,
                     shap_values=None, measurements=None, patient_info=None):

    from PIL import Image as PILImage
    import io

    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=20*mm, bottomMargin=25*mm)

    S    = make_styles()
    W    = A4[0] - 30*mm
    story = []

    # ── HEADER ─────────────────────────────────────────────────────────
    story.append(Paragraph('MedAI Vision', S['title']))
    story.append(Paragraph(
        'Explainable AI for Medical Diagnosis  ·  DenseNet-121  ·  XAI Report',
        S['subtitle']))
    story.append(HRFlowable(width=W, thickness=1, color=ACCENT, spaceAfter=6))

    # Meta row
    ts = datetime.now().strftime('%d %B %Y, %H:%M')
    meta_data = [
        [Paragraph('REPORT ID', S['label']),   Paragraph(report_id, S['value']),
         Paragraph('DATE', S['label']),         Paragraph(ts, S['value']),
         Paragraph('MODALITY', S['label']),     Paragraph(modality.upper(), S['value']),
         Paragraph('BACKBONE', S['label']),     Paragraph(backbone, S['value'])],
    ]
    meta_tbl = Table(meta_data, colWidths=[22*mm,30*mm,16*mm,42*mm,22*mm,28*mm,22*mm,28*mm])
    meta_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), PANEL),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [PANEL]),
        ('ROUNDEDCORNERS', [4]),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(meta_tbl)
    story.append(Spacer(1, 8))

    # ── PATIENT INFO ───────────────────────────────────────────────────
    pat = patient_info or {}
    if any(pat.get(k) for k in ['name','age','gender','pat_id','doctor','symptoms','history']):
        story.append(Paragraph('▸  Patient Information', S['section']))
        pat_rows = []
        if pat.get('name'):
            pat_rows.append([Paragraph('PATIENT NAME', S['label']), Paragraph(pat['name'], S['value']),
                             Paragraph('AGE / GENDER', S['label']),
                             Paragraph((pat.get('age','—') or '—') + (' / ' + pat['gender'] if pat.get('gender') else ''), S['value'])])
        if pat.get('pat_id') or pat.get('doctor'):
            pat_rows.append([Paragraph('PATIENT ID', S['label']), Paragraph(pat.get('pat_id','—') or '—', S['value']),
                             Paragraph('REFERRING DOCTOR', S['label']), Paragraph(pat.get('doctor','—') or '—', S['value'])])
        if pat_rows:
            pt_tbl = Table(pat_rows, colWidths=[30*mm, W/2-30*mm, 35*mm, W/2-35*mm])
            pt_tbl.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), PANEL),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('LEFTPADDING', (0,0), (-1,-1), 6),
                ('RIGHTPADDING', (0,0), (-1,-1), 6),
                ('LINEBELOW', (0,0), (-1,-2), 0.5, colors.HexColor('#1a2540')),
            ]))
            story.append(pt_tbl)
        if pat.get('symptoms'):
            story.append(Paragraph(f'<b>Symptoms / Clinical Notes:</b>  {pat["symptoms"]}',
                ParagraphStyle('ps', fontName='Helvetica', fontSize=8, textColor=WHITE, leading=12, backColor=PANEL,
                               leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=4)))
        if pat.get('history'):
            story.append(Paragraph(f'<b>Medical History:</b>  {pat["history"]}',
                ParagraphStyle('ph', fontName='Helvetica', fontSize=8, textColor=WHITE, leading=12, backColor=PANEL,
                               leftIndent=6, rightIndent=6, spaceBefore=2, spaceAfter=2)))
        story.append(Spacer(1, 8))

    # ── DIAGNOSIS BANNER ───────────────────────────────────────────────
    cls_color = CLASS_COLORS.get(predicted_class, ACCENT)
    conf_pct  = f'{confidence*100:.1f}%'
    diag_data = [[
        Paragraph(f'<font color="#{cls_color.hexval()[2:]}">■</font>  {predicted_class.upper()}',
                  ParagraphStyle('d', fontName='Helvetica-Bold', fontSize=26,
                                 textColor=cls_color, alignment=TA_CENTER)),
        Paragraph(f'Confidence<br/><font size="22"><b>{conf_pct}</b></font>',
                  ParagraphStyle('c', fontName='Helvetica', fontSize=9,
                                 textColor=WHITE, alignment=TA_CENTER, leading=16)),
    ]]
    diag_tbl = Table(diag_data, colWidths=[W*0.6, W*0.4])
    diag_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (0,0), colors.HexColor(cls_color.hexval()[:7] + '33')),
        ('BACKGROUND', (1,0), (1,0), PANEL),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('BOX', (0,0), (-1,-1), 2, cls_color),
        ('LINEAFTER', (0,0), (0,0), 1, cls_color),
    ]))
    story.append(diag_tbl)
    story.append(Spacer(1, 8))

    # ── IMAGES: Original + Heatmap ─────────────────────────────────────
    story.append(Paragraph('▸  Visual Analysis  —  Grad-CAM++ Attention', S['section']))

    # Convert original numpy array to base64
    pil_orig = PILImage.fromarray(img_array.astype('uint8'))
    buf_orig = io.BytesIO()
    pil_orig.save(buf_orig, format='PNG')
    buf_orig.seek(0)
    orig_b64 = base64.b64encode(buf_orig.read()).decode()

    is_normal = (predicted_class == 'Normal')

    # ── Original image — full width, single column, always safe ──────────────
    orig_w  = W - 4*mm
    rl_orig = b64_to_rl_image(orig_b64, orig_w/mm, orig_w/mm * 0.75)
    orig_tbl = Table([[rl_orig]], colWidths=[orig_w + 4*mm])
    orig_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), PANEL),
        ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('BOX', (0,0), (-1,-1), 1, DGREY),
    ]))
    story.append(orig_tbl)
    story.append(Paragraph(
        'Original X-Ray',
        ParagraphStyle('cap', fontName='Helvetica-Bold', fontSize=7,
                        textColor=GREY, alignment=TA_CENTER)))
    story.append(Spacer(1, 6))

    # ── Heatmap — only for Infection / Fracture, never for Normal ─────────────
    if not is_normal and heatmap_b64:
        rl_heat  = b64_to_rl_image(heatmap_b64, (W - 4*mm)/mm, (W - 4*mm)/mm * 0.28)
        heat_tbl = Table([[rl_heat]], colWidths=[W])
        heat_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), PANEL),
            ('ALIGN',      (0,0), (-1,-1), 'CENTER'),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('BOX', (0,0), (-1,-1), 1, DGREY),
        ]))
        story.append(heat_tbl)
        story.append(Paragraph(
            'Left: Original  ·  Center: Grad-CAM++ Activation Map  ·  Right: Overlay with contours',
            ParagraphStyle('caph', fontName='Helvetica-Oblique', fontSize=7,
                            textColor=GREY, alignment=TA_CENTER)))
        story.append(Spacer(1, 8))
    else:
        story.append(Paragraph(
            '✓  No abnormal regions detected — heatmap not applicable for normal predictions.',
            ParagraphStyle('nm', fontName='Helvetica', fontSize=9,
                           textColor=colors.HexColor('#10b981'),
                           leading=14, alignment=TA_CENTER)))
        story.append(Spacer(1, 8))

    # ── CLASS PROBABILITIES — removed as per requirement ────────────────

    # ── LIME FEATURE WEIGHTS ──────────────────────────────────────────
    if lime_regions and lime_weights:
        story.append(Paragraph('▸  LIME  —  Superpixel Region Influence', S['section']))
        lime_rows = [[Paragraph('REGION', S['label']),
                      Paragraph('INFLUENCE %', S['label']),
                      Paragraph('WEIGHT', S['label'])]]
        for reg, wt in zip(lime_regions[:6], lime_weights[:6]):
            bar_len = max(int(wt * 0.8), 1)
            lime_rows.append([
                Paragraph(str(reg), S['body']),
                Paragraph(f'{wt:.0f}%', S['body']),
                Paragraph('█' * bar_len, ParagraphStyle('lb', fontName='Courier',
                            fontSize=7, textColor=ORANGE)),
            ])
        lime_tbl = Table(lime_rows, colWidths=[65*mm, 25*mm, W-90*mm])
        lime_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), DGREY),
            ('BACKGROUND', (0,1), (-1,-1), PANEL),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [PANEL, BG2]),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('LEFTPADDING', (0,0), (-1,-1), 7),
            ('BOX', (0,0), (-1,-1), 1, DGREY),
            ('LINEBELOW', (0,0), (-1,0), 1, ORANGE),
        ]))
        story.append(lime_tbl)
        story.append(Spacer(1, 8))

    # ── SHAP REGION VALUES ────────────────────────────────────────────
    if shap_values:
        story.append(Paragraph('▸  SHAP  —  Region Contribution Values', S['section']))
        shap_rows = [[Paragraph('REGION', S['label']),
                      Paragraph('SHAP VALUE', S['label']),
                      Paragraph('EFFECT', S['label'])]]
        sorted_shap = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
        for reg, val in sorted_shap[:9]:
            direction = 'Supports ↑' if val > 0 else 'Contradicts ↓'
            d_color   = GREEN if val > 0 else RED
            shap_rows.append([
                Paragraph(str(reg), S['body']),
                Paragraph(f'{val:+.4f}', ParagraphStyle('sv', fontName='Courier-Bold',
                            fontSize=8.5, textColor=d_color)),
                Paragraph(direction, ParagraphStyle('sd', fontName='Helvetica-Bold',
                            fontSize=8, textColor=d_color)),
            ])
        shap_tbl = Table(shap_rows, colWidths=[55*mm, 35*mm, W-90*mm])
        shap_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), DGREY),
            ('BACKGROUND', (0,1), (-1,-1), PANEL),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [PANEL, BG2]),
            ('TOPPADDING', (0,0), (-1,-1), 3),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3),
            ('LEFTPADDING', (0,0), (-1,-1), 7),
            ('BOX', (0,0), (-1,-1), 1, DGREY),
            ('LINEBELOW', (0,0), (-1,0), 1, colors.HexColor('#7c3aed')),
        ]))
        story.append(shap_tbl)
        story.append(Spacer(1, 8))

    # ── NLG EXPLANATION ───────────────────────────────────────────────
    story.append(Paragraph('▸  AI-Generated Explanation  —  NLG Module', S['section']))
    nlg_content = parse_nlg_to_paragraphs(lime_text, S)
    nlg_inner   = [[elem] for elem in nlg_content]
    if nlg_inner:
        # Wrap all NLG in a panel table
        flat_story = [[Paragraph('AI Diagnostic Summary', S['nlg_bold'])]]
        for row in nlg_inner:
            flat_story.append(row)
        nlg_tbl = Table(flat_story, colWidths=[W])
        nlg_tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), PANEL),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ('BOX', (0,0), (-1,-1), 1, DGREY),
            ('LINEAFTER', (0,0), (0,-1), 3, ACCENT),
        ]))
        story.append(nlg_tbl)
    story.append(Spacer(1, 8))

    # ── DISCLAIMER ─────────────────────────────────────────────────────
    story.append(HRFlowable(width=W, thickness=0.5, color=DGREY, spaceAfter=4))
    story.append(Paragraph(
        'DISCLAIMER: This report is generated by an AI system (MedAI Vision) using DenseNet-121 '
        'with Grad-CAM++, LIME, SHAP, and NLG-based explanation. It is intended as a decision '
        'support tool only and must NOT be used as the sole basis for clinical diagnosis or '
        'treatment decisions. Always consult a qualified radiologist or clinician.',
        S['disclaimer']))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"  PDF saved → {output_path}")