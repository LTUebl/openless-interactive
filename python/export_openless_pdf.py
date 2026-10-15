"""
Generate Comprehensive OpenLESS PDF Report
Paper: Turner et al., 2024 (https://doi.org/10.1101/2024.11.28.24318160)
Dataset: OpenCapData_71 (Stepoff-Leaning vs. Stepoff-Upright)
"""

import os
import shutil
import pandas as pd
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    HRFlowable,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw total page count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))

        # Running header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(36, 762, "OpenLESS Automated Biomechanical Report | OpenCapData_71")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.6)
            self.line(36, 756, 576, 756)

        # Running footer
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.6)
        self.line(36, 38, 576, 38)
        self.drawString(
            36,
            26,
            "OpenLESS: Automated Landing Error Scoring System (Turner et al., 2024 | medRxiv 10.1101/2024.11.28.24318160)",
        )
        self.drawRightString(576, 26, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_pdf_report(
    full_csv_path,
    short_csv_path,
    comparison_fig_path,
    qc_plots_dict,
    output_pdf_path,
):
    full_df = pd.read_csv(full_csv_path).set_index("Trial_ID")
    short_df = pd.read_csv(short_csv_path).set_index("Trial_ID")

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=40,
        bottomMargin=44,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1A202C"),
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
    )
    section_h1 = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=6,
        spaceAfter=3,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#2D3748"),
    )
    caption_style = ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#718096"),
        alignment=1,
    )
    th_style = ParagraphStyle(
        "TH",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1,
    )
    td_style = ParagraphStyle(
        "TD",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9.0,
        textColor=colors.HexColor("#2D3748"),
    )
    td_center = ParagraphStyle(
        "TDCenter",
        parent=td_style,
        alignment=1,
    )

    # Card styles
    card_label_style = ParagraphStyle(
        "CardLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#2D3748"),
    )
    card_score_lean = ParagraphStyle(
        "CardScoreLean",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#276749"),
    )
    card_score_upr = ParagraphStyle(
        "CardScoreUpr",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#C53030"),
    )
    card_rating_lean = ParagraphStyle(
        "CardRatingLean",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#22543D"),
    )
    card_rating_upr = ParagraphStyle(
        "CardRatingUpr",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#742A2A"),
    )
    card_desc = ParagraphStyle(
        "CardDesc",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#4A5568"),
    )

    story = []

    # ==================== PAGE 1: HEADER & EXECUTIVE SUMMARY ====================
    story.append(Paragraph("OpenLESS: Automated Landing Error Scoring System", title_style))
    story.append(Spacer(1, 2))
    meta_text = (
        "<b>Subject Data:</b> OpenCapData_71 (Jump Landing Trials) &nbsp;|&nbsp; "
        "<b>Algorithm Reference:</b> Turner et al. (2024), <i>medRxiv</i> DOI: 10.1101/2024.11.28.24318160<br/>"
        f"<b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y')} &nbsp;|&nbsp; "
        "<b>Processing:</b> OpenCap Camera 60 Hz, 4th-order Butterworth Low-Pass 12 Hz"
    )
    story.append(Paragraph(meta_text, subtitle_style))
    story.append(Spacer(1, 5))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2B6CB0"), spaceAfter=6))

    leaning_tot = int(short_df.loc["Stepoff-Leaning", "Total_Score"])
    upright_tot = int(short_df.loc["Stepoff-Upright", "Total_Score"])

    card_left = [
        Paragraph("Trial 1: Stepoff-Leaning", card_label_style),
        Paragraph(f"Total LESS Score: {leaning_tot} / 19", card_score_lean),
        Paragraph("Rating: EXCELLENT / LOW RISK", card_rating_lean),
        Spacer(1, 2),
        Paragraph(
            "Soft landing with active sagittal dissipation (deep knee & hip flexion, forward trunk lean). "
            "Symmetric forefoot initial ground contact.",
            card_desc,
        ),
    ]

    card_right = [
        Paragraph("Trial 2: Stepoff-Upright", card_label_style),
        Paragraph(f"Total LESS Score: {upright_tot} / 19", card_score_upr),
        Paragraph("Rating: POOR / HIGH RISK", card_rating_upr),
        Spacer(1, 2),
        Paragraph(
            "Stiff, erect landing with restricted flexion, upright rigid spine, asymmetric heel-strike, "
            "lateral hip shift, and dynamic knee valgus collapse.",
            card_desc,
        ),
    ]

    card_table = Table([[card_left, card_right]], colWidths=[265, 265])
    card_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#F0FFF4")),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#FFF5F5")),
                ("BOX", (0, 0), (0, 0), 1, colors.HexColor("#9AE6B4")),
                ("BOX", (1, 0), (1, 0), 1, colors.HexColor("#FEB2B2")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(card_table)
    story.append(Spacer(1, 6))

    # ==================== TABLE OF 19 LESS ITEMS ====================
    story.append(Paragraph("Itemized Landing Error Scoring System (LESS) Criteria", section_h1))

    def fmt_item(item_idx, name, crit, l_score, l_detail, u_score, u_detail):
        l_color = "#C53030" if l_score > 0 else "#276749"
        u_color = "#C53030" if u_score > 0 else "#276749"
        l_txt = f"<font color='{l_color}'><b>{l_score}</b></font> <font size='6.3' color='#4A5568'>({l_detail})</font>"
        u_txt = f"<font color='{u_color}'><b>{u_score}</b></font> <font size='6.3' color='#4A5568'>({u_detail})</font>"
        return [
            Paragraph(f"<b>{item_idx}</b>", td_center),
            Paragraph(f"<b>{name}</b>", td_style),
            Paragraph(crit, td_style),
            Paragraph(l_txt, td_style),
            Paragraph(u_txt, td_style),
        ]

    l_row = full_df.loc["Stepoff-Leaning"]
    u_row = full_df.loc["Stepoff-Upright"]

    table_rows = [
        [
            Paragraph("<b>#</b>", th_style),
            Paragraph("<b>LESS Item</b>", th_style),
            Paragraph("<b>Operational Criteria (Turner et al., 2024)</b>", th_style),
            Paragraph("<b>Stepoff-Leaning</b>", th_style),
            Paragraph("<b>Stepoff-Upright</b>", th_style),
        ],
        fmt_item(
            1,
            "Knee Flexion (IC)",
            "&lt; 30° knee flexion at initial contact",
            int(l_row["1_Knee_Flexion_IC_T"]),
            f"L: {l_row['1_Knee_Flexion_IC_L_deg']:.1f}°, R: {l_row['1_Knee_Flexion_IC_R_deg']:.1f}°",
            int(u_row["1_Knee_Flexion_IC_T"]),
            f"L: {u_row['1_Knee_Flexion_IC_L_deg']:.1f}°, R: {u_row['1_Knee_Flexion_IC_R_deg']:.1f}°",
        ),
        fmt_item(
            2,
            "Hip Flexion (IC)",
            "&lt; 30° hip flexion at initial contact",
            int(l_row["2_Hip_Flexion_IC_T"]),
            f"L: {l_row['2_Hip_Flexion_IC_L_deg']:.1f}°, R: {l_row['2_Hip_Flexion_IC_R_deg']:.1f}°",
            int(u_row["2_Hip_Flexion_IC_T"]),
            f"L: {u_row['2_Hip_Flexion_IC_L_deg']:.1f}°, R: {u_row['2_Hip_Flexion_IC_R_deg']:.1f}°",
        ),
        fmt_item(
            3,
            "Trunk Flexion (IC)",
            "&lt; 10° trunk flexion (&gt; -10° lumbar extension)",
            int(l_row["3_Trunk_Flexion_IC_T"]),
            f"{l_row['3_Trunk_Flexion_IC_deg']:.1f}°",
            int(u_row["3_Trunk_Flexion_IC_T"]),
            f"{u_row['3_Trunk_Flexion_IC_deg']:.1f}°",
        ),
        fmt_item(
            4,
            "Ankle Plantarflexion",
            "Heel contact prior to or simultaneous with toes",
            int(l_row["4_Ankle_Plantar_Flexion_IC_T"]),
            "Toe-first contact",
            int(u_row["4_Ankle_Plantar_Flexion_IC_T"]),
            "L: Heel-first, R: Toe",
        ),
        fmt_item(
            5,
            "Asymmetrical Timing",
            "One foot lands &gt; 34 ms before the other",
            int(l_row["5_Asymmetrical_Timing"]),
            "0.0 ms diff",
            int(u_row["5_Asymmetrical_Timing"]),
            "16.7 ms diff",
        ),
        fmt_item(
            6,
            "Asymm. Heel/Toe Strike",
            "One foot lands heel-first, other toe-first",
            int(l_row["6_Asymmetrical_HeelToe_ToeHeel"]),
            "Both toe-first",
            int(u_row["6_Asymmetrical_HeelToe_ToeHeel"]),
            "L heel vs R toe",
        ),
        fmt_item(
            7,
            "Lateral Trunk Flexion",
            "&gt; 5° lateral lumbar bending to either side",
            int(l_row["7_Lateral_Trunk_Flexion_IC_T"]),
            f"{l_row['7_Lateral_Trunk_Flexion_IC_deg']:.1f}°",
            int(u_row["7_Lateral_Trunk_Flexion_IC_T"]),
            f"{u_row['7_Lateral_Trunk_Flexion_IC_deg']:.1f}°",
        ),
        fmt_item(
            8,
            "Medial Knee Pos. (IC)",
            "Hip rot &gt; 8° &amp; add &gt; -1°, or hip add &gt; 1°",
            int(l_row["8_Medial_Knee_IC_T"]),
            f"rot: {l_row['8_Medial_Knee_IC_L_deg']:.1f}°, add: {l_row['9_Stance_Hip_abd_IC_L_deg']:.1f}°",
            int(u_row["8_Medial_Knee_IC_T"]),
            f"rot: {u_row['8_Medial_Knee_IC_L_deg']:.1f}°, add: {u_row['9_Stance_Hip_abd_IC_L_deg']:.1f}°",
        ),
        fmt_item(
            9,
            "Wide Stance Width",
            "&gt; 12° hip abduction at IC (&lt; -12° adduction)",
            int(l_row["9_Stance_Wide_IC_T"]),
            f"L: {l_row['9_Stance_Hip_abd_IC_L_deg']:.1f}°, R: {l_row['9_Stance_Hip_abd_IC_R_deg']:.1f}°",
            int(u_row["9_Stance_Wide_IC_T"]),
            f"L: {u_row['9_Stance_Hip_abd_IC_L_deg']:.1f}°, R: {u_row['9_Stance_Hip_abd_IC_R_deg']:.1f}°",
        ),
        fmt_item(
            10,
            "Narrow Stance Width",
            "&gt; 0° hip adduction at initial contact",
            int(l_row["10_Stance_Narrow_IC_T"]),
            f"L: {l_row['9_Stance_Hip_abd_IC_L_deg']:.1f}°, R: {l_row['9_Stance_Hip_abd_IC_R_deg']:.1f}°",
            int(u_row["10_Stance_Narrow_IC_T"]),
            f"L: {u_row['9_Stance_Hip_abd_IC_L_deg']:.1f}°, R: {u_row['9_Stance_Hip_abd_IC_R_deg']:.1f}°",
        ),
        fmt_item(
            11,
            "Foot Inward Rotation",
            "&gt; 10° subtalar angle inward (IR)",
            int(l_row["11_IR_Foot_IC_T"]),
            f"L: {l_row['11_IR_Foot_IC_L_deg']:.1f}°, R: {l_row['11_IR_Foot_IC_R_deg']:.1f}°",
            int(u_row["11_IR_Foot_IC_T"]),
            f"L: {u_row['11_IR_Foot_IC_L_deg']:.1f}°, R: {u_row['11_IR_Foot_IC_R_deg']:.1f}°",
        ),
        fmt_item(
            12,
            "Foot Outward Rotation",
            "&gt; 15° subtalar angle outward (&lt; -15°)",
            int(l_row["12_ER_Foot_IC_T"]),
            f"L: {l_row['12_ER_Foot_IC_L_deg']:.1f}°, R: {l_row['12_ER_Foot_IC_R_deg']:.1f}°",
            int(u_row["12_ER_Foot_IC_T"]),
            f"L: {u_row['12_ER_Foot_IC_L_deg']:.1f}°, R: {u_row['12_ER_Foot_IC_R_deg']:.1f}°",
        ),
        fmt_item(
            13,
            "Knee Flexion (Max)",
            "&lt; 65° maximum knee flexion in stance",
            int(l_row["13_Knee_Flexion_Displacement_T"]),
            f"L: {l_row['13_Knee_Flexion_Displacement_L_deg']:.1f}°, R: {l_row['13_Knee_Flexion_Displacement_R_deg']:.1f}°",
            int(u_row["13_Knee_Flexion_Displacement_T"]),
            f"L: {u_row['13_Knee_Flexion_Displacement_L_deg']:.1f}°, R: {u_row['13_Knee_Flexion_Displacement_R_deg']:.1f}°",
        ),
        fmt_item(
            14,
            "Hip Flexion (Max)",
            "&lt; 45° maximum hip flexion at peak knee flexion",
            int(l_row["14_Hip_Flexion_Displacement_T"]),
            f"L: {l_row['14_Hip_Flexion_Displacement_L_deg']:.1f}°, R: {l_row['14_Hip_Flexion_Displacement_R_deg']:.1f}°",
            int(u_row["14_Hip_Flexion_Displacement_T"]),
            f"L: {u_row['14_Hip_Flexion_Displacement_L_deg']:.1f}°, R: {u_row['14_Hip_Flexion_Displacement_R_deg']:.1f}°",
        ),
        fmt_item(
            15,
            "Trunk Flexion (Max)",
            "&lt; 10° trunk displacement between IC and Max",
            int(l_row["15_Trunk_Flexion_Max_T"]),
            f"Δ: {abs(l_row['3_Trunk_Flexion_IC_deg'] - l_row['15_Trunk_Flexion_Max_deg']):.1f}°",
            int(u_row["15_Trunk_Flexion_Max_T"]),
            f"Δ: {abs(u_row['3_Trunk_Flexion_IC_deg'] - u_row['15_Trunk_Flexion_Max_deg']):.1f}°",
        ),
        fmt_item(
            16,
            "Medial Knee Pos. (Max)",
            "Hip rot &gt; 8° &amp; add &gt; -1°, or hip add &gt; 1° at Max",
            int(l_row["16_Medial_Knee_Max_T"]),
            f"L add: {l_row['17_Asymmetrical_Load_L_deg']:.1f}°, R: {l_row['17_Asymmetrical_Load_R_deg']:.1f}°",
            int(u_row["16_Medial_Knee_Max_T"]),
            f"L add: {u_row['17_Asymmetrical_Load_L_deg']:.1f}°, R: {u_row['17_Asymmetrical_Load_R_deg']:.1f}°",
        ),
        fmt_item(
            17,
            "Asymmetric Loading",
            "&gt; 5° difference in max hip adduction (shift)",
            int(l_row["17_Asymmetrical_Load_T"]),
            f"Diff: {abs(l_row['17_Asymmetrical_Load_L_deg'] - l_row['17_Asymmetrical_Load_R_deg']):.1f}°",
            int(u_row["17_Asymmetrical_Load_T"]),
            f"Diff: {abs(u_row['17_Asymmetrical_Load_L_deg'] - u_row['17_Asymmetrical_Load_R_deg']):.1f}°",
        ),
        fmt_item(
            18,
            "Sagittal Displacement",
            "0 = Soft (&gt;55° knee, &gt;73° hip), 1 = Avg, 2 = Stiff",
            int(l_row["18_Joint_Displacement_T"]),
            "0 (Soft landing)",
            int(u_row["18_Joint_Displacement_T"]),
            "2 (Stiff landing)",
        ),
        fmt_item(
            19,
            "Overall Impression",
            "0 = Excellent, 1 = Average, 2 = Poor",
            int(l_row["19_Overall_Impression_T"]),
            "0 (Excellent)",
            int(u_row["19_Overall_Impression_T"]),
            "2 (Poor)",
        ),
        [
            Paragraph("<b>∑</b>", td_center),
            Paragraph("<b>TOTAL LESS SCORE</b>", ParagraphStyle("TB", parent=td_style, fontName="Helvetica-Bold", fontSize=7.5)),
            Paragraph("<b>Sum of all 19 Movement Error Items</b>", ParagraphStyle("TB2", parent=td_style, fontName="Helvetica-Bold", fontSize=7.5)),
            Paragraph(f"<font color='#276749'><b>{leaning_tot}</b></font> <b>/ 19</b>", ParagraphStyle("TB3", parent=td_style, fontName="Helvetica-Bold", fontSize=8)),
            Paragraph(f"<font color='#C53030'><b>{upright_tot}</b></font> <b>/ 19</b>", ParagraphStyle("TB4", parent=td_style, fontName="Helvetica-Bold", fontSize=8)),
        ],
    ]

    less_table = Table(table_rows, colWidths=[18, 105, 177, 120, 120])
    table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2.0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.0),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E0")),
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#EDF2F7")),
        ]
    )

    for r in range(1, len(table_rows) - 1):
        if r % 2 == 0:
            table_style.add("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F7FAFC"))

    less_table.setStyle(table_style)
    story.append(less_table)

    # ==================== PAGE 2: KINEMATIC TRAJECTORIES & ANALYSIS ====================
    story.append(PageBreak())
    story.append(Paragraph("Biomechanical Trajectories & Kinematic Comparison", section_h1))
    story.append(
        Paragraph(
            "Continuous 3D joint kinematics during the stance phase (initial contact to toe-off) extracted "
            "from OpenCap markerless motion capture and analyzed through the OpenLESS automated pipeline.",
            body_style,
        )
    )
    story.append(Spacer(1, 4))

    if os.path.exists(comparison_fig_path):
        story.append(Image(comparison_fig_path, width=540, height=385))
        story.append(
            Paragraph(
                "Figure 1: Joint kinematics during landing stance comparing Stepoff-Leaning (blue) and Stepoff-Upright (red), "
                "with clinical thresholds and itemized LESS error score distribution.",
                caption_style,
            )
        )
    story.append(Spacer(1, 8))

    analysis_p1 = (
        "<b>Sagittal Plane Energy Dissipation:</b> In <i>Stepoff-Leaning</i>, the jumper proactively flexes the hips (peak 80.2°) "
        "and knees (peak 81.4°) accompanied by a pronounced forward trunk lean (45.2°), effectively distributing vertical ground reaction "
        "forces across the lower limb musculature. In contrast, <i>Stepoff-Upright</i> exhibits a rigid landing strategy with limited knee flexion "
        "(54.8° on left), limited hip flexion (39.3°), and an upright rigid spine (displacement Δ = 0.1°), which substantially elevates "
        "peak ground impact and joint contact stresses transmitted directly through the tibiofemoral joint."
    )
    analysis_p2 = (
        "<b>Frontal Plane Alignment & Valgus Collapse:</b> In <i>Stepoff-Upright</i>, as the left limb reaches peak landing depth, "
        "the hip moves into positive adduction (+1.1°), producing dynamic knee valgus collapse. Furthermore, an asymmetric "
        "hip adduction discrepancy of 10.8° between right and left limbs indicates pronounced lateral trunk/hip shifting, "
        "a primary clinical predictor of non-contact anterior cruciate ligament (ACL) injury."
    )
    story.append(Paragraph(analysis_p1, body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(analysis_p2, body_style))

    # ==================== PAGE 3: EVENT DETECTION & QUALITY CONTROL ====================
    story.append(PageBreak())
    story.append(Paragraph("Jump-Landing Event Detection & Quality Control", section_h1))
    story.append(
        Paragraph(
            "Automated detection of Initial Ground Contact (IC) and Toe-Off (TO) from 4th-order low-pass filtered "
            "(12 Hz) vertical coordinates of big toe and heel markers, validating stance phase bracketing.",
            body_style,
        )
    )
    story.append(Spacer(1, 6))

    timing_data = [
        [
            Paragraph("<b>Trial ID</b>", th_style),
            Paragraph("<b>Limb</b>", th_style),
            Paragraph("<b>Toe Minima (s)</b>", th_style),
            Paragraph("<b>Heel Minima (s)</b>", th_style),
            Paragraph("<b>Contact Type</b>", th_style),
            Paragraph("<b>IC Time (s)</b>", th_style),
            Paragraph("<b>TO Time (s)</b>", th_style),
            Paragraph("<b>Stance Duration (s)</b>", th_style),
        ],
        [
            Paragraph("<b>Stepoff-Leaning</b>", td_style),
            Paragraph("Left", td_center),
            Paragraph("1.800", td_center),
            Paragraph("1.900", td_center),
            Paragraph("<font color='#276749'>Toe-first</font>", td_center),
            Paragraph("1.800", td_center),
            Paragraph("2.717", td_center),
            Paragraph("0.917", td_center),
        ],
        [
            Paragraph("<b>Stepoff-Leaning</b>", td_style),
            Paragraph("Right", td_center),
            Paragraph("1.800", td_center),
            Paragraph("1.850", td_center),
            Paragraph("<font color='#276749'>Toe-first</font>", td_center),
            Paragraph("1.800", td_center),
            Paragraph("2.467", td_center),
            Paragraph("0.667", td_center),
        ],
        [
            Paragraph("<b>Stepoff-Upright</b>", td_style),
            Paragraph("Left", td_center),
            Paragraph("1.117", td_center),
            Paragraph("1.067", td_center),
            Paragraph("<font color='#C53030'>Heel-first</font>", td_center),
            Paragraph("1.067", td_center),
            Paragraph("1.350", td_center),
            Paragraph("0.283", td_center),
        ],
        [
            Paragraph("<b>Stepoff-Upright</b>", td_style),
            Paragraph("Right", td_center),
            Paragraph("1.050", td_center),
            Paragraph("1.067", td_center),
            Paragraph("<font color='#276749'>Toe-first</font>", td_center),
            Paragraph("1.050", td_center),
            Paragraph("1.767", td_center),
            Paragraph("0.717", td_center),
        ],
    ]

    timing_table = Table(timing_data, colWidths=[95, 45, 65, 65, 75, 60, 60, 75])
    timing_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("BACKGROUND", (0, 1), (-1, 2), colors.HexColor("#F0FFF4")),
                ("BACKGROUND", (0, 3), (-1, 4), colors.HexColor("#FFF5F5")),
            ]
        )
    )
    story.append(timing_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("Marker Vertical Trajectories & Ground Contact Verification", ParagraphStyle("H2", parent=section_h1, fontSize=10.5)))
    qc_grid_data = [
        [
            Image(qc_plots_dict["Stepoff-Leaning_left"], width=265, height=130),
            Image(qc_plots_dict["Stepoff-Leaning_right"], width=265, height=130),
        ],
        [
            Image(qc_plots_dict["Stepoff-Upright_left"], width=265, height=130),
            Image(qc_plots_dict["Stepoff-Upright_right"], width=265, height=130),
        ],
    ]
    qc_table = Table(qc_grid_data, colWidths=[270, 270])
    qc_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    story.append(qc_table)
    story.append(
        Paragraph(
            "Figure 2: Quality control verification plots displaying filtered heel (orange) and toe (blue) vertical trajectories. "
            "Dashed green line marks initial contact (IC); dashed red line marks toe-off (TO).",
            caption_style,
        )
    )
    story.append(Spacer(1, 8))

    manifest_p = (
        "<b>Data Artifacts Generated:</b><br/>"
        "• Comprehensive Scores: <code>OpenCapData_71/less_scores_final.csv</code> (68 parameters)<br/>"
        "• Summary LESS Scores: <code>OpenCapData_71/less_short_final.csv</code> (19 criteria + Total)<br/>"
        "• Clipped/Filtered Stance Kinematics: <code>OpenCapData_71/OpenLESS_Results/*_kin_*_filtered.csv</code><br/>"
        "• Python Pipeline: <code>run_openless_scoring.py</code> (Turner et al., 2024 implementation)"
    )
    story.append(Paragraph(manifest_p, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF Report generated successfully: {output_pdf_path}")

    # Copy to root OpenCapData_71
    root_pdf_path = os.path.join(os.path.dirname(os.path.dirname(output_pdf_path)), "OpenLESS_Report_OpenCapData_71.pdf")
    shutil.copy2(output_pdf_path, root_pdf_path)
    print(f"Copied PDF Report to: {root_pdf_path}")


if __name__ == "__main__":
    base_dir = r"OpenCapData_71\OpenLESS_Results"
    full_csv = os.path.join(base_dir, "OpenCapData_71_less_scores_final.csv")
    short_csv = os.path.join(base_dir, "OpenCapData_71_less_short_final.csv")
    comp_fig = os.path.join(base_dir, "OpenLESS_comparison_summary.png")
    qc_plots = {
        "Stepoff-Leaning_left": os.path.join(base_dir, "Stepoff-Leaning_left_qc.png"),
        "Stepoff-Leaning_right": os.path.join(base_dir, "Stepoff-Leaning_right_qc.png"),
        "Stepoff-Upright_left": os.path.join(base_dir, "Stepoff-Upright_left_qc.png"),
        "Stepoff-Upright_right": os.path.join(base_dir, "Stepoff-Upright_right_qc.png"),
    }
    out_pdf = os.path.join(base_dir, "OpenLESS_Report_OpenCapData_71.pdf")

    build_pdf_report(full_csv, short_csv, comp_fig, qc_plots, out_pdf)
