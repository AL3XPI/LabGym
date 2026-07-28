'''
Copyright (C)
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
'''

import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import datetime
import json
import logging
import sys
import yaml

logger = logging.getLogger(__name__)

# Default parameter configurations
DEFAULT_TRAIN_PARAMS = {
    'dim_conv': 64,
    'dim_tconv': 32,
    'channel': 1,
    'time_step': 15,
    'level_conv': 2,
    'level_tconv': 1,
    'aug_methods': [],
    'augvalid': True,
    'include_bodyparts': True,
    'std': 0.0,
    'background_free': True,
    'black_background': True,
    'behavior_mode': 0,
    'social_distance': 0.0,
    'color_costar': False,
    'training_onfly': False,
    'out_path': None,
    'out_folder': None,
    'path_to_annotation': None,
    'path_to_trainingimages': None,
    'iteration_num': 200,
    'inference_size': 480,
}

DEFAULT_INFER_PARAMS = {
    'use_detector': False,
    'animal_kinds': ['animal'],
    'animal_to_include': [],
    'behavior_to_include': ['all'],
    'detector_batch': 1,
    'detection_threshold': 0.0,
    'background_path': None,
    'framewidth': None,
    'delta': 10000.0,
    'decode_animalnumber': False,
    'animal_number': None,
    'autofind_t': False,
    'decode_t': False,
    't': 0.0,
    'duration': 0.0,
    'decode_extraction': False,
    'ex_start': 0,
    'ex_end': None,
    'dim_tconv': 8,
    'dim_conv': 8,
    'channel': 1,
    'length': 15,
    'animal_vs_bg': 0,
    'stable_illumination': True,
    'animation_analyzer': True,
    'ID_colors': [(255, 255, 255)],
    'parameter_to_analyze': [],
    'include_bodyparts': False,
    'std': 0.0,
    'uncertain': 0.0,
    'min_length': None,
    'show_legend': True,
    'background_free': True,
    'black_background': True,
    'normalize_distance': True,
    'social_distance': 0.0,
    'color_costar': False,
    'specific_behaviors': {},
    'correct_ID': False,
}

def str2bool(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_json_or_val(v):
    if v is None:
        return None
    if isinstance(v, str) and (v.startswith('{') or v.startswith('[')):
        try:
            return json.loads(v)
        except Exception:
            pass
    return v

def merge_configs(cli_args, default_params):
    merged = default_params.copy()
    
    # Load config file if specified
    if cli_args.config and os.path.exists(cli_args.config):
        logger.info(f"Loading baseline configurations from {cli_args.config}")
        with open(cli_args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config and isinstance(yaml_config, dict):
                for k, v in yaml_config.items():
                    merged[k] = v
                    
    # Overlay CLI overrides
    for k, v in vars(cli_args).items():
        if k not in ['config', 'func'] and v is not None:
            merged[k] = v
            
    return merged

def get_centroid(embedding_map, file_list):
    """Calculates the mathematical centroid of a cluster of videos and returns the most representative file."""
    import numpy as np
    if not file_list:
        return None
    if len(file_list) == 1:
        return file_list[0]
        
    embs = [embedding_map[f] for f in file_list if f in embedding_map]
    if not embs:
        return file_list[0]
        
    mean_emb = np.mean(embs, axis=0)
    mean_norm = np.linalg.norm(mean_emb)
    if mean_norm == 0:
        return file_list[0]
        
    best_sim = -2.0
    centroid_file = file_list[0]
    
    for f in file_list:
        if f not in embedding_map:
            continue
        emb = embedding_map[f]
        emb_norm = np.linalg.norm(emb)
        if emb_norm == 0:
            continue
        
        sim = np.dot(emb, mean_emb) / (emb_norm * mean_norm)
        if sim > best_sim:
            best_sim = sim
            centroid_file = f
            
    return centroid_file

def parse_pair(s):
    # Extracts true and predicted classes from strings like "A -> B" or "A -> B (5 errors)"
    clean = s.split(" (")[0]
    parts = clean.split("->")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return None

def generate_pdf_report(filepath, dataset_name, report, cm, classnames, example_map, embedding_map, nlp_major, redundant_pairs, emergent_pairs, generalization_pairs):
    """Generates a styled, professional PDF report of the triage action plan headlessly, matching the official LabGym Guide styling."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.colors import HexColor, Color
        from reportlab.lib import colors
        from reportlab.lib.units import inch
    except ImportError:
        logger.error("ReportLab is required to generate PDFs. Please run 'pip install reportlab'.")
        print("Error: ReportLab is required to generate PDFs. Please run 'pip install reportlab'.")
        return False

    # Document setup
    doc = SimpleDocTemplate(filepath, pagesize=letter, rightMargin=54, leftMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=18, textColor=HexColor("#00274C"), spaceAfter=6)
    subtitle_style = ParagraphStyle('SubTitle', parent=styles['Normal'], fontName='Helvetica', fontSize=10, textColor=HexColor("#666666"), spaceAfter=4)
    
    h2_style = ParagraphStyle('H2', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=12, textColor=HexColor("#00274C"), spaceBefore=14, spaceAfter=8)
    body_style = ParagraphStyle('Body', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=14, textColor=HexColor("#333333"), spaceAfter=10)
    action_style = ParagraphStyle('Action', parent=styles['Normal'], fontName='Helvetica-BoldOblique', fontSize=10, textColor=HexColor("#D82C20"), spaceAfter=10)
    item_style = ParagraphStyle('Item', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=14, spaceAfter=6)
    caption_style = ParagraphStyle('Caption', parent=styles['Normal'], fontName='Helvetica-Oblique', fontSize=9, textColor=HexColor("#555555"), leading=13)

    Story = []
    
    # 1. Header Cover Block
    package_dir = os.path.dirname(__file__)
    logo_path = os.path.join(package_dir, "assets", "icons", "labgym.png")
    
    if os.path.exists(logo_path):
        logo_img = Image(logo_path)
        logo_img.drawWidth = 1.1 * inch
        logo_img.drawHeight = 1.1 * inch
        
        details_table = Table([[Paragraph(f"<b>TABLE OF CONTENTS</b>", title_style)],
                               [Paragraph(f"<b>LabGym Triage Action Plan</b>", ParagraphStyle('SubT', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=13, textColor=HexColor("#333333")))],
                               [Paragraph(f"Dataset Target: {dataset_name} | Date: {datetime.datetime.now().strftime('%m/%d/%Y')}", subtitle_style)],
                               [Paragraph(f"Last Updated: {datetime.datetime.now().strftime('%m/%d/%Y')}", subtitle_style)]],
                              colWidths=[4.2 * inch])
        details_table.setStyle(TableStyle([
            ('BOTTOMPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 0),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        
        header_table = Table([[logo_img, details_table]], colWidths=[1.3 * inch, 4.3 * inch])
        header_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LEFTPADDING', (1,0), (1,0), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
            ('TOPPADDING', (0,0), (-1,-1), 5),
        ]))
        Story.append(header_table)
    else:
        Story.append(Paragraph("LabGym Triage Action Plan", title_style))
        Story.append(Paragraph(f"Dataset Target: {dataset_name} | Date Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style))

    # Clean thick blue line below header block
    line_table = Table([[""]], colWidths=[doc.width], rowHeights=[2])
    line_table.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 2, HexColor("#00274C")),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    Story.append(line_table)
    Story.append(Spacer(1, 15))

    # Overall Metrics Summary Block
    accuracy = report.get('accuracy', 0.0)
    macro_f1 = report.get('macro avg', {}).get('f1-score', 0.0)
    Story.append(Paragraph(f"<b>Overall Model Diagnostics Summary:</b>", ParagraphStyle('SumH', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, textColor=HexColor("#00274C"))))
    Story.append(Paragraph(f"Accuracy: <b>{accuracy*100:.1f}%</b> | Macro F1-Score: <b>{macro_f1*100:.1f}%</b>", body_style))
    Story.append(Spacer(1, 5))

    # Confusion Matrix Table
    Story.append(Paragraph("Confusion Matrix Overview", h2_style))
    Story.append(Paragraph("<i>Cells with gold borders highlight systematic class confusions addressed in diagnostics.</i>", caption_style))
    Story.append(Spacer(1, 5))
    
    headers = ["True \\ Pred"] + [str(idx + 1) for idx in range(len(classnames))]
    cm_data = [headers]
    
    for i in range(len(classnames)):
        row_data = [f"{i + 1}. {classnames[i]}"]
        for j in range(len(classnames)):
            row_data.append(str(cm[i][j]))
        cm_data.append(row_data)

    t_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), HexColor("#00274C")), 
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('ALIGN', (0,1), (0,-1), 'LEFT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 7),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('BACKGROUND', (0,1), (0,-1), HexColor("#f0f0f0")), 
        ('FONTNAME', (0,1), (0,-1), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ])
    
    for i in range(len(classnames)):
        row_sum = sum(cm[i])
        for j in range(len(classnames)):
            val = cm[i][j]
            if val > 0:
                if i == j:
                    intensity = min(1.0, val / row_sum) if row_sum > 0 else 0
                    bg_color = Color(0, 0.8, 0, alpha=0.1 + (0.4 * intensity))
                else:
                    pct = val / row_sum if row_sum > 0 else 0
                    intensity = min(1.0, pct / 0.5)
                    bg_color = Color(0.9, 0, 0, alpha=0.1 + (0.5 * intensity))
                t_style.add('BACKGROUND', (j+1, i+1), (j+1, i+1), bg_color)

    # Highlight all confusions
    for i in range(len(classnames)):
        for j in range(len(classnames)):
            if i != j and cm[i][j] > 0:
                t_style.add('BOX', (j+1, i+1), (j+1, i+1), 1.5, HexColor("#FFCB05"))

    first_col_width = 1.4 * inch
    remaining_width = doc.width - first_col_width
    data_col_width = remaining_width / len(classnames)
    col_widths = [first_col_width] + [data_col_width] * len(classnames)
    
    cm_table = Table(cm_data, colWidths=col_widths)
    cm_table.setStyle(t_style)
    Story.append(cm_table)
    Story.append(Spacer(1, 15))

    # Helper function for side-by-side figure rendering
    def add_figure(jpg_path, title_text, caption_text):
        if os.path.exists(jpg_path):
            img = Image(jpg_path)
            img.drawWidth = 1.9 * inch
            img.drawHeight = 1.9 * inch
            
            desc_text = f"<b>{title_text}</b><br/>{caption_text}"
            desc_para = Paragraph(desc_text, caption_style)
            
            fig_table = Table([[img, desc_para]], colWidths=[2.1 * inch, doc.width - 2.1 * inch])
            fig_table.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (1,0), (1,0), 10),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('TOPPADDING', (0,0), (-1,-1), 4),
            ]))
            Story.append(fig_table)
        else:
            Story.append(Paragraph(f"• <b>{title_text}</b>: {caption_text} (Centroid image not found)", item_style))
        Story.append(Spacer(1, 5))

    # --- Section: Systematic Major Confusions (Decoupled automated diagnostics) ---
    if nlp_major:
        Story.append(Paragraph("Systematic Major Confusions (Auto-Centroids)", h2_style))
        Story.append(Paragraph("Below are the systematic major confusions (where errors exceed 10% of class support) detected in this test run. For each confusion cell, the mathematical centroid has been calculated using Cosine Similarity to capture the structural cluster orientation of behavior cells in the high-dimensional feature space.", body_style))
        Story.append(Spacer(1, 5))
        
        for count, i, j, prop in nlp_major:
            src = classnames[i]
            tgt = classnames[j]
            examples = example_map.get((src, tgt), [])
            if examples:
                avi_path = get_centroid(embedding_map, examples)
                jpg_path = avi_path.replace(".avi", ".jpg")
                file_name = os.path.basename(avi_path)
                
                title_lbl = f"{src} -> {tgt}"
                caption_lbl = f"Detected systematic major confusion of {src} classified as {tgt}.<br/>Error metrics: <b>{count} errors</b> ({prop*100:.1f}% error proportion).<br/><i>Centroid video: {file_name}</i>"
                add_figure(jpg_path, title_lbl, caption_lbl)
        Story.append(Spacer(1, 10))

    # --- Section: Redundant Behaviors (Hypothesis 1) ---
    Story.append(Paragraph("Hypothesis 1: Redundant Behaviors", h2_style))
    Story.append(Paragraph("These behaviors look identical to the software. No further action is required.", body_style))
    
    if not redundant_pairs:
        Story.append(Paragraph("<i>None selected.</i>", item_style))
    else:
        for src, tgt in redundant_pairs:
            Story.append(Paragraph(f"• {src} -> {tgt}", item_style))
    Story.append(Spacer(1, 10))

    # --- Section: Emergent Behaviors (Hypothesis 2) ---
    Story.append(Paragraph("Hypothesis 2: Emergent Behaviors", h2_style))
    Story.append(Paragraph("The software is confusing these behaviors because there is likely a third, completely different behavior happening that it hasn't been taught yet.", body_style))
    Story.append(Paragraph("ACTION REQUIRED: Go to the video folders and manually move the mixed-up videos into a new behavior folder.", action_style))
    
    if not emergent_pairs:
        Story.append(Paragraph("<i>None selected.</i>", item_style))
    else:
        for src, tgt in emergent_pairs:
            examples = example_map.get((src, tgt), [])
            if examples:
                avi_path = get_centroid(embedding_map, examples)
                jpg_path = avi_path.replace(".avi", ".jpg")
                file_name = os.path.basename(avi_path)
                
                title_lbl = f"{src} -> {tgt}"
                caption_lbl = f"Confusion suggests emergent behavior subclass may be present.<br/><i>Reference centroid file: {file_name}</i>"
                add_figure(jpg_path, title_lbl, caption_lbl)
    Story.append(Spacer(1, 10))

    # --- Section: Poor Generalization (Hypothesis 3) ---
    Story.append(Paragraph("Hypothesis 3: Poor Generalization", h2_style))
    Story.append(Paragraph("The software is failing here because it hasn't seen enough different examples of this behavior.", body_style))
    Story.append(Paragraph("ACTION REQUIRED: Add more videos of these specific behaviors in different environments or angles.", action_style))
    
    if not generalization_pairs:
        Story.append(Paragraph("<i>None selected.</i>", item_style))
    else:
        export_dir = os.path.dirname(filepath)
        h3_ref_dir = os.path.join(export_dir, "H3_Variance_References")
        for src, tgt in generalization_pairs:
            Story.append(Paragraph(f"• <b>{src} -> {tgt}</b>", item_style))
            
            safe_source = src.replace(" ", "_")
            safe_target = tgt.replace(" ", "_")
            
            for idx in range(3):
                jpg_name = f"{safe_source}_to_{safe_target}_sample{idx+1}.jpg"
                avi_name = f"{safe_source}_to_{safe_target}_sample{idx+1}.avi"
                jpg_path = os.path.join(h3_ref_dir, jpg_name)
                
                if os.path.exists(jpg_path):
                    title_lbl = f"Sample {idx+1}"
                    caption_lbl = f"Variance reference capture representing class generalization issue.<br/><i>Reference file: H3_Variance_References/{avi_name}</i>"
                    add_figure(jpg_path, title_lbl, caption_lbl)
    Story.append(Spacer(1, 10))

    # --- Baseline References ---
    Story.append(Paragraph("Baseline References (True Positives)", h2_style))
    Story.append(Paragraph("Below are representative examples of correctly classified behaviors. Use these as a visual baseline for what the model currently considers an 'ideal' representation of the class.", body_style))
    
    has_baselines = False
    for i in range(len(classnames)):
        cls_name = classnames[i]
        if cm[i][i] > 0:
            examples = example_map.get((cls_name, cls_name), [])
            if examples:
                has_baselines = True
                avi_path = get_centroid(embedding_map, examples)
                jpg_path = avi_path.replace(".avi", ".jpg")
                file_name = os.path.basename(avi_path)
                
                title_lbl = f"{cls_name} (True Positive)"
                caption_lbl = f"Representative baseline centroid frame showing ideal class representation.<br/><i>Reference centroid file: {file_name}</i>"
                add_figure(jpg_path, title_lbl, caption_lbl)
                
    if not has_baselines:
        Story.append(Paragraph("<i>No correct classifications available to generate baselines.</i>", item_style))

    # Canvas draw helpers for running header & footer on pages 2+
    def draw_first_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Oblique", 8)
        canvas.setFillColor(HexColor("#555555"))
        canvas.drawString(54, 30, "LabGym Practical Diagnostics Pipeline - Triage Action Plan")
        canvas.restoreState()

    def draw_later_pages(canvas, doc):
        canvas.saveState()
        # Running header on top left
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(HexColor("#00274C"))
        canvas.drawString(54, doc.pagesize[1] - 36, f"LabGym Triage Action Plan | Dataset: {dataset_name}")
        
        # Running header thin line below
        canvas.setStrokeColor(HexColor("#CCCCCC"))
        canvas.setLineWidth(0.5)
        canvas.line(54, doc.pagesize[1] - 42, doc.pagesize[0] - 54, doc.pagesize[1] - 42)
        
        # Page number on top right
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.black)
        canvas.drawRightString(doc.pagesize[0] - 54, doc.pagesize[1] - 36, f"Page {doc.page}")
        
        # Running footer
        canvas.setFont("Helvetica-Oblique", 8)
        canvas.setFillColor(HexColor("#555555"))
        canvas.drawString(54, 30, "LabGym Practical Diagnostics Pipeline - Triage Action Plan")
        canvas.restoreState()

    doc.build(Story, onFirstPage=draw_first_page, onLaterPages=draw_later_pages)
    return True

# --- SUBCOMMAND RUNNERS ---

def handle_train(args):
    from LabGym.categorizer import Categorizers
    merged = merge_configs(args, DEFAULT_TRAIN_PARAMS)
    
    data_path = merged.get('data_path')
    model_path = merged.get('model_path')
    network_type = merged.get('network_type')
    
    if not data_path or not model_path:
        print("Error: --data_path and --model_path are required parameters for training.")
        sys.exit(1)
        
    os.makedirs(model_path, exist_ok=True)
    
    CA = Categorizers()
    
    dim_conv = int(merged['dim_conv'])
    dim_tconv = int(merged['dim_tconv'])
    channel = int(merged['channel'])
    time_step = int(merged['time_step'])
    level_conv = int(merged['level_conv'])
    level_tconv = int(merged['level_tconv'])
    aug_methods = merged['aug_methods']
    augvalid = merged['augvalid']
    include_bodyparts = merged['include_bodyparts']
    std = float(merged['std'])
    background_free = merged['background_free']
    black_background = merged['black_background']
    behavior_mode = int(merged['behavior_mode'])
    social_distance = float(merged['social_distance'])
    color_costar = merged['color_costar']
    training_onfly = merged['training_onfly']
    out_path = merged.get('out_path')
    out_folder = merged.get('out_folder')
    
    if not network_type:
        if merged.get('animation_analyzer', True):
            network_type = 'combnet'
        else:
            network_type = 'pattern_recognizer'
            
    print(f"Starting headless training loop. Network type: {network_type}, On-fly: {training_onfly}")
    
    if network_type == 'pattern_recognizer':
        if behavior_mode >= 3:
            time_step = 0
            std = 0
            include_bodyparts = False
        else:
            channel = 3
            
        if training_onfly:
            CA.train_pattern_recognizer_onfly(
                data_path, model_path, out_path=out_path, dim=dim_conv, channel=channel,
                time_step=time_step, level=level_conv, include_bodyparts=include_bodyparts,
                std=std, background_free=background_free, black_background=black_background,
                behavior_mode=behavior_mode, social_distance=social_distance
            )
        else:
            CA.train_pattern_recognizer(
                data_path, model_path, out_path=out_path, dim=dim_conv, channel=channel,
                time_step=time_step, level=level_conv, aug_methods=aug_methods, augvalid=augvalid,
                include_bodyparts=include_bodyparts, std=std, background_free=background_free,
                black_background=black_background, behavior_mode=behavior_mode,
                social_distance=social_distance, out_folder=out_folder
            )
            
    elif network_type == 'animation_analyzer':
        if training_onfly:
            CA.train_animation_analyzer_onfly(
                data_path, model_path, out_path=out_path, dim=dim_tconv, channel=channel,
                time_step=time_step, level=level_tconv, include_bodyparts=include_bodyparts,
                std=std, background_free=background_free, black_background=black_background,
                behavior_mode=behavior_mode, social_distance=social_distance, color_costar=color_costar
            )
        else:
            CA.train_animation_analyzer(
                data_path, model_path, out_path=out_path, dim=dim_tconv, channel=channel,
                time_step=time_step, level=level_tconv, aug_methods=aug_methods, augvalid=augvalid,
                include_bodyparts=include_bodyparts, std=std, background_free=background_free,
                black_background=black_background, behavior_mode=behavior_mode,
                social_distance=social_distance, color_costar=color_costar, out_folder=out_folder
            )
            
    elif network_type == 'combnet' or network_type == 'combined_network':
        if behavior_mode == 2:
            channel = 3
            
        if training_onfly:
            CA.train_combnet_onfly(
                data_path, model_path, out_path=out_path, dim_tconv=dim_tconv, dim_conv=dim_conv,
                channel=channel, time_step=time_step, level_tconv=level_tconv, level_conv=level_conv,
                include_bodyparts=include_bodyparts, std=std, background_free=background_free,
                black_background=black_background, behavior_mode=behavior_mode,
                social_distance=social_distance, color_costar=color_costar
            )
        else:
            CA.train_combnet(
                data_path, model_path, out_path=out_path, dim_tconv=dim_tconv, dim_conv=dim_conv,
                channel=channel, time_step=time_step, level_tconv=level_tconv, level_conv=level_conv,
                aug_methods=aug_methods, augvalid=augvalid, include_bodyparts=include_bodyparts,
                std=std, background_free=background_free, black_background=black_background,
                behavior_mode=behavior_mode, social_distance=social_distance,
                color_costar=color_costar, out_folder=out_folder
            )
    elif network_type == 'detector':
        path_to_annotation = merged.get('path_to_annotation')
        path_to_trainingimages = merged.get('path_to_trainingimages')
        iteration_num = int(merged.get('iteration_num', 200))
        inference_size = int(merged.get('inference_size', 480))
        
        if not path_to_annotation or not path_to_trainingimages:
            print("Error: --path_to_annotation and --path_to_trainingimages are required for detector training.")
            sys.exit(1)
            
        print(f"Starting headless Detector training. Iterations: {iteration_num}, Inference size: {inference_size}")
        from LabGym.detector import Detector
        DT = Detector()
        DT.train(path_to_annotation, path_to_trainingimages, model_path, iteration_num, inference_size)
    else:
        print(f"Error: Unknown network type: {network_type}")
        sys.exit(1)
        
    print("Training finished successfully.")

def handle_test(args):
    import random
    import shutil
    from LabGym.categorizer import Categorizers
    merged = merge_configs(args, {})
    
    model_type = merged.get('model_type', 'categorizer')
    model_path = merged.get('model_path')
    result_path = merged.get('result_path')
    
    if not result_path:
        result_path = os.getcwd()
        
    os.makedirs(result_path, exist_ok=True)
    
    if model_type == 'detector':
        path_to_annotation = merged.get('path_to_annotation')
        groundtruth_path = merged.get('groundtruth_path')
        
        if not path_to_annotation or not groundtruth_path or not model_path:
            print("Error: --path_to_annotation, --groundtruth_path (testing images folder), and --model_path are required for detector testing.")
            sys.exit(1)
            
        print(f"Starting headless Detector testing. Detector: {model_path}, Images: {groundtruth_path}")
        from LabGym.detector import Detector
        DT = Detector()
        DT.test(path_to_annotation, groundtruth_path, model_path, result_path)
        print(f"Detector testing complete! Results saved in: {result_path}")
        return
        
    groundtruth_path = merged.get('groundtruth_path')
    if not groundtruth_path or not model_path:
        print("Error: --groundtruth_path and --model_path are required parameters for testing.")
        sys.exit(1)
        
    CA = Categorizers()
    
    print(f"Starting headless evaluation of model {model_path} on dataset {groundtruth_path}")
    
    report, cm, example_map, embedding_map = CA.test_categorizer(groundtruth_path, model_path, result_path=result_path)
    
    classnames = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    # Pre-calculate errors using proportional math
    cm_rows = len(cm)
    nlp_major = []
    total_support = sum(report.get(classnames[i], {}).get('support', 0) for i in range(cm_rows))
    
    for i in range(cm_rows):
        true_class = classnames[i]
        metrics = report.get(true_class, {})
        support = float(metrics.get('support', 0))
        f1_score = float(metrics.get('f1-score', 0.0))
        
        if support == 0:
            continue
            
        min_support_threshold = max(20, total_support * 0.01)
        if support < min_support_threshold:
            continue
            
        if f1_score >= 0.85:
            continue
            
        for j in range(cm_rows):
            if i != j and cm[i][j] > 0:
                error_count = cm[i][j]
                error_proportion = error_count / support
                if error_proportion >= 0.10:
                    nlp_major.append((error_count, i, j, error_proportion))
                    
    # Parse explicit triage hypotheses mapping if provided (YAML or CLI overrides)
    redundant_raw = merged.get('redundant', [])
    emergent_raw = merged.get('emergent', [])
    generalization_raw = merged.get('generalization', [])
    
    redundant_pairs = []
    emergent_pairs = []
    generalization_pairs = []
    
    for item in redundant_raw:
        pair = parse_pair(item)
        if pair: redundant_pairs.append(pair)
    for item in emergent_raw:
        pair = parse_pair(item)
        if pair: emergent_pairs.append(pair)
    for item in generalization_raw:
        pair = parse_pair(item)
        if pair: generalization_pairs.append(pair)
        
    # By default, compute centroids for ALL major systematic confusions and include them
    all_mapped_pairs = set(redundant_pairs + emergent_pairs + generalization_pairs)
    for count, i, j, prop in nlp_major:
        src = classnames[i]
        tgt = classnames[j]
        if (src, tgt) not in all_mapped_pairs:
            emergent_pairs.append((src, tgt))
            
    # Prepare diagnostic folders
    export_dir = os.path.join(result_path, "LabGym_Diagnostics")
    os.makedirs(export_dir, exist_ok=True)
    pdf_path = os.path.join(export_dir, "Triage_Action_Plan.pdf")
    
    # If there are poor generalization samples, export their H3 Variance references
    if generalization_pairs:
        h3_ref_dir = os.path.join(export_dir, "H3_Variance_References")
        os.makedirs(h3_ref_dir, exist_ok=True)
        for src, tgt in generalization_pairs:
            key = (src, tgt)
            examples = example_map.get(key, [])
            if examples:
                samples = random.sample(examples, min(3, len(examples)))
                for idx, avi_path in enumerate(samples):
                    if os.path.exists(avi_path):
                        safe_source = src.replace(" ", "_")
                        safe_target = tgt.replace(" ", "_")
                        
                        new_avi_name = f"{safe_source}_to_{safe_target}_sample{idx+1}.avi"
                        shutil.copy2(avi_path, os.path.join(h3_ref_dir, new_avi_name))
                        
                        jpg_path = avi_path.replace(".avi", ".jpg")
                        if os.path.exists(jpg_path):
                            new_jpg_name = f"{safe_source}_to_{safe_target}_sample{idx+1}.jpg"
                            shutil.copy2(jpg_path, os.path.join(h3_ref_dir, new_jpg_name))
                            
    print("Generating Triage Action Plan PDF...")
    success = generate_pdf_report(
        pdf_path, os.path.basename(groundtruth_path), report, cm, classnames,
        example_map, embedding_map, nlp_major, redundant_pairs, emergent_pairs, generalization_pairs
    )
    
    if success:
        print(f"Headless evaluation and Automated Diagnostics complete! Output folder: {export_dir}")
    else:
        print("Completed testing but PDF report generation failed due to missing reportlab dependency.")

def handle_infer(args):
    import pandas as pd
    import matplotlib as mpl
    from LabGym.analyzebehavior import AnalyzeAnimal
    from LabGym.analyzebehavior_dt import AnalyzeAnimalDetector
    
    categorizer_path = getattr(args, 'path_to_categorizer', None)
    
    temp_defaults = DEFAULT_INFER_PARAMS.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            yaml_config = yaml.safe_load(f)
            if yaml_config and isinstance(yaml_config, dict):
                categorizer_path = yaml_config.get('path_to_categorizer', categorizer_path)
                
    if categorizer_path and os.path.exists(os.path.join(categorizer_path, 'model_parameters.txt')):
        logger.info(f"Inheriting categorizer settings from {categorizer_path}")
        parameters = pd.read_csv(os.path.join(categorizer_path, 'model_parameters.txt'))
        
        if 'dim_conv' in parameters:
            temp_defaults['dim_conv'] = int(parameters['dim_conv'][0])
        if 'dim_tconv' in parameters:
            temp_defaults['dim_tconv'] = int(parameters['dim_tconv'][0])
        if 'channel' in parameters:
            temp_defaults['channel'] = int(parameters['channel'][0])
        if 'time_step' in parameters:
            temp_defaults['length'] = int(parameters['time_step'][0])
            if temp_defaults['length'] < 3:
                temp_defaults['length'] = 3
        if 'network' in parameters:
            temp_defaults['animation_analyzer'] = (int(parameters['network'][0]) == 2)
        if 'inner_code' in parameters:
            temp_defaults['include_bodyparts'] = (int(parameters['inner_code'][0]) == 0)
            
    merged = merge_configs(args, temp_defaults)
    
    path_to_videos = merged.get('path_to_videos')
    result_path = merged.get('result_path')
    path_to_detector = merged.get('path_to_detector')
    use_detector = merged.get('use_detector')
    
    if not path_to_videos or not result_path:
        print("Error: --path_to_videos and --result_path are required parameters for inference.")
        sys.exit(1)
        
    os.makedirs(result_path, exist_ok=True)
    
    if isinstance(path_to_videos, str):
        path_to_videos = [path_to_videos]
        
    resolved_videos = []
    for p in path_to_videos:
        if os.path.isdir(p):
            video_extensions = ('.mp4', '.mov', '.avi', '.m4v', '.mkv', '.mpg', '.mpeg', '.jpg', '.jpeg', '.png', '.tiff', '.bmp')
            found_files = [
                os.path.join(p, f) for f in os.listdir(p)
                if f.lower().endswith(video_extensions)
            ]
            resolved_videos.extend(found_files)
        else:
            resolved_videos.append(p)
            
    if not resolved_videos:
        print(f"Error: No valid video or image files found in input paths: {path_to_videos}")
        sys.exit(1)
        
    behaviornames_and_colors = {}
    if categorizer_path and os.path.exists(os.path.join(categorizer_path, 'model_parameters.txt')):
        parameters = pd.read_csv(os.path.join(categorizer_path, 'model_parameters.txt'))
        complete_colors = list(mpl.colors.cnames.values())
        colors_list = [['#ffffff', c] for c in complete_colors]
        
        for behavior_name in list(parameters['classnames']):
            idx = list(parameters['classnames']).index(behavior_name)
            if idx < len(colors_list):
                behaviornames_and_colors[behavior_name] = colors_list[idx]
            else:
                behaviornames_and_colors[behavior_name] = ['#ffffff', '#ffffff']
                
    behavior_mode = int(merged['behavior_mode'])
    detector_batch = int(merged['detector_batch'])
    detection_threshold = float(merged['detection_threshold'])
    animal_kinds = merged['animal_kinds']
    animal_to_include = merged['animal_to_include']
    behavior_to_include = merged['behavior_to_include']
    framewidth = merged.get('framewidth')
    delta = float(merged['delta'])
    decode_animalnumber = merged['decode_animalnumber']
    animal_number = parse_json_or_val(merged.get('animal_number'))
    autofind_t = merged['autofind_t']
    decode_t = merged['decode_t']
    t = float(merged['t'])
    duration = float(merged['duration'])
    decode_extraction = merged['decode_extraction']
    ex_start = int(merged['ex_start'])
    ex_end = merged.get('ex_end')
    dim_tconv = int(merged['dim_tconv'])
    dim_conv = int(merged['dim_conv'])
    channel = int(merged['channel'])
    length = int(merged['length'])
    animal_vs_bg = int(merged['animal_vs_bg'])
    stable_illumination = merged['stable_illumination']
    animation_analyzer = merged['animation_analyzer']
    ID_colors = parse_json_or_val(merged['ID_colors'])
    parameter_to_analyze = merged['parameter_to_analyze']
    include_bodyparts = merged['include_bodyparts']
    std = float(merged['std'])
    uncertain = float(merged['uncertain'])
    min_length = merged.get('min_length')
    show_legend = merged['show_legend']
    background_free = merged['background_free']
    black_background = merged['black_background']
    normalize_distance = merged['normalize_distance']
    social_distance = float(merged['social_distance'])
    color_costar = merged['color_costar']
    correct_ID = merged['correct_ID']
    specific_behaviors = parse_json_or_val(merged['specific_behaviors'])
    background_path = merged.get('background_path')
    
    print(f"Loaded behavior inference engine. Starting headless tracking & prediction loop.")
    
    if behavior_mode == 3:  # Static images individual detection
        if not categorizer_path or not path_to_detector:
            print("Error: categorizer_path and path_to_detector are required for image behavior mode 3.")
            sys.exit(1)
            
        if not animal_to_include:
            animal_to_include = animal_kinds
        if detector_batch <= 0:
            detector_batch = 1
        if behavior_to_include and behavior_to_include[0] == 'all':
            behavior_to_include = list(behaviornames_and_colors.keys())
            
        AAD = AnalyzeAnimalDetector()
        AAD.analyze_images_individuals(
            path_to_detector, resolved_videos, result_path, animal_kinds,
            path_to_categorizer=categorizer_path, generate=False,
            animal_to_include=animal_to_include, behavior_to_include=behavior_to_include,
            names_and_colors=behaviornames_and_colors, imagewidth=framewidth,
            dim_conv=dim_conv, channel=channel, detection_threshold=detection_threshold,
            uncertain=uncertain, background_free=background_free,
            black_background=black_background, social_distance=0
        )
    else:  # Videos mode
        all_events = {}
        event_data = {}
        all_time = []
        
        if use_detector:
            for animal_name in animal_kinds:
                all_events[animal_name] = {}
            if not animal_to_include:
                animal_to_include = animal_kinds
            if detector_batch <= 0:
                detector_batch = 1
                
        if not categorizer_path:
            behavior_to_include = []
        else:
            if behavior_to_include and behavior_to_include[0] == 'all':
                behavior_to_include = list(behaviornames_and_colors.keys())
                
        categorize_behavior = (categorizer_path is not None)
        
        for video_file in resolved_videos:
            print(f"Processing video: {video_file}")
            filename = os.path.splitext(os.path.basename(video_file))[0].split('_')
            
            vid_animal_number = animal_number
            vid_t = t
            vid_ex_start = ex_start
            vid_ex_end = ex_end
            
            if decode_animalnumber:
                if use_detector:
                    vid_animal_number = {}
                    number = [x[1:] for x in filename if len(x) > 1 and x[0] == 'n']
                    for a, animal_name in enumerate(animal_kinds):
                        vid_animal_number[animal_name] = int(number[a]) if a < len(number) else 1
                else:
                    for x in filename:
                        if len(x) > 1 and x[0] == 'n':
                            vid_animal_number = int(x[1:])
                            
            if decode_t:
                for x in filename:
                    if len(x) > 1 and x[0] == 'b':
                        vid_t = float(x[1:])
                        
            if decode_extraction:
                for x in filename:
                    if len(x) > 2:
                        if x[:2] == 'xs':
                            vid_ex_start = int(x[2:])
                        if x[:2] == 'xe':
                            vid_ex_end = int(x[2:])
                            
            if vid_animal_number is None:
                if use_detector:
                    vid_animal_number = {animal_name: 1 for animal_name in animal_kinds}
                else:
                    vid_animal_number = 1
                    
            if not use_detector:
                AA = AnalyzeAnimal()
                AA.prepare_analysis(
                    video_file, result_path, vid_animal_number, delta=delta,
                    names_and_colors=behaviornames_and_colors, framewidth=framewidth,
                    stable_illumination=stable_illumination, dim_tconv=dim_tconv,
                    dim_conv=dim_conv, channel=channel, include_bodyparts=include_bodyparts,
                    std=std, categorize_behavior=categorize_behavior,
                    animation_analyzer=animation_analyzer, path_background=background_path,
                    autofind_t=autofind_t, t=vid_t, duration=duration,
                    ex_start=vid_ex_start, ex_end=vid_ex_end, length=length,
                    animal_vs_bg=animal_vs_bg
                )
                if behavior_mode == 0:
                    AA.acquire_information(background_free=background_free, black_background=black_background)
                    AA.craft_data()
                    interact_all = False
                else:
                    AA.acquire_information_interact_basic(background_free=background_free, black_background=black_background)
                    interact_all = True
                    
                if categorize_behavior:
                    AA.categorize_behaviors(categorizer_path, uncertain=uncertain, min_length=min_length)
                
                AA.annotate_video(ID_colors, behavior_to_include, show_legend=show_legend, interact_all=interact_all)
                AA.export_results(normalize_distance=normalize_distance, parameter_to_analyze=parameter_to_analyze)
                
                if categorize_behavior:
                    for n in AA.event_probability:
                        all_events[len(all_events)] = AA.event_probability[n]
                    if len(all_time) < len(AA.all_time):
                        all_time = AA.all_time
            else:
                AAD = AnalyzeAnimalDetector()
                AAD.prepare_analysis(
                    path_to_detector, video_file, result_path, vid_animal_number,
                    animal_kinds, behavior_mode, names_and_colors=behaviornames_and_colors,
                    framewidth=framewidth, dim_tconv=dim_tconv, dim_conv=dim_conv,
                    channel=channel, include_bodyparts=include_bodyparts, std=std,
                    categorize_behavior=categorize_behavior, animation_analyzer=animation_analyzer,
                    t=vid_t, duration=duration, length=length, social_distance=social_distance
                )
                if behavior_mode == 1:
                    AAD.acquire_information_interact_basic(
                        batch_size=detector_batch, background_free=background_free,
                        black_background=black_background
                    )
                else:
                    AAD.acquire_information(
                        batch_size=detector_batch, background_free=background_free,
                        black_background=black_background, color_costar=color_costar
                    )
                if behavior_mode != 1:
                    AAD.craft_data()
                if categorize_behavior:
                    AAD.categorize_behaviors(categorizer_path, uncertain=uncertain, min_length=min_length)
                if correct_ID:
                    AAD.correct_identity(specific_behaviors)
                    
                AAD.annotate_video(animal_to_include, ID_colors, behavior_to_include, show_legend=show_legend)
                AAD.export_results(normalize_distance=normalize_distance, parameter_to_analyze=parameter_to_analyze)
                
                if categorize_behavior:
                    for animal_name in animal_kinds:
                        for n in AAD.event_probability[animal_name]:
                            all_events[animal_name][len(all_events[animal_name])] = AAD.event_probability[animal_name][n]
                    if len(all_time) < len(AAD.all_time):
                        all_time = AAD.all_time
                        
        if categorize_behavior and all_time:
            max_length = len(all_time)
            from LabGym.tools import plot_events
            if not use_detector:
                for n in all_events:
                    event_data[len(event_data)] = all_events[n] + [['NA', -1]] * (max_length - len(all_events[n]))
                all_events_df = pd.DataFrame(event_data, index=all_time)
                all_events_df.to_excel(os.path.join(result_path, 'all_events.xlsx'), float_format='%.2f', index_label='time/ID')
                plot_events(result_path, event_data, all_time, behaviornames_and_colors, behavior_to_include, width=0, height=0)
                
                folders = [i for i in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, i))]
                folders.sort()
                for behavior_name in behaviornames_and_colors:
                    all_summary = []
                    for folder in folders:
                        individual_summary = os.path.join(result_path, folder, behavior_name, 'all_summary.xlsx')
                        if os.path.exists(individual_summary):
                            all_summary.append(pd.read_excel(individual_summary))
                    if len(all_summary) >= 1:
                        all_summary = pd.concat(all_summary, ignore_index=True)
                        all_summary.to_excel(os.path.join(result_path, behavior_name + '_summary.xlsx'), float_format='%.2f', index_label='ID/parameter')
            else:
                for animal_name in animal_to_include:
                    for n in all_events[animal_name]:
                        event_data[len(event_data)] = all_events[animal_name][n] + [['NA', -1]] * (max_length - len(all_events[animal_name][n]))
                    event_data[len(event_data)] = [['NA', -1]] * max_length
                del event_data[len(event_data) - 1]
                
                all_events_df = pd.DataFrame(event_data, index=all_time)
                all_events_df.to_excel(os.path.join(result_path, 'all_events.xlsx'), float_format='%.2f', index_label='time/ID')
                plot_events(result_path, event_data, all_time, behaviornames_and_colors, behavior_to_include, width=0, height=0)
                
                folders = [i for i in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, i))]
                folders.sort()
                for animal_name in animal_kinds:
                    for behavior_name in behaviornames_and_colors:
                        all_summary = []
                        for folder in folders:
                            individual_summary = os.path.join(result_path, folder, behavior_name, animal_name + '_all_summary.xlsx')
                            if os.path.exists(individual_summary):
                                all_summary.append(pd.read_excel(individual_summary))
                        if len(all_summary) >= 1:
                            all_summary = pd.concat(all_summary, ignore_index=True)
                            all_summary.to_excel(os.path.join(result_path, animal_name + '_' + behavior_name + '_summary.xlsx'), float_format='%.2f', index_label='ID/parameter')
                            
    print("Headless inference finished successfully.")

# --- CLI ENTRYPOINT ---

def main():
    blue_color = "\033[94m"
    pink_color = "\033[95m"
    reset_color = "\033[0m"

    ascii_banner = (
        f"{blue_color}██╗      █████╗ ██████╗  {pink_color}██████╗██╗   ██╗███╗   ███╗{reset_color}\n"
        f"{blue_color}██║     ██╔══██╗██╔══██╗{pink_color}██╔════╝╚██╗ ██╔╝████╗ ████║{reset_color}\n"
        f"{blue_color}██║     ███████║██████╔╝{pink_color}██║  ███╗╚████╔╝ ██╔████╔██║{reset_color}\n"
        f"{blue_color}██║     ██╔══██║██╔══██╗{pink_color}██║   ██║ ╚██╔╝  ██║╚██╔╝██║{reset_color}\n"
        f"{blue_color}███████╗██║  ██║██████╔╝{pink_color}╚██████╔╝  ██║   ██║ ╚═╝ ██║{reset_color}\n"
        f"{blue_color}╚══════╝╚═╝  ╚═╝╚═════╝  {pink_color}╚═════╝   ╚═╝   ╚═╝     ╚═╝{reset_color}"
    )

    class CustomHelpFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
        def _format_action(self, action):
            formatted = super()._format_action(action)
            if not action.option_strings:
                return formatted
                
            blue_color = "\033[94m"
            pink_color = "\033[95m"
            reset_color = "\033[0m"
            
            dest = action.dest
            detector_dests = {
                'path_to_annotation', 'path_to_trainingimages', 'iteration_num', 'inference_size',
                'path_to_detector', 'detector_batch', 'detection_threshold', 'animal_kinds'
            }
            categorizer_dests = {
                'data_path', 'out_folder', 'dim_conv', 'dim_tconv', 'channel', 'time_step',
                'level_conv', 'level_tconv', 'aug_methods', 'augvalid', 'include_bodyparts',
                'std', 'background_free', 'black_background', 'behavior_mode', 'social_distance',
                'color_costar', 'training_onfly', 'path_to_categorizer', 'behavior_to_include',
                'parameter_to_analyze', 'uncertain', 'min_length', 'specific_behaviors',
                'groundtruth_path', 'redundant', 'emergent', 'generalization'
            }
            
            color = None
            if dest in detector_dests:
                color = blue_color
            elif dest in categorizer_dests:
                color = pink_color
                
            if color:
                for opt in action.option_strings:
                    formatted = formatted.replace(opt, f"{color}{opt}{reset_color}")
            return formatted

    parser = argparse.ArgumentParser(
        description=ascii_banner + "\n\nLabGym CLI: Headless command-line interface for training, testing, and behavioral inference.",
        formatter_class=CustomHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help="Available subcommands")
    
    # Train command parser
    train_parser = subparsers.add_parser('train', help="Train sub-networks headlessly", formatter_class=CustomHelpFormatter)
    train_parser.add_argument('--config', type=str, help="Path to YAML configuration file specifying baseline settings")
    train_parser.add_argument('--network_type', type=str, choices=['pattern_recognizer', 'animation_analyzer', 'combnet', 'combined_network', 'detector'], help="Type of network to train")
    train_parser.add_argument('--data_path', type=str, help="Path to training examples dataset (for Categorizer)")
    train_parser.add_argument('--model_path', type=str, help="Path to folder where output trained model / detector will be stored")
    train_parser.add_argument('--out_path', type=str, help="Output path (e.g. for checkpoint saves)")
    train_parser.add_argument('--dim_conv', type=int, help="Input dimension for Pattern Recognizer")
    train_parser.add_argument('--dim_tconv', type=int, help="Input dimension for Animation Analyzer")
    train_parser.add_argument('--channel', type=int, help="Input channel depth (1 or 3)")
    train_parser.add_argument('--time_step', type=int, help="Time steps/frames per behavior example")
    train_parser.add_argument('--level_conv', type=int, help="Complexity level of Pattern Recognizer")
    train_parser.add_argument('--level_tconv', type=int, help="Complexity level of Animation Analyzer")
    train_parser.add_argument('--aug_methods', type=str, nargs='+', help="List of augmentation methods to apply")
    train_parser.add_argument('--augvalid', type=str2bool, nargs='?', const=True, help="Validate augmentation data")
    train_parser.add_argument('--include_bodyparts', type=str2bool, nargs='?', const=True, help="Include body parts in calculations")
    train_parser.add_argument('--std', type=float, help="STD parameter for body part filtering")
    train_parser.add_argument('--background_free', type=str2bool, nargs='?', const=True, help="Exclude background from analysis")
    train_parser.add_argument('--black_background', type=str2bool, nargs='?', const=True, help="Use black background for animations")
    train_parser.add_argument('--behavior_mode', type=int, help="Behavior mode selection (e.g. 0, 1, 2, 3)")
    train_parser.add_argument('--social_distance', type=float, help="Interaction social distance constraint")
    train_parser.add_argument('--color_costar', type=str2bool, nargs='?', const=True, help="Keep supporting actors in color")
    train_parser.add_argument('--training_onfly', type=str2bool, nargs='?', const=True, help="Train using on-the-fly generated behavior examples")
    train_parser.add_argument('--out_folder', type=str, help="Target folder for caching training frames")
    train_parser.add_argument('--path_to_annotation', type=str, help="Path to COCO JSON annotation file (for Detector training)")
    train_parser.add_argument('--path_to_trainingimages', type=str, help="Path to training images folder (for Detector training)")
    train_parser.add_argument('--iteration_num', type=int, help="Number of training iterations (for Detector training)")
    train_parser.add_argument('--inference_size', type=int, help="Inferencing frame size (for Detector training)")
    train_parser.set_defaults(func=handle_train)
    
    # Test command parser
    test_parser = subparsers.add_parser('test', help="Evaluate a model on a ground-truth dataset", formatter_class=CustomHelpFormatter)
    test_parser.add_argument('--config', type=str, help="Path to YAML configuration file specifying baseline settings")
    test_parser.add_argument('--model_type', type=str, choices=['categorizer', 'detector'], default='categorizer', help="Type of model under evaluation")
    test_parser.add_argument('--path_to_annotation', type=str, help="Path to COCO JSON annotation file (required for detector testing)")
    test_parser.add_argument('--groundtruth_path', type=str, help="Path to testing ground-truth dataset folder (or images directory for detector)")
    test_parser.add_argument('--model_path', type=str, help="Path to model directory under evaluation")
    test_parser.add_argument('--result_path', type=str, help="Folder to write test statistics and diagnostics reports")
    test_parser.add_argument('--redundant', type=str, nargs='+', help="List of confusions to assign to Redundant Behaviors hypothesis")
    test_parser.add_argument('--emergent', type=str, nargs='+', help="List of confusions to assign to Emergent Behaviors hypothesis")
    test_parser.add_argument('--generalization', type=str, nargs='+', help="List of confusions to assign to Poor Generalization hypothesis")
    test_parser.set_defaults(func=handle_test)
    
    # Infer command parser
    infer_parser = subparsers.add_parser('infer', help="Execute animal tracking and behavior classification on videos or images", formatter_class=CustomHelpFormatter)
    infer_parser.add_argument('--config', type=str, help="Path to YAML configuration file specifying baseline settings")
    infer_parser.add_argument('--path_to_videos', type=str, nargs='+', help="Path to videos or directory containing videos for analysis")
    infer_parser.add_argument('--result_path', type=str, help="Path to write inference output files")
    infer_parser.add_argument('--path_to_detector', type=str, help="Path to object detector model")
    infer_parser.add_argument('--path_to_categorizer', type=str, help="Path to behavior classifier model")
    infer_parser.add_argument('--use_detector', type=str2bool, nargs='?', const=True, help="Use object detector for tracking")
    infer_parser.add_argument('--animal_kinds', type=str, nargs='+', help="Categories of animals/objects in detector")
    infer_parser.add_argument('--animal_to_include', type=str, nargs='+', help="Animals to keep in outputs")
    infer_parser.add_argument('--behavior_to_include', type=str, nargs='+', help="Behaviors to categorize")
    infer_parser.add_argument('--detector_batch', type=int, help="Detection processing batch size")
    infer_parser.add_argument('--detection_threshold', type=float, help="Detection threshold (static images mode)")
    infer_parser.add_argument('--background_path', type=str, help="Path to background reference image")
    infer_parser.add_argument('--framewidth', type=int, help="Target resizing width for processing frames")
    infer_parser.add_argument('--delta', type=float, help="Optogenetics stimulation delta trigger threshold")
    infer_parser.add_argument('--decode_animalnumber', type=str2bool, nargs='?', const=True, help="Extract animal counts from video file names")
    infer_parser.add_argument('--animal_number', type=str, help="Fixed integer or json mapping of animal counts")
    infer_parser.add_argument('--autofind_t', type=str2bool, nargs='?', const=True, help="Auto-calculate optogenetics trigger start")
    infer_parser.add_argument('--decode_t', type=str2bool, nargs='?', const=True, help="Extract start times from file names")
    infer_parser.add_argument('--t', type=float, help="Start time offset for behavior analysis")
    infer_parser.add_argument('--duration', type=float, help="Total analysis window duration in seconds")
    infer_parser.add_argument('--decode_extraction', type=str2bool, nargs='?', const=True, help="Extract start/end times for extraction")
    infer_parser.add_argument('--ex_start', type=int, help="Extraction start frame number")
    infer_parser.add_argument('--ex_end', type=int, help="Extraction end frame number")
    infer_parser.add_argument('--dim_tconv', type=int, help="Input size for Animation Analyzer")
    infer_parser.add_argument('--dim_conv', type=int, help="Input size for Pattern Recognizer")
    infer_parser.add_argument('--channel', type=int, help="Input channel size")
    infer_parser.add_argument('--length', type=int, help="Temporal window frame length")
    infer_parser.add_argument('--animal_vs_bg', type=int, help="Animal brightness relative to background")
    infer_parser.add_argument('--stable_illumination', type=str2bool, nargs='?', const=True, help="Illumination stability flag")
    infer_parser.add_argument('--animation_analyzer', type=str2bool, nargs='?', const=True, help="Activate Animation Analyzer component")
    infer_parser.add_argument('--ID_colors', type=str, help="Colormap sequence mapping IDs to colors")
    infer_parser.add_argument('--parameter_to_analyze', type=str, nargs='+', help="Target kinematics parameters to extract")
    infer_parser.add_argument('--include_bodyparts', type=str2bool, nargs='?', const=True, help="Include bodyparts details")
    infer_parser.add_argument('--std', type=float, help="Filter STD parameter value")
    infer_parser.add_argument('--uncertain', type=float, help="Maximum classification uncertainty gap threshold")
    infer_parser.add_argument('--min_length', type=int, help="Minimum behavior validation length in frames")
    infer_parser.add_argument('--show_legend', type=str2bool, nargs='?', const=True, help="Show behavioral legend overlays")
    infer_parser.add_argument('--background_free', type=str2bool, nargs='?', const=True, help="Remove environment background from calculations")
    infer_parser.add_argument('--black_background', type=str2bool, nargs='?', const=True, help="Convert background elements to solid black")
    infer_parser.add_argument('--normalize_distance', type=str2bool, nargs='?', const=True, help="Normalize metrics relative to contour sizes")
    infer_parser.add_argument('--social_distance', type=float, help="Social distance radius parameter")
    infer_parser.add_argument('--color_costar', type=str2bool, nargs='?', const=True, help="Render costar characters in RGB scale")
    infer_parser.add_argument('--specific_behaviors', type=str, help="Specific behavior definitions data object")
    infer_parser.add_argument('--correct_ID', type=str2bool, nargs='?', const=True, help="Run ID correction heuristics")
    infer_parser.set_defaults(func=handle_infer)
    
    args = parser.parse_args()
    
    if not args.command:
        try:
            print("Starting LabGym GUI...")
            from LabGym.__main__ import main as gui_main
            gui_main()
        except Exception as e:
            print(f"Error starting GUI: {e}")
            print("\nFor headless CLI help, use: labgym --help")
            sys.exit(1)
        sys.exit(0)
        
    args.func(args)

if __name__ == '__main__':
    main()
