"""
Screening report generator (Phase 6).
Creates a simple HTML report for a screening result.
"""
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

def generate_report(screening_id: str, patient_data: dict, screening_data: dict,
                    output_dir: str = "app/reports") -> str:
    """
    Generate a printable HTML screening report.
    Returns the path to the saved report file.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"report_{screening_id}.html")

    p = patient_data
    s = screening_data
    is_dr = s["prediction"] >= 0.5
    result_label = "DR Present" if is_dr else "No DR Detected"
    risk_class = "risk" if is_dr else "ok"

    # Image URLs (relative to /uploads for browser access)
    img_rel = s.get("image_path", "").replace("\\", "/")
    gapcam_rel = s.get("gapcam_path", "").replace("\\", "/")
    img_url = "/" + img_rel if img_rel else None
    gapcam_url = "/" + gapcam_rel if gapcam_rel else None

    # Use screening date if available; otherwise now in IST
    s_date_raw = s.get("screening_date")
    if s_date_raw and isinstance(s_date_raw, str) and "T" in s_date_raw:
        # ISO string from API: parse and format
        try:
            s_dt = datetime.fromisoformat(s_date_raw)
            if s_dt.tzinfo is None:
                s_dt = s_dt.replace(tzinfo=ZoneInfo("UTC"))
            report_date = s_dt.astimezone(IST).strftime("%d %B %Y, %H:%M")
        except Exception:
            report_date = datetime.now(IST).strftime("%d %B %Y, %H:%M")
    else:
        report_date = datetime.now(IST).strftime("%d %B %Y, %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DR Screening Report — {p['patient_id']} — {s['screening_id']}</title>
<style>
  *{{box-sizing:border-box;font-family:"Segoe UI",system-ui,sans-serif;}}
  body{{margin:0;padding:2rem;background:#f4f7f8;color:#1a2e35;line-height:1.5;}}
  .page{{max-width:800px;margin:0 auto;background:#fff;border-radius:12px;
         box-shadow:0 4px 20px rgba(0,0,0,.1);overflow:hidden;}}
  .header{{background:linear-gradient(135deg,#122d3a,#0d7377);color:#fff;padding:1.5rem 2rem;}}
  .header h1{{margin:0;font-size:1.5rem;}}
  .header p{{margin:4px 0 0;opacity:.9;font-size:.9rem;}}
  .body{{padding:1.5rem 2rem;}}
  .section{{margin-bottom:1.2rem;}}
  .section h2{{font-size:1rem;color:#0d7377;border-bottom:2px solid #0d7377;
               padding-bottom:4px;margin-bottom:.6rem;}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:1rem;}}
  .field{{}}
  .field .label{{font-size:.78rem;color:#657786;text-transform:uppercase;letter-spacing:.5px;}}
  .field .value{{font-weight:600;font-size:.95rem;}}
  .result-banner{{padding:1.2rem;border-radius:10px;text-align:center;margin:.8rem 0;
                  background:{"#fdf0f0" if is_dr else "#e8f6f3"};}}
  .result-banner h3{{margin:0 0 .3rem;font-size:1.3rem;color:{"#c0392b" if is_dr else "#1a7a4e"};}}
  .result-banner .conf{{font-size:1.05rem;margin:.2rem 0;}}
  .result-banner .prob{{color:#657786;font-size:.85rem;}}
  .images{{display:flex;gap:1rem;flex-wrap:wrap;margin:.8rem 0;}}
  .img-box{{flex:1;min-width:200px;}}
  .img-box img{{width:100%;border-radius:8px;border:1px solid #dde6e7;}}
  .img-box p{{text-align:center;font-size:.82rem;color:#657786;margin:4px 0 0;}}
  .disclaimer{{margin-top:1.2rem;padding:.8rem;background:#fff8e6;border-radius:6px;
               font-size:.82rem;color:#8a6d1f;border-left:4px solid #e67e22;}}
  .footer{{padding:.8rem 2rem;background:#f8f9fa;text-align:center;
           font-size:.78rem;color:#9aa5ab;border-top:1px solid #dde6e7;}}
  .meta{{display:flex;gap:1.5rem;flex-wrap:wrap;font-size:.85rem;color:#657786;}}
  @media print{{body{{background:#fff;padding:0;}}.page{{box-shadow:none;border-radius:0;}}}}
</style>
</head>
<body>
<div class="page">
  <div class="header">
    <h1>SIH-2026 Diabetic Retinopathy Screening Report</h1>
    <p>AI-Assisted Screening | EfficientNetV2B0 (INT8) | AUC 0.996</p>
  </div>
  <div class="body">
    <div class="section">
      <h2>Patient Information</h2>
      <div class="grid">
        <div class="field"><div class="label">Patient ID</div><div class="value">{p['patient_id']}</div></div>
        <div class="field"><div class="label">Name</div><div class="value">{p.get('name','N/A')}</div></div>
        <div class="field"><div class="label">Age</div><div class="value">{p.get('age','N/A')}</div></div>
        <div class="field"><div class="label">Gender</div><div class="value">{p.get('gender','N/A')}</div></div>
        <div class="field"><div class="label">Contact</div><div class="value">{p.get('phone','N/A')}</div></div>
        <div class="field"><div class="label">Registered</div><div class="value">{p.get('created_at','N/A')}</div></div>
      </div>
    </div>

    <div class="section">
      <h2>Screening Details</h2>
      <div class="grid">
        <div class="field"><div class="label">Screening ID</div><div class="value">{s['screening_id']}</div></div>
        <div class="field"><div class="label">Date</div><div class="value">{s.get('screening_date', report_date)}</div></div>
        <div class="field"><div class="label">Model Version</div><div class="value">EfficientNetV2B0 v1.0.0</div></div>
        <div class="field"><div class="label">Decision Threshold</div><div class="value">0.50</div></div>
      </div>
    </div>

    <div class="result-banner {risk_class}">
      <h3>AI Screening Result: {result_label}</h3>
      <div class="conf">Confidence: <b>{(s['confidence']*100):.1f}%</b></div>
      <div class="prob">Raw probability: {s['prediction']:.4f} (threshold=0.50)</div>
      <div class="conf" style="margin-top:.3rem;font-size:.88rem;">
        {"DR features detected — refer to ophthalmologist for clinical examination" if is_dr else "No DR features detected — routine follow-up recommended"}
      </div>
    </div>

    {"<div class=\"images\"><div class=\"img-box\"><img src=\"" + gapcam_url + "\" alt=\"Retinal Image with GAP-CAM Heatmap\"><p>Retinal Image with GAP-CAM Heatmap</p></div><div class=\"img-box\"><img src=\"" + img_url + "\" alt=\"Original Retinal Image\"><p>Original Retinal Image</p></div></div>" if gapcam_url and img_url else "<p><em>Image not available</em></p>"}

    <div class="section">
      <h2>Model Performance Reference</h2>
      <div class="grid">
        <div class="field"><div class="label">Test AUC-ROC</div><div class="value">0.996</div></div>
        <div class="field"><div class="label">Test Sensitivity</div><div class="value">96.1%</div></div>
        <div class="field"><div class="label">Test Specificity</div><div class="value">97.0%</div></div>
        <div class="field"><div class="label">Test Accuracy</div><div class="value">96.5%</div></div>
      </div>
    </div>

    <div class="section">
      <h2>Clinician Notes</h2>
      <div style="padding:.6rem;background:#f8f9fa;border-radius:6px;min-height:60px;
                  border:1px solid #dde6e7;">{s.get('clinician_notes', '<em style="color:#9aa5ab;">No notes recorded.</em>')}</div>
    </div>

    <div class="disclaimer">
      <strong>Disclaimer:</strong> This AI screening result is intended as a decision-support tool for trained healthcare staff and is not a substitute for evaluation by a qualified medical professional. Always refer to an ophthalmologist for confirmation of any positive screening result.
    </div>
  </div>
  <div class="footer">
    SIH-2026 DR Screening | Generated: {report_date} | Report ID: {s['screening_id']}
  </div>
</div>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path
