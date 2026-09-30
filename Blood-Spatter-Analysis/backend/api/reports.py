"""Reports API — generate and download analysis reports."""

import json
from datetime import datetime
from pathlib import Path

from flask import Blueprint, request, jsonify, send_file

from backend.config import Config
from backend.database.models import AnalysisModel, CaseModel, get_db
from backend.utils.logger import get_logger

log = get_logger(__name__)
reports_bp = Blueprint("reports", __name__, url_prefix="/api/report")

DISCLAIMER = (
    "This software provides computational estimates for academic and research purposes. "
    "Results should not be treated as definitive forensic conclusions and "
    "require validation by qualified forensic professionals."
)


def _get_models():
    db = get_db(Config.DB_PATH)
    return CaseModel(db), AnalysisModel(db)


@reports_bp.route("/generate", methods=["POST"])
def generate_report():
    data = request.get_json(silent=True) or {}
    case_id = data.get("case_id", "")
    if not case_id:
        return jsonify({"error": "case_id is required."}), 400

    case_model, analysis_model = _get_models()
    case = case_model.get_by_id(case_id)
    if not case:
        return jsonify({"error": "Case not found."}), 404

    analyses = analysis_model.get_by_case(case_id)

    report = _build_report(case, analyses)
    report_path = Config.REPORTS_FOLDER / f"{case_id}_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    # Update case with report reference
    case_model.update(case_id, {"report_path": str(report_path)})

    # Try to generate PDF
    pdf_path = _try_generate_pdf(report, case_id)

    return jsonify({
        "status":      "success",
        "report":      report,
        "json_path":   str(report_path),
        "pdf_path":    str(pdf_path) if pdf_path else None,
        "pdf_available": pdf_path is not None,
    })


@reports_bp.route("/download/<case_id>", methods=["GET"])
def download_report(case_id):
    fmt = request.args.get("format", "json")
    if fmt == "pdf":
        pdf_path = Config.REPORTS_FOLDER / f"{case_id}_report.pdf"
        if pdf_path.exists():
            return send_file(str(pdf_path), as_attachment=True, download_name=f"BPA_Report_{case_id}.pdf")
        return jsonify({"error": "PDF not found. Generate report first."}), 404

    json_path = Config.REPORTS_FOLDER / f"{case_id}_report.json"
    if not json_path.exists():
        return jsonify({"error": "Report not found. Generate it first."}), 404
    return send_file(str(json_path), as_attachment=True, download_name=f"BPA_Report_{case_id}.json")


# ------------------------------------------------------------------ #

def _build_report(case: dict, analyses: list) -> dict:
    """Aggregate all analysis results into a structured report."""
    classification_results = []
    droplet_summaries = []
    convergence_results = []
    reconstruction_results = []

    for a in analyses:
        if "classification" in a:
            classification_results.append(a["classification"])
        if "droplets" in a:
            droplet_summaries.append(a["droplets"])
        if "convergence_2d" in a:
            convergence_results.append(a["convergence_2d"])
        if "reconstruction_3d" in a:
            reconstruction_results.append(a["reconstruction_3d"])

    return {
        "report_title": "Deep Learning-Based Bloodstain Pattern Analysis Report",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "disclaimer": DISCLAIMER,
        "case": {
            "id":          case["id"],
            "name":        case["name"],
            "description": case.get("description", ""),
            "created_at":  case.get("created_at", ""),
            "images":      case.get("images", []),
        },
        "classification": classification_results,
        "droplet_analysis": droplet_summaries,
        "convergence_2d": convergence_results,
        "reconstruction_3d": reconstruction_results,
        "summary": {
            "total_images":           len(case.get("images", [])),
            "classifications_run":    len(classification_results),
            "droplet_analyses_run":   len(droplet_summaries),
            "convergences_computed":  len(convergence_results),
            "reconstructions_computed": len(reconstruction_results),
        },
    }


def _try_generate_pdf(report: dict, case_id: str) -> Path | None:
    """Generate a PDF using fpdf2 if available."""
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "BPA Analysis Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, f"Case: {report['case']['name']}", ln=True)
        pdf.cell(0, 6, f"Generated: {report['generated_at']}", ln=True)
        pdf.ln(4)

        # Disclaimer
        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(0, 5, report["disclaimer"])
        pdf.ln(4)

        # Case info
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Case Information", ln=True)
        pdf.set_font("Helvetica", "", 10)
        c = report["case"]
        for k, v in [("Case ID", c["id"]), ("Name", c["name"]),
                     ("Description", c.get("description", "N/A")),
                     ("Created", c.get("created_at", "N/A")),
                     ("Images", str(len(c.get("images", []))))]:
            pdf.cell(50, 6, f"{k}:", ln=False)
            pdf.cell(0, 6, str(v), ln=True)
        pdf.ln(4)

        # Classification
        if report["classification"]:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Classification Results", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for cls in report["classification"]:
                s = cls.get("spatter", {})
                v = cls.get("velocity", {})
                pdf.cell(0, 6, f"Spatter Type (AI): {s.get('predicted_type', 'N/A')} ({s.get('confidence', 0):.1f}%)", ln=True)
                pdf.cell(0, 6, f"Velocity Category: {v.get('category', 'N/A')} ({v.get('confidence', 0):.1f}%)", ln=True)
            pdf.ln(4)

        # Droplets
        if report["droplet_analysis"]:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Droplet Analysis", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for da in report["droplet_analysis"]:
                s = da.get("summary", {})
                pdf.cell(0, 6, f"Droplets detected: {s.get('count', 0)}", ln=True)
                if s.get("avg_angle") is not None:
                    pdf.cell(0, 6, f"Avg. Impact Angle: {s['avg_angle']:.1f}°", ln=True)
            pdf.ln(4)

        # Convergence
        if report["convergence_2d"]:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "2D Convergence", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for conv in report["convergence_2d"]:
                if conv.get("status") == "success":
                    pdf.cell(0, 6, f"Convergence Point: ({conv.get('convergence_x'):.1f}, {conv.get('convergence_y'):.1f}) px", ln=True)
                    pdf.cell(0, 6, f"RMS Error: {conv.get('error_px'):.2f} px", ln=True)
            pdf.ln(4)

        # 3D Point of Origin
        if report["reconstruction_3d"]:
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "3D Point of Origin (Estimated)", ln=True)
            pdf.set_font("Helvetica", "", 10)
            for rec in report["reconstruction_3d"]:
                if rec.get("status") == "success":
                    pdf.cell(0, 6, f"X={rec.get('x'):.2f}, Y={rec.get('y'):.2f}, Z={rec.get('z'):.2f}", ln=True)
                    pdf.cell(0, 6, f"Estimation Error: {rec.get('error_units'):.2f} units", ln=True)

        pdf_path = Config.REPORTS_FOLDER / f"{case_id}_report.pdf"
        pdf.output(str(pdf_path))
        log.info(f"PDF report saved: {pdf_path}")
        return pdf_path

    except ImportError:
        log.warning("fpdf2 not installed — PDF generation skipped.")
        return None
    except Exception as e:
        log.exception(f"PDF generation failed: {e}")
        return None
