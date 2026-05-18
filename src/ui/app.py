"""Asteroid Hazard Predictor — Gradio UI."""
from datetime import date, timedelta
import gradio as gr

from src.ui.predictions_service import PredictionsService
from src.utils import get_logger

log = get_logger(__name__)

_service: PredictionsService | None = None

def get_service() -> PredictionsService:
    global _service
    if _service is None:
        _service = PredictionsService()
    return _service

# ───────────────────────────────────────────
# Custom theme & CSS for a cosmic look
# ───────────────────────────────────────────
theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

CSS = """
.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
    min-height: 100vh;
}
#title { text-align: center; padding: 1rem 0; }
#title h1 {
    background: linear-gradient(90deg, #c7d2fe, #f0abfc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.5rem !important; margin: 0;
}
#title p { color: #cbd5e1; margin-top: .25rem; }
.stat-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px; padding: 1rem; text-align: center;
}
.stat-card h2 { color: #f0abfc; font-size: 2rem; margin: 0; }
.stat-card p  { color: #94a3b8; font-size: 0.85rem; margin: 0; }
"""

# ───────────────────────────────────────────
# Callbacks
# ───────────────────────────────────────────
def load(start_str: str, end_str: str):
    service = get_service()
    try:
        start = date.fromisoformat(start_str)
        end = date.fromisoformat(end_str)
    except ValueError:
        return None, _stat_html({}), "Invalid date format (use YYYY-MM-DD)"

    df_raw  = service.get_by_range(start, end)
    stats   = service.compute_stats(df_raw)
    df_view = service.format_for_display(df_raw)
    msg = f"Loaded **{stats['total']}** predictions" if stats['total'] else "No predictions in this range."
    return df_view, _stat_html(stats), msg

def load_all():
    service = get_service()
    df_raw  = service.get_all()
    stats   = service.compute_stats(df_raw)
    df_view = service.format_for_display(df_raw)
    total   = stats.get("total", 0)
    msg = f"Loaded **{total}** predictions (full history)" if total else "No predictions found."
    return df_view, _stat_html(stats), msg

def preset(days: int):
    today = date.today()
    start = today - timedelta(days=days)
    return start.isoformat(), today.isoformat()

def _stat_html(s: dict) -> str:
    if not s or s.get("total", 0) == 0:
        s = {
            "total": 0, "hazardous": 0, "safe": 0, "avg_prob": 0.0,
            "model": "—", "predicted": 0,
            "next_train": "—", "history_days": 0,
            v
        }
    return f"""
<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;">
  <div class="stat-card"><h2>{s['total']}</h2><p>Total</p></div>
  <div class="stat-card"><h2 style="color:#fca5a5">{s['hazardous']}</h2><p>Hazardous</p></div>
  <div class="stat-card"><h2 style="color:#86efac">{s['safe']}</h2><p>Safe</p></div>
  <div class="stat-card"><h2>{s['avg_prob']*100:.2f}%</h2><p>Avg Hazard Prob</p></div>
  <div class="stat-card"><h2>{s['model']}</h2><p>Champion Model</p></div>
  <div class="stat-card"><h2 style="color:#c4b5fd">{s.get('predicted', 0)}</h2><p>Real Predictions</p></div>
  <div class="stat-card"><h2 style="color:#fbbf24">{s.get('next_train', '—')}</h2>
  <p>Next Retrain</p>
  <p style="font-size:0.72rem; color:#64748b; margin-top:0.3rem">
    trained till {s.get('training_cutoff', '—')}
  </p>
</div>
  <div class="stat-card"><h2 style="color:#67e8f9">{s['history_days']}d</h2><p>Days of History</p></div>
</div>"""

# ───────────────────────────────────────────
# Layout
# ───────────────────────────────────────────
with gr.Blocks(theme=theme, css=CSS, title="Asteroid Hazard Predictor") as app:
    gr.HTML("""
    <div id="title">
      <h1>Asteroid Hazard Predictor</h1>
    </div>""")

    stats_box = gr.HTML(_stat_html({}))

    with gr.Row():
        with gr.Column(scale=1):
            start_in = gr.Textbox(label="Start", value=date.today().isoformat())
            end_in   = gr.Textbox(label="End",   value=date.today().isoformat())
            with gr.Row():
                btn_today = gr.Button("Today",      size="sm")
                btn_7     = gr.Button("Last 7d",    size="sm")
                btn_all = gr.Button("All History", size="sm")
            load_btn = gr.Button("Load Predictions", variant="primary")
            status = gr.Markdown()

        with gr.Column(scale=3):
            table = gr.Dataframe(
                label="Predictions (sorted by hazard probability)",
                interactive=False, wrap=True,
            )

    # wire up
    btn_today.click(lambda: preset(0),  outputs=[start_in, end_in])
    btn_7.click(    lambda: preset(7),  outputs=[start_in, end_in])
    btn_all.click(load_all, outputs=[table, stats_box, status])
    load_btn.click(load, inputs=[start_in, end_in], outputs=[table, stats_box, status])
    app.load(lambda: load(date.today().isoformat(), date.today().isoformat()),
             outputs=[table, stats_box, status])


def main():
    app.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()