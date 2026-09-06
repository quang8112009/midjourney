"""Build the 120-image blind depth labelling tool and anonymised directory.

Strictly follows:
- 120 images (60 OFF, 60 ON; 60 in_front_of, 60 behind)
- Anonymised filenames: img_001.png ... img_120.png
- Hidden mapping file (not printed to output)
- Standalone, offline HTML tool with single-image step-through, standing criteria,
  Yes/No/Can't tell buttons, Back/Next navigation, localStorage caching, and CSV export.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from PIL import Image

from scripts.eval_spatial_depth_dedicated import (
    DEPTH_24_SPECS,
    MonocularDepthEvaluator,
)
from scripts.eval_spatial_rigorous_benchmark import StrictSpatialEvaluator

SEEDS_5 = [42, 100, 2024, 7777, 12345]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator_2d = StrictSpatialEvaluator(device=device)
    evaluator_3d = MonocularDepthEvaluator(device=device)

    src_dir = ROOT_DIR / "benchmarks" / "images" / "depth_dedicated_n192"
    target_dir = ROOT_DIR / "benchmarks" / "blind_depth_evaluation"
    img_target_dir = target_dir / "images"

    if target_dir.exists():
        shutil.rmtree(target_dir)

    img_target_dir.mkdir(parents=True, exist_ok=True)

    front_specs = [s for s in DEPTH_24_SPECS if s["relation"] == "in_front_of"]
    behind_specs = [s for s in DEPTH_24_SPECS if s["relation"] == "behind"]

    rng = random.Random(4294967291)

    front_0 = [(s, seed, 0.00) for s in front_specs for seed in SEEDS_5]
    front_6 = [(s, seed, 6.00) for s in front_specs for seed in SEEDS_5]
    behind_0 = [(s, seed, 0.00) for s in behind_specs for seed in SEEDS_5]
    behind_6 = [(s, seed, 6.00) for s in behind_specs for seed in SEEDS_5]

    sample_f0 = rng.sample(front_0, 30)
    sample_f6 = rng.sample(front_6, 30)
    sample_b0 = rng.sample(behind_0, 30)
    sample_b6 = rng.sample(behind_6, 30)

    all_samples = sample_f0 + sample_f6 + sample_b0 + sample_b6
    rng.shuffle(all_samples)

    hidden_mapping = []
    tool_items = []

    agree_count = 0
    disagree_count = 0

    for idx, (spec, seed, strength) in enumerate(all_samples, start=1):
        anon_id = f"img_{idx:03d}"
        anon_filename = f"{anon_id}.png"
        src_filename = f"{spec['id']}_s{seed}_str_{strength:.2f}.png"

        # Copy image
        shutil.copy2(src_dir / src_filename, img_target_dir / anon_filename)

        # Run automated metrics for hidden mapping only
        img = Image.open(src_dir / src_filename).convert("RGB")
        s_det, o_det = evaluator_2d.detect_entities(img, spec["subject"], spec["object"])
        sat_2d, r_2d = evaluator_2d.check_relation(s_det, o_det, spec["relation"])
        sat_3d, s_d, o_d, r_3d = evaluator_3d.estimate_relative_depth(
            img, s_det, o_det, spec["relation"]
        )

        if sat_2d == sat_3d:
            agree_count += 1
        else:
            disagree_count += 1

        # Hidden mapping entry (stored privately)
        hidden_mapping.append(
            {
                "anonymous_id": anon_id,
                "anonymous_filename": anon_filename,
                "original_filename": src_filename,
                "spec_id": spec["id"],
                "prompt": spec["prompt"],
                "subject": spec["subject"],
                "object": spec["object"],
                "relation": spec["relation"],
                "condition_strength": strength,
                "seed": seed,
                "verdict_2d_predicate": bool(sat_2d),
                "reason_2d": r_2d,
                "verdict_depth_anything_v2": bool(sat_3d),
                "reason_3d": r_3d,
                "depth_subject": float(s_d) if s_d is not None else None,
                "depth_object": float(o_d) if o_d is not None else None,
            }
        )

        # Item for the blind HTML tool (no condition, no seed, no metrics)
        rel_str = spec["relation"].replace("_", " ")
        question = (
            f"Is the '{spec['subject']}' actually {rel_str} the '{spec['object']}' in this image?"
        )
        tool_items.append(
            {
                "id": anon_id,
                "image": f"images/{anon_filename}",
                "subject": spec["subject"],
                "object": spec["object"],
                "relation": rel_str,
                "question": question,
            }
        )

    # Save hidden mapping
    mapping_path = target_dir / "hidden_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(hidden_mapping, f, indent=2)

    # Build Self-Contained HTML Labelling Tool
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Blind Depth Validation Tool (120 Images)</title>
  <style>
    :root {{
      --bg: #121214;
      --card-bg: #1e1e24;
      --accent: #ffd700;
      --text: #e6edf3;
      --muted: #8b949e;
      --border: #30363d;
      --yes: #238636;
      --yes-hover: #2ea043;
      --no: #da3633;
      --no-hover: #f85149;
      --cant: #8b949e;
      --cant-hover: #6e7681;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
    }}
    .criterion-box {{
      background: #161b22;
      border: 1px solid #388bfd;
      border-radius: 8px;
      padding: 12px 20px;
      max-width: 780px;
      width: 100%;
      margin-bottom: 16px;
      font-size: 14px;
      line-height: 1.5;
    }}
    .criterion-title {{
      font-weight: bold;
      color: #58a6ff;
      margin-bottom: 4px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      max-width: 780px;
      width: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      box-shadow: 0 8px 24px rgba(0,0,0,0.5);
    }}
    .top-bar {{
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
    }}
    .counter {{
      font-size: 16px;
      font-weight: bold;
      color: var(--accent);
    }}
    .status-badge {{
      font-size: 13px;
      padding: 4px 10px;
      border-radius: 12px;
      background: #30363d;
    }}
    .status-badge.answered {{
      background: #1f6feb;
      color: #fff;
    }}
    .question-box {{
      text-align: center;
      margin-bottom: 16px;
      min-height: 48px;
    }}
    .question-text {{
      font-size: 18px;
      font-weight: 600;
      color: #fff;
    }}
    .img-container {{
      width: 512px;
      height: 512px;
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 8px;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 20px;
      overflow: hidden;
    }}
    .img-container img {{
      width: 100%;
      height: 100%;
      object-fit: contain;
    }}
    .btn-group {{
      display: flex;
      gap: 12px;
      width: 100%;
      margin-bottom: 20px;
    }}
    .btn {{
      flex: 1;
      padding: 14px 16px;
      border: 2px solid transparent;
      border-radius: 8px;
      font-size: 16px;
      font-weight: bold;
      cursor: pointer;
      color: #fff;
      transition: all 0.15s ease;
    }}
    .btn-yes {{ background: var(--yes); }}
    .btn-yes:hover {{ background: var(--yes-hover); }}
    .btn-yes.selected {{ border-color: #7ee787; box-shadow: 0 0 12px rgba(126,231,135,0.6); }}
    
    .btn-no {{ background: var(--no); }}
    .btn-no:hover {{ background: var(--no-hover); }}
    .btn-no.selected {{ border-color: #ffa198; box-shadow: 0 0 12px rgba(255,161,152,0.6); }}
    
    .btn-cant {{ background: #30363d; color: #c9d1d9; }}
    .btn-cant:hover {{ background: var(--cant-hover); }}
    .btn-cant.selected {{ border-color: #f0883e; background: #6e7681; box-shadow: 0 0 12px rgba(240,136,62,0.6); }}

    .nav-bar {{
      display: flex;
      justify-content: space-between;
      width: 100%;
      border-top: 1px solid var(--border);
      padding-top: 16px;
    }}
    .nav-btn {{
      background: #21262d;
      border: 1px solid var(--border);
      color: var(--text);
      padding: 8px 16px;
      border-radius: 6px;
      font-weight: 600;
      cursor: pointer;
    }}
    .nav-btn:hover {{ background: #30363d; }}
    .nav-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
    .export-btn {{
      background: #1f6feb;
      border: none;
      color: #fff;
      padding: 8px 20px;
      border-radius: 6px;
      font-weight: bold;
      cursor: pointer;
    }}
    .export-btn:hover {{ background: #388bfd; }}
    .keyboard-hint {{
      font-size: 12px;
      color: var(--muted);
      margin-top: 12px;
      text-align: center;
    }}
  </style>
</head>
<body>

  <div class="criterion-box">
    <div class="criterion-title">Standing Ground-Truth Criterion:</div>
    <div>1. <b>If objects overlap:</b> Whichever object occludes / overlaps in front of the other is <b>in front</b>.</div>
    <div>2. <b>If objects do not overlap:</b> Whichever object has the lower ground contact point (closer to camera in ground-plane perspective) is <b>in front</b>.</div>
    <div>3. <b>Can't tell:</b> Choose this if either entity is completely missing, unidentifiable, or ambiguous.</div>
  </div>

  <div class="card">
    <div class="top-bar">
      <div class="counter" id="counter-text">Image 1 / 120</div>
      <div class="status-badge" id="status-badge">Unanswered</div>
    </div>

    <div class="question-box">
      <div class="question-text" id="question-text">Loading question...</div>
    </div>

    <div class="img-container">
      <img id="main-image" src="" alt="Blind Sample">
    </div>

    <div class="btn-group">
      <button class="btn btn-yes" id="btn-yes" onclick="selectAnswer('yes')">Yes (1 / Y)</button>
      <button class="btn btn-no" id="btn-no" onclick="selectAnswer('no')">No (2 / N)</button>
      <button class="btn btn-cant" id="btn-cant" onclick="selectAnswer('cant_tell')">Can't tell (3 / C)</button>
    </div>

    <div class="nav-bar">
      <button class="nav-btn" id="btn-prev" onclick="navigate(-1)">&larr; Back</button>
      <button class="export-btn" onclick="exportCSV()">Download human_labels.csv</button>
      <button class="nav-btn" id="btn-next" onclick="navigate(1)">Next &rarr;</button>
    </div>

    <div class="keyboard-hint">
      Shortcuts: [1] or [Y] = Yes | [2] or [N] = No | [3] or [C] = Can't Tell | [&larr;] = Back | [&rarr;] = Next
    </div>
  </div>

  <script>
    const items = {json.dumps(tool_items, indent=2)};
    let currentIndex = 0;
    let answers = {{}};

    // Load localStorage answers if available
    try {{
      const saved = localStorage.getItem("blind_depth_answers_v1");
      if (saved) {{
        answers = JSON.parse(saved);
      }}
    }} catch(e) {{}}

    function renderCurrent() {{
      const item = items[currentIndex];
      document.getElementById("counter-text").innerText = `Image ${{currentIndex + 1}} / ${{items.length}} (${{item.id}})`;
      document.getElementById("question-text").innerText = item.question;
      document.getElementById("main-image").src = item.image;

      const ans = answers[item.id];
      const badge = document.getElementById("status-badge");
      if (ans) {{
        badge.innerText = `Recorded: ${{ans.toUpperCase()}}`;
        badge.className = "status-badge answered";
      }} else {{
        badge.innerText = "Unanswered";
        badge.className = "status-badge";
      }}

      // Reset button styles
      document.getElementById("btn-yes").className = "btn btn-yes" + (ans === "yes" ? " selected" : "");
      document.getElementById("btn-no").className = "btn btn-no" + (ans === "no" ? " selected" : "");
      document.getElementById("btn-cant").className = "btn btn-cant" + (ans === "cant_tell" ? " selected" : "");

      document.getElementById("btn-prev").disabled = (currentIndex === 0);
      document.getElementById("btn-next").disabled = (currentIndex === items.length - 1);
    }}

    function selectAnswer(val) {{
      const item = items[currentIndex];
      answers[item.id] = val;
      try {{
        localStorage.setItem("blind_depth_answers_v1", JSON.stringify(answers));
      }} catch(e) {{}}
      renderCurrent();
      // Auto-advance after selection if not on last image
      if (currentIndex < items.length - 1) {{
        setTimeout(() => {{
          navigate(1);
        }}, 180);
      }}
    }}

    function navigate(delta) {{
      const nextIdx = currentIndex + delta;
      if (nextIdx >= 0 && nextIdx < items.length) {{
        currentIndex = nextIdx;
        renderCurrent();
      }}
    }}

    function exportCSV() {{
      let csv = "image_id,label\\n";
      items.forEach(it => {{
        const a = answers[it.id] || "unanswered";
        csv += `${{it.id}},${{a}}\\n`;
      }});

      const blob = new Blob([csv], {{ type: "text/csv;charset=utf-8;" }});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "human_depth_labels.csv";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }}

    // Keyboard navigation
    window.addEventListener("keydown", (e) => {{
      const key = e.key.toLowerCase();
      if (key === "1" || key === "y") {{
        selectAnswer("yes");
      }} else if (key === "2" || key === "n") {{
        selectAnswer("no");
      }} else if (key === "3" || key === "c") {{
        selectAnswer("cant_tell");
      }} else if (e.key === "ArrowLeft") {{
        navigate(-1);
      }} else if (e.key === "ArrowRight") {{
        navigate(1);
      }}
    }});

    renderCurrent();
  </script>
</body>
</html>
"""

    html_tool_path = target_dir / "label_depth_images.html"
    with open(html_tool_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("\n" + "=" * 80)
    print("BLIND DEPTH EVALUATION TOOL READY")
    print("=" * 80)
    print("Sample Composition:")
    print("  Total Blinded Images:         120")
    print("  Condition OFF (0.00):          60 (30 in_front_of, 30 behind)")
    print("  Condition ON (6.00):           60 (30 in_front_of, 30 behind)")
    print(f"  Cases where 2D & 3D Agree:    {agree_count} ({agree_count/120*100:.1f}%)")
    print(
        f"  Cases where 2D & 3D Disagree: {disagree_count} ({disagree_count/120*100:.1f}%)"
    )
    print(f"\nTool File:    {html_tool_path.resolve()}")
    print(f"Mapping File: {mapping_path.resolve()} (Saved privately; contents not revealed)")


if __name__ == "__main__":
    main()
