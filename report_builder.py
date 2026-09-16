"""
report_builder.py — Rich multi-section HTML report for the Ensemble Seismograph.

Sections:
  1. Chapter Overview  — grouped bar chart of mean Z-scores per chapter
  2. Seismograph       — dual Z-score lines; click any point to read verse
  3. Fracture Table    — sortable list of top verses by CFI / Shared Seam
"""

import json
import os
import re

import pandas as pd

# Version of plotly.js the template was written against.
PLOTLY_CDN_VERSION = "2.26.0"
_PLOTLY_CDN_TAG = (
    f'<script src="https://cdn.plot.ly/plotly-{PLOTLY_CDN_VERSION}.min.js"></script>'
)


def _plotly_script(offline: bool = False, plotly_js_path: str | None = None) -> str:
    """Return the <script> element that provides plotly.js.

    offline=False (default) links the CDN build: small file, needs a network
    connection to open.

    offline=True embeds the library in the page, so the report opens from a USB
    stick or an air-gapped machine. The source is taken from *plotly_js_path* if
    given, otherwise from the installed plotly package (plotly.offline.get_plotlyjs).
    Adds roughly 3.6 MB to the report. Falls back to the CDN, with a warning, if
    the library cannot be located.
    """
    if not offline:
        return _PLOTLY_CDN_TAG

    src = None
    if plotly_js_path:
        if not os.path.isfile(plotly_js_path):
            print(f"[report_builder] plotly_js_path not found: {plotly_js_path}")
        else:
            src = open(plotly_js_path, encoding="utf-8").read()
            origin = plotly_js_path
    if src is None:
        try:
            from plotly.offline import get_plotlyjs
            src = get_plotlyjs()
            origin = "installed plotly package"
        except Exception as exc:                      # pragma: no cover
            print(f"[report_builder] could not embed plotly.js ({exc}); using the CDN instead.")
            return _PLOTLY_CDN_TAG

    # A literal </script> anywhere in the source would close the tag early.
    src = src.replace("</script", r"<\/script")

    found = re.search(r"plotly\.js v(\d+)\.(\d+)\.(\d+)", src[:400])
    version = ".".join(found.groups()) if found else "unknown"
    if found and found.group(1) != PLOTLY_CDN_VERSION.split(".")[0]:
        print(f"[report_builder] WARNING: embedding plotly.js {version}, but this "
              f"template was written for {PLOTLY_CDN_VERSION}. Check the charts render.")
    print(f"[report_builder] Embedding plotly.js {version} from {origin} "
          f"({len(src) / 1e6:.1f} MB).")

    return ("<!-- plotly.js bundled for offline use — "
            "Copyright 2012-2023 Plotly, Inc., MIT licence -->\n"
            "<script>\n" + src + "\n</script>")


def build_report_html(df: pd.DataFrame, book_name: str, z_threshold: float = 2.0,
                      pearson_r: float = 0.0, spearman_rho: float = 0.0,
                      offline: bool = False, plotly_js_path: str | None = None) -> str:
    """Return a self-contained HTML string."""

    # ── Chapter aggregates ──────────────────────────────────────────────
    df2 = df.copy()
    df2["chapter"] = df2["verse_id"].apply(lambda x: int(x.split(".")[1]))
    ch = (
        df2.groupby("chapter")
        .agg(
            mean_gpt=("global_z_gpt", "mean"),
            mean_dicta=("global_z_dicta", "mean"),
            n_seams=("Strict_Shared_Seam", "sum"),
        )
        .reset_index()
    )

    # ── Summary stats ───────────────────────────────────────────────────
    n_v = len(df)
    n_gpt    = int((df["global_z_gpt"]   >= z_threshold).sum())
    n_dicta  = int((df["global_z_dicta"] >= z_threshold).sum())
    n_shared = int(df["Strict_Shared_Seam"].sum())

    # ── Seams / fracture table rows ─────────────────────────────────────
    seams = df[df["Strict_Shared_Seam"]].sort_values("CFI_mag", ascending=False)
    if seams.empty:
        seams = df.nlargest(30, "CFI_mag")
        note = (
            f"No strict shared seams at Z≥{z_threshold}. "
            "Showing top 30 verses by Combined Fracture Index."
        )
    else:
        note = (
            f"{n_shared} verse(s) flagged by both models under FDR control, "
            "sorted by CFI."
        )

    rows_html = ""
    for _, r in seams.iterrows():
        badge = "<span class='badge'>Shared</span>" if r["Strict_Shared_Seam"] else ""
        gc = " hi" if r["global_z_gpt"]   >= z_threshold else ""
        dc = " hi" if r["global_z_dicta"] >= z_threshold else ""
        rows_html += (
            f"<tr class=\"{'sr' if r['Strict_Shared_Seam'] else ''}\">"
            f"<td>{r['verse_id']}</td>"
            f"<td class='heb'>{r['text']}</td>"
            f"<td class='n{gc}'>{r['global_z_gpt']:.3f}</td>"
            f"<td class='n{dc}'>{r['global_z_dicta']:.3f}</td>"
            f"<td class='n cf'>{r['CFI_mag']:.3f}</td>"
            f"<td class='n'>{r.get('cfi_percentile', 0.0):.1f}</td>"
            f"<td>{badge}</td></tr>\n"
        )

    # ── Composite / CFI seams table rows ─────────────────────────────────
    cfi_seams = df[df["Significant_CFI_Seam"] & ~df["Strict_Shared_Seam"]].sort_values("CFI_mag", ascending=False)
    cfi_rows_html = ""
    for _, r in cfi_seams.iterrows():
        gc = " hi" if r["global_z_gpt"]   >= z_threshold else ""
        dc = " hi" if r["global_z_dicta"] >= z_threshold else ""
        cfi_rows_html += (
            f"<tr>"
            f"<td>{r['verse_id']}</td>"
            f"<td class='heb'>{r['text']}</td>"
            f"<td class='n{gc}'>{r['global_z_gpt']:.3f}</td>"
            f"<td class='n{dc}'>{r['global_z_dicta']:.3f}</td>"
            f"<td class='n cf'>{r['CFI_mag']:.3f}</td>"
            f"<td class='n'>{r.get('cfi_percentile', 0.0):.1f}</td>"
            f"<td><span class='badge' style='background:rgba(211,199,180,0.18);color:#d3c7b4;'>Composite</span></td></tr>\n"
        )

    # ── Chapter X-axis ticks ─────────────────────────────────────────────
    ch_starts: dict[int, int] = {}
    for i, vid in enumerate(df["verse_id"]):
        c = int(vid.split(".")[1])
        if c not in ch_starts:
            ch_starts[c] = i
    tv_js  = json.dumps(list(ch_starts.values()))
    tt_js  = json.dumps([str(k) for k in ch_starts.keys()])

    # ── JS data payloads ────────────────────────────────────────────────
    vcols = ["verse_id", "text", "global_z_gpt", "global_z_dicta", "CFI_mag", "Strict_Shared_Seam", "Significant_CFI_Seam", "Is_Regime_Change", "Regime_ID"]
    verses_js  = df[vcols].to_json(orient="records", force_ascii=False)
    chapter_js = ch.to_json(orient="records", force_ascii=False)

    # ── Render ──────────────────────────────────────────────────────────
    return _HTML.format(
        book=book_name,
        z=z_threshold,
        n_v=f"{n_v:,}",
        n_gpt=n_gpt,
        n_dicta=n_dicta,
        n_shared=n_shared,
        n_cfi=len(cfi_seams),
        note=note,
        rows=rows_html,
        cfi_rows=cfi_rows_html,
        verses_js=verses_js,
        chapter_js=chapter_js,
        tv=tv_js,
        tt=tt_js,
        n_minus1=n_v - 1,
        pearson_r=pearson_r,
        spearman_rho=spearman_rho,
        plotly_script=_plotly_script(offline, plotly_js_path),
    )


# ════════════════════════════════════════════════════════════════════════════
# HTML template — uses .format() so JS {{ }} work as literal braces
# ════════════════════════════════════════════════════════════════════════════
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Seismograph — {book}</title>
{plotly_script}
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Crimson+Text:ital@0;1&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
body{{background:#17100f;color:#e8dedd;font-family:Inter,sans-serif;font-size:14px;line-height:1.6}}
/* header */
.hdr{{background:linear-gradient(135deg,#331a1c,#3d2123);border-bottom:1px solid #46292b;padding:22px 40px 14px}}
.hdr h1{{font-family:'Crimson Text',Georgia,serif;font-size:1.9rem;font-weight:400;color:#f7eeed}}
.hdr small{{color:#555;font-size:11px;display:block;margin-top:3px}}
/* stats bar */
.sb{{display:flex;gap:1px;background:#2b1a1b;border-bottom:1px solid #46292b}}
.st{{flex:1;padding:16px 16px;background:#17100f;text-align:center}}
.st b{{display:block;font-size:2rem;font-weight:700;line-height:1;color:#ff7a6e}}
.st b.B{{color:#dfa13c}} .st b.G{{color:#d3c7b4}} .st b.W{{color:#f7eeed}}
.st span{{font-size:10px;color:#9c8081;letter-spacing:0;margin-top:5px;display:block}}
/* tabs */
.tabs{{display:flex;padding:0 40px;border-bottom:1px solid #46292b;background:#17100f}}
.tab{{padding:12px 22px;background:none;border:none;border-bottom:3px solid transparent;color:#8c7071;font-family:Inter,sans-serif;font-size:12px;font-weight:600;cursor:pointer;letter-spacing:0;transition:.2s}}
.tab:hover{{color:#f2e9e8}} .tab.on{{color:#f2e9e8;border-bottom-color:#ff7a6e}}
/* sections */
.sec{{display:none;padding:28px 40px;min-height:80vh}} .sec.on{{display:block}}
.stitle{{font-family:'Crimson Text',serif;font-size:1.35rem;color:#999;margin-bottom:18px}}
/* chart wrapper */
.cw{{background:#211415;border-radius:0;padding:4px;margin-bottom:20px}}
/* verse panel */
.vp{{background:#211415;border:1px solid #46292b;border-radius:0;padding:20px 24px;min-height:120px;transition:.3s}}
.vp .hint{{color:#6e5152;font-style:italic;text-align:center;padding:28px 0;font-size:13px}}
.vp .vid{{font-size:10px;color:#8c7071;letter-spacing:0;font-weight:600;margin-bottom:9px}}
.vp .vt{{font-family:'SBL Hebrew','David','Noto Sans Hebrew',Arial,serif;font-size:1.48rem;line-height:1.9;direction:rtl;text-align:right;color:#f7eeed;margin-bottom:13px;border-right:3px solid #46292b;padding-right:14px}}
.vp.seam .vt{{border-right-color:#ff7a6e}}
.scores{{display:flex;gap:10px;flex-wrap:wrap}}
.pill{{padding:4px 11px;border-radius:0;font-size:11px;font-weight:600;font-family:monospace}}
.pg{{background:rgba(255,122,110,.1);color:#ff7a6e;border:1px solid rgba(255,122,110,.25)}}
.pd{{background:rgba(223,161,60,.1);color:#dfa13c;border:1px solid rgba(223,161,60,.25)}}
.pc{{background:rgba(211,199,180,.1);color:#d3c7b4;border:1px solid rgba(211,199,180,.25)}}
.ps{{background:rgba(255,122,110,.18);color:#ff9e94;border:1px solid rgba(210,80,70,.4)}}
/* table */
.note{{color:#9c8081;font-size:11px;margin-bottom:14px;font-style:italic}}
table{{width:100%;border-collapse:collapse}}
thead{{position:sticky;top:0;background:#17100f;z-index:5}}
th{{padding:9px 14px;text-align:left;font-size:11.5px;font-weight:600;letter-spacing:0;color:#9c8081;border-bottom:1px solid #46292b}}
td{{padding:8px 14px;border-bottom:1px solid #3d2123;vertical-align:middle}}
tr:hover td{{background:#3d2123}} .sr td{{background:rgba(255,122,110,.04)}}
.heb{{font-family:'SBL Hebrew','David','Noto Sans Hebrew',serif;font-size:1.05rem;direction:rtl;text-align:right;max-width:360px}}
.n{{font-family:monospace;font-size:12px;text-align:right}}
.n.hi{{color:#ff7a6e;font-weight:700}} .cf{{color:#d3c7b4}}
.badge{{background:rgba(255,122,110,.18);color:#ff7a6e;padding:3px 8px;border-radius:0;font-size:10.5px;font-weight:700;letter-spacing:0}}
/* disclaimer */
.disc{{margin-top:32px;padding:14px 16px;background:#120b0b;border-left:3px solid #46292b;border-radius:0;font-size:11px;color:#6e5152;font-style:italic}}
</style>
</head>
<body>

<div class="hdr">
  <h1>&#128300; Ensemble Fracture Seismograph &#8212; {book}</h1>
  <small>GPT-Neo (causal LM) &middot; DictaBERT (masked LM PLL) &middot; Westminster Leningrad Codex</small>
</div>

<div class="sb">
  <div class="st"><b class="W">{n_v}</b><span>Verses</span></div>
  <div class="st"><b>{n_gpt}</b><span>GPT-Neo peaks</span></div>
  <div class="st"><b class="B">{n_dicta}</b><span>DictaBERT peaks</span></div>
  <div class="st"><b class="G">{n_shared}</b><span>Shared Seams</span></div>
  <div class="st"><b class="W">{pearson_r:.2f}</b><span>Pearson r</span></div>
  <div class="st"><b class="W">{spearman_rho:.2f}</b><span>Spearman &rho;</span></div>
  <div class="st" style="flex:2; padding-top:16px;">
    <label style="font-size:11px;color:#ccc;cursor:pointer;">
      <input type="checkbox" id="mask-toggle" onclick="toggleMask(this)"> Show only non-consensus verses
    </label>
  </div>
</div>

<div class="tabs">
  <button class="tab on" onclick="show('overview',this)">&#128202; Chapter Overview</button>
  <button class="tab"    onclick="show('seismo',this)">&#128200;&#65039; Seismograph</button>
  <button class="tab"    onclick="show('table',this)">&#128279; Fracture Table</button>
</div>

<!-- SECTION 1: Overview -->
<div class="sec on" id="sec-overview">
  <div class="stitle">Chapter-Level Fracture Profile</div>
  <div class="cw" id="ch-chart"></div>
  <div class="disc">&#9888; Results reflect stylistic signal as measured by modern Hebrew language models.
  They do not constitute literary or historical dating. Shared Seams are the most empirically
  robust findings &#8212; both models independently detected anomalous signal at the same verse.</div>
</div>

<!-- SECTION 2: Seismograph -->
<div class="sec" id="sec-seismo">
  <div class="stitle">Dual-Model Verse Seismograph
    <span style="font-size:11px;color:#8c7071;font-family:Inter,sans-serif;"> &#8212; click any point to inspect verse</span>
  </div>
  <div class="cw" id="seismo-chart"></div>
  <div class="vp" id="vp">
    <div class="hint">Click any point on the seismograph to read the Hebrew verse and its scores.</div>
  </div>
</div>

<!-- SECTION 3: Table -->
<div class="sec" id="sec-table">
  <div class="stitle">Fracture Table: Strict Shared Seams</div>
  <p class="note">{note}</p>
  <div style="overflow-x:auto; margin-bottom: 40px;">
    <table>
      <thead>
        <tr>
          <th>Verse</th><th style="text-align:right">Hebrew Text</th>
          <th>Z GPT</th><th>Z Dicta</th><th>CFI</th><th>%Rank</th><th></th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </div>

  <div class="stitle">Fracture Table: Composite Seams ({n_cfi})</div>
  <p class="note">Verses that did not independently clear FDR on both models, but achieved global FDR significance through their Combined Fracture Index (CFI) vector magnitude.</p>
  <div style="overflow-x:auto">
    <table>
      <thead>
        <tr>
          <th>Verse</th><th style="text-align:right">Hebrew Text</th>
          <th>Z GPT</th><th>Z Dicta</th><th>CFI</th><th>%Rank</th><th></th>
        </tr>
      </thead>
      <tbody>{cfi_rows}</tbody>
    </table>
  </div>
</div>

<script>
const VERSES = {verses_js};
const CHAPTERS = {chapter_js};
const ZT = {z};
window._isMasked = false;

function toggleMask(chk) {{
  window._isMasked = chk.checked;
  const maskRanges = [[1,5], [24,27], [36,39], [40,55], [56,66]];
  
  function isConsensus(vid) {{
    if (!vid) return false;
    const ch = parseInt(vid.split('.')[1]);
    for (let r of maskRanges) {{
      if (ch >= r[0] && ch <= r[1]) return true;
    }}
    return false;
  }}
  
  document.querySelectorAll('tbody tr').forEach(tr => {{
    const vid = tr.cells[0].innerText;
    if (isConsensus(vid)) {{
      tr.style.display = window._isMasked ? 'none' : '';
    }} else {{
      tr.style.display = '';
    }}
  }});
  
  if(document.getElementById('sec-seismo').classList.contains('on')) renderSeismo();
}}

// ─── Tab switching ────────────────────────────────────────────────────────
function show(name, btn) {{
  document.querySelectorAll('.sec').forEach(s=>s.classList.remove('on'));
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('on'));
  document.getElementById('sec-'+name).classList.add('on');
  btn.classList.add('on');
  if (name==='seismo' && !window._sp) renderSeismo();
}}

// ─── Overview chart ───────────────────────────────────────────────────────
(function() {{
  const chs = CHAPTERS.map(r=>r.chapter);
  const gv  = CHAPTERS.map(r=>+r.mean_gpt.toFixed(3));
  const dv  = CHAPTERS.map(r=>+r.mean_dicta.toFixed(3));
  const lay = {{
    paper_bgcolor:'#211415', plot_bgcolor:'#211415',
    font:{{family:'Inter,sans-serif',color:'#888',size:11}},
    margin:{{l:50,r:20,t:40,b:80}}, height:380, barmode:'group', bargap:0.2,
    xaxis:{{
      title:'Chapter',color:'#8c7071',gridcolor:'#2b1a1b',tickfont:{{size:10}},
      tickmode:'linear', dtick:1,
      rangeslider:{{visible:true, bgcolor:'#120b0b', thickness:0.06}},
      range:[0.5, Math.min(20, chs.length)+0.5],
    }},
    yaxis:{{title:'Mean Z-Score',color:'#8c7071',gridcolor:'#2b1a1b',zeroline:true,zerolinecolor:'#55393a'}},
    legend:{{bgcolor:'rgba(0,0,0,0)',x:1,xanchor:'right',y:1}},
  }};
  Plotly.newPlot('ch-chart', [
    {{type:'bar',x:chs,y:gv,name:'GPT-Neo',
      marker:{{color:'rgba(255,122,110,0.55)',line:{{width:0}}}}}},
    {{type:'bar',x:chs,y:dv,name:'DictaBERT',
      marker:{{color:'rgba(223,161,60,0.55)',line:{{width:0}}}}}},
  ], lay, {{responsive:true,displayModeBar:true,
    modeBarButtonsToKeep:['zoom2d','pan2d','resetScale2d']}});
}})();

// ─── Seismograph chart ────────────────────────────────────────────────────
window._sp = false;
function renderSeismo() {{
  const isMasked = window._isMasked;
  const maskRanges = [[1,5], [24,27], [36,39], [40,55], [56,66]];
  function isConsensus(vid) {{
    if (!vid) return false;
    const ch = parseInt(vid.split('.')[1]);
    for (let r of maskRanges) {{
      if (ch >= r[0] && ch <= r[1]) return true;
    }}
    return false;
  }}

  const zg = VERSES.map(v => (isMasked && isConsensus(v.verse_id)) ? null : v.global_z_gpt);
  const zd = VERSES.map(v => (isMasked && isConsensus(v.verse_id)) ? null : v.global_z_dicta);
  const ids= VERSES.map(v=>v.verse_id);
  const shX= VERSES.map((v,i)=>v.Strict_Shared_Seam && !(isMasked && isConsensus(v.verse_id)) ?i:null).filter(i=>i!==null);
  const shY= shX.map(i=>VERSES[i].global_z_gpt);
  
  const cfiX = VERSES.map((v,i)=>v.Significant_CFI_Seam && !v.Strict_Shared_Seam && !(isMasked && isConsensus(v.verse_id)) ? i : null).filter(i=>i!==null);
  const cfiY = cfiX.map(i=>VERSES[i].global_z_gpt);
  
  // Extract PELT boundaries
  const peltX = VERSES.map((v,i)=>v.Is_Regime_Change && !(isMasked && isConsensus(v.verse_id)) ? i : null).filter(i=>i!==null);

  const lay = {{
    paper_bgcolor:'#211415', plot_bgcolor:'#211415',
    font:{{family:'Inter,sans-serif',color:'#888',size:11}},
    margin:{{l:50,r:20,t:20,b:80}}, height:380, hovermode:'closest',
    dragmode: 'pan',
    xaxis:{{
      title:'Chapter', color:'#8c7071', gridcolor:'#2b1a1b',
      tickvals:{tv}, ticktext:{tt}, tickangle:-50, tickfont:{{size:9}},
      rangeslider:{{visible:true, bgcolor:'#120b0b', thickness:0.05}},
      range:[0, Math.min(150, {n_minus1})],
    }},
    yaxis:{{
      title:'Z-Score', color:'#8c7071', gridcolor:'#2b1a1b',
      zeroline:true, zerolinecolor:'#55393a',
      fixedrange: true,
    }},
    shapes:[
      {{type:'line',x0:0,x1:{n_minus1},y0:ZT,y1:ZT,
        line:{{color:'rgba(244,162,97,0.35)',width:1,dash:'dash'}}}},
      {{type:'line',x0:0,x1:{n_minus1},y0:-ZT,y1:-ZT,
        line:{{color:'rgba(244,162,97,0.15)',width:1,dash:'dash'}}}},
      ...peltX.map(x => ({{
        type: 'line', x0: x, x1: x, y0: 0, y1: 1, yref: 'paper',
        line: {{color: 'rgba(255, 255, 255, 0.15)', width: 2, dash: 'dot'}}
      }}))
    ],
    legend:{{bgcolor:'rgba(0,0,0,0)',x:1,xanchor:'right',y:1}},
  }};

  const traces = [
    {{type:'scatter',mode:'lines',y:zg,x:[...Array(zg.length).keys()],
      name:'Z (GPT-Neo)',line:{{color:'#ff7a6e',width:1.5}},
      customdata:ids, hovertemplate:'<b>%{{customdata}}</b><br>Z-GPT: %{{y:.3f}}<extra></extra>'}},
    {{type:'scatter',mode:'lines',y:zd,x:[...Array(zd.length).keys()],
      name:'Z (DictaBERT)',line:{{color:'#dfa13c',width:1.5}},
      customdata:ids, hovertemplate:'<b>%{{customdata}}</b><br>Z-Dicta: %{{y:.3f}}<extra></extra>'}},
  ];
  if(cfiX.length) traces.push({{
    type:'scatter',mode:'markers',x:cfiX,y:cfiY,name:'Composite Seam',
    marker:{{color:'#d3c7b4',size:7,symbol:'circle',line:{{width:1,color:'#111'}}}},
    customdata:cfiX.map(i=>VERSES[i].verse_id),
    hovertemplate:'<b>Composite seam</b><br>%{{customdata}}<extra></extra>',
  }});
  if(shX.length) traces.push({{
    type:'scatter',mode:'markers',x:shX,y:shY,name:'Strict Shared Seam',
    marker:{{color:'#d62828',size:11,symbol:'diamond',line:{{width:1.5,color:'white'}}}},
    customdata:shX.map(i=>VERSES[i].verse_id),
    hovertemplate:'<b>STRICT SHARED SEAM</b><br>%{{customdata}}<extra></extra>',
  }});

  const el = document.getElementById('seismo-chart');
  Plotly.newPlot(el, traces, lay, {{responsive:true,displayModeBar:true,
    modeBarButtonsToKeep:['zoom2d','pan2d','resetScale2d']}});

  el.on('plotly_click', function(data) {{
    const i = data.points[0].pointIndex;
    const v = VERSES[i];
    if(!v) return;
    const vp = document.getElementById('vp');
    vp.className = 'vp' + (v.Strict_Shared_Seam?' seam':'');
    vp.innerHTML =
      '<div class="vid">' + v.verse_id + ' &middot; REGIME ' + v.Regime_ID + '</div>' +
      '<div class="vt">' + v.text + '</div>' +
      '<div class="scores">' +
        '<span class="pill pg">GPT Z: ' + v.global_z_gpt.toFixed(3) + '</span>' +
        '<span class="pill pd">Dicta Z: ' + v.global_z_dicta.toFixed(3) + '</span>' +
        '<span class="pill pc">CFI: ' + v.CFI_mag.toFixed(3) + '</span>' +
        (v.Strict_Shared_Seam ? '<span class="pill ps">&#9889; STRICT SHARED SEAM</span>' : '') +
        (v.Significant_CFI_Seam && !v.Strict_Shared_Seam ? '<span class="pill" style="background:rgba(211,199,180,0.18);color:#d3c7b4;border:1px solid rgba(211,199,180,0.4)">&#9889; Composite seam</span>' : '') +
        (v.Is_Regime_Change ? '<span style="color:#fff;font-size:11px;margin-left:8px;align-self:center;">&#9873; STARTS NEW REGIME</span>' : '') +
      '</div>';
  }});
  window._sp = true;
}}
</script>
</body>
</html>
"""
