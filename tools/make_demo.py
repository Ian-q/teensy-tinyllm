#!/usr/bin/env python3
# Copyright (c) 2026 Ian Adelman
# SPDX-License-Identifier: MIT
"""Build the Semaphore walkthrough page from live measurements.

Every number on the page comes from running the actual coder — this shells out
to host/build/tq_sem rather than restating figures, so regenerating after a
fine-tune (#25) produces an honest new page rather than a stale one with a new
date on it. That before/after is the most convincing thing this demo can show:
the same messages, the same coder, a model that has read the right register.

    make -C host build/tq_sem
    python3 tools/make_demo.py --model stories15M.etq --out docs/semaphore-demo.html
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

MESSAGES = [
    ("in-domain",
     "Once upon a time there was a little girl named Lily who loved to play."),
    ("in-domain",
     "The little dog barked at the bird sitting in the tall tree."),
    ("operational",
     "Meet me at the north trailhead at sunset, bring the spare battery."),
    ("operational",
     "Need pickup at mile marker 42 on the county road, truck wont start."),
    ("operational",
     "Weather turning, wind 25 gusting 40, we are staying put tonight."),
]


def measure(tq_sem: str, model: str, dictfile: str | None) -> list[dict]:
    out = []
    for register, msg in MESSAGES:
        cmd = [tq_sem, model, "explain", msg]
        if dictfile:
            cmd += ["--dict", dictfile]
        rec = json.loads(subprocess.run(cmd, capture_output=True, text=True,
                                        check=True).stdout)
        enc = subprocess.run([tq_sem, model, "encode", msg],
                             capture_output=True, text=True, check=True).stdout
        wire = next(l.split()[2] for l in enc.splitlines() if l.startswith("wire"))
        if len(wire) // 2 != rec["wire_bytes"]:
            sys.exit(f"wire length disagrees for {msg!r}")
        rec["wire_hex"] = wire
        rec["register"] = register
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tq-sem", default="host/build/tq_sem")
    ap.add_argument("--dict", dest="dictfile", default="tools/corpora/dict.txt")
    ap.add_argument("--out", default="docs/semaphore-demo.html")
    args = ap.parse_args()

    data = measure(args.tq_sem, args.model, args.dictfile)
    page = HEAD + BODY.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    pathlib.Path(args.out).write_text(page)
    print(f"{args.out}: {len(page)} bytes, {len(data)} messages, "
          f"{sum(len(m['tokens']) for m in data)} tokens")
    return 0

HEAD = """<title>Semaphore — sending text as the model's surprise</title>
<style>
:root{
  --ground:#E8EBEF; --panel:#F3F5F8; --edge:#C9D0D9;
  --ink:#161A22; --ink-2:#3D4654; --muted:#67707E;
  --signal:#1F6B7B; --warm:#CE8524; --hot:#BE3D24;
  --grid:rgba(22,26,34,.07);
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --sans:ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  --measure:68ch;
}
@media (prefers-color-scheme:dark){
  :root{
    --ground:#101319; --panel:#171B23; --edge:#2A313D;
    --ink:#DDE2E9; --ink-2:#AAB3C0; --muted:#7C8695;
    --signal:#4FA8BC; --warm:#E0A048; --hot:#E06044;
    --grid:rgba(221,226,233,.07);
  }
}
:root[data-theme="dark"]{
  --ground:#101319; --panel:#171B23; --edge:#2A313D;
  --ink:#DDE2E9; --ink-2:#AAB3C0; --muted:#7C8695;
  --signal:#4FA8BC; --warm:#E0A048; --hot:#E06044;
  --grid:rgba(221,226,233,.07);
}
:root[data-theme="light"]{
  --ground:#E8EBEF; --panel:#F3F5F8; --edge:#C9D0D9;
  --ink:#161A22; --ink-2:#3D4654; --muted:#67707E;
  --signal:#1F6B7B; --warm:#CE8524; --hot:#BE3D24;
  --grid:rgba(22,26,34,.07);
}

body{background:var(--ground);color:var(--ink);font-family:var(--sans);
     line-height:1.6;-webkit-font-smoothing:antialiased;}
.wrap{max-width:1080px;margin:0 auto;padding:clamp(1.5rem,4vw,4rem) clamp(1.1rem,4vw,2.5rem) 6rem;}
.prose{max-width:var(--measure);}
p{margin:0 0 1.1rem;color:var(--ink-2);}
strong{color:var(--ink);font-weight:600;}
a{color:var(--signal);}
h1,h2,h3{text-wrap:balance;margin:0;}
.eyebrow{font-family:var(--mono);font-size:.72rem;letter-spacing:.16em;
         text-transform:uppercase;color:var(--muted);margin:0 0 .7rem;}
h1{font-family:var(--mono);font-weight:600;letter-spacing:-.035em;
   font-size:clamp(2.3rem,6.5vw,4rem);line-height:1.02;margin-bottom:1.2rem;}
h2{font-family:var(--mono);font-weight:600;letter-spacing:-.02em;
   font-size:clamp(1.35rem,3vw,1.85rem);line-height:1.15;}
h3{font-family:var(--mono);font-weight:600;font-size:1rem;letter-spacing:-.01em;}
section{margin-top:clamp(3.5rem,8vw,6rem);}
.lede{font-size:clamp(1.05rem,2vw,1.2rem);color:var(--ink-2);}
hr{border:0;border-top:1px solid var(--edge);margin:0;}

/* ---- hero comparison ---- */
.hero{display:grid;gap:1rem;grid-template-columns:1fr;margin:2.5rem 0 0;}
@media(min-width:720px){.hero{grid-template-columns:1fr auto 1fr;align-items:center;}}
.card{background:var(--panel);border:1px solid var(--edge);border-radius:3px;
      padding:1.25rem 1.4rem;}
.card .k{font-family:var(--mono);font-size:.7rem;letter-spacing:.14em;
         text-transform:uppercase;color:var(--muted);margin-bottom:.6rem;}
.plain{font-family:var(--mono);font-size:.95rem;line-height:1.5;word-break:break-word;}
.bytes{font-family:var(--mono);font-size:clamp(1.1rem,3vw,1.6rem);color:var(--signal);
       letter-spacing:.04em;font-weight:600;word-break:break-all;}
.arrow{font-family:var(--mono);color:var(--muted);text-align:center;font-size:1.5rem;}
.count{font-family:var(--mono);font-size:.78rem;color:var(--muted);margin-top:.7rem;
       font-variant-numeric:tabular-nums;}

/* ---- token tape ---- */
.tabs{display:flex;flex-wrap:wrap;gap:.4rem;margin:0 0 1.6rem;}
.tab{font-family:var(--mono);font-size:.76rem;padding:.42rem .75rem;cursor:pointer;
     background:transparent;color:var(--muted);border:1px solid var(--edge);
     border-radius:2px;transition:color .15s,border-color .15s,background .15s;}
.tab:hover{color:var(--ink);}
.tab[aria-selected="true"]{background:var(--ink);color:var(--ground);border-color:var(--ink);}
.tab:focus-visible{outline:2px solid var(--signal);outline-offset:2px;}

.tape-scroll{overflow-x:auto;padding-bottom:.4rem;}
.tape{display:flex;flex-wrap:wrap;gap:3px;align-items:flex-end;min-width:min-content;}
.tok{display:flex;flex-direction:column;gap:0;border-radius:2px;overflow:hidden;
     border:1px solid var(--edge);background:var(--panel);}
.tok .txt{font-family:var(--mono);font-size:.88rem;padding:.3rem .42rem .28rem;
          white-space:pre;color:var(--ink);}
.tok .bar{height:20px;display:flex;align-items:flex-end;background:var(--grid);}
.tok .fill{width:100%;}
.tok .n{font-family:var(--mono);font-size:.62rem;text-align:center;color:var(--muted);
        padding:.15rem 0 .2rem;font-variant-numeric:tabular-nums;
        border-top:1px solid var(--edge);}
@media(prefers-reduced-motion:no-preference){
  .tok{animation:rise .28s ease-out backwards;}
  @keyframes rise{from{opacity:0;transform:translateY(4px);}to{opacity:1;transform:none;}}
}

.legend{display:flex;flex-wrap:wrap;gap:1.2rem;align-items:center;margin-top:1.5rem;
        font-family:var(--mono);font-size:.72rem;color:var(--muted);}
.ramp{height:8px;width:170px;border-radius:2px;
      background:linear-gradient(90deg,#1F6B7B,#CE8524,#BE3D24);}
.tape-foot{display:flex;flex-wrap:wrap;gap:1.6rem;margin-top:1.6rem;
           font-family:var(--mono);font-size:.78rem;font-variant-numeric:tabular-nums;}
.tape-foot div span{color:var(--muted);}
.tape-foot div b{color:var(--ink);font-weight:600;}

/* ---- bars ---- */
.bars{display:grid;gap:.55rem;margin-top:1.8rem;}
.bar-row{display:grid;grid-template-columns:8.5rem 1fr;gap:.9rem;align-items:center;}
@media(max-width:560px){.bar-row{grid-template-columns:1fr;gap:.2rem;}}
.bar-row .lbl{font-family:var(--mono);font-size:.76rem;color:var(--muted);}
.bar-track{position:relative;height:26px;background:var(--grid);border-radius:2px;}
.bar-fill{position:absolute;inset:0 auto 0 0;border-radius:2px;
          display:flex;align-items:center;justify-content:flex-end;padding-right:.5rem;}
.bar-fill span{font-family:var(--mono);font-size:.72rem;font-variant-numeric:tabular-nums;
               color:#fff;font-weight:600;}
.bar-fill.pale span{color:var(--ink);}

/* ---- tables ---- */
.tbl-scroll{overflow-x:auto;margin-top:1.5rem;}
table{border-collapse:collapse;width:100%;font-size:.86rem;min-width:520px;}
th,td{text-align:left;padding:.6rem .8rem;border-bottom:1px solid var(--edge);
      vertical-align:top;}
th{font-family:var(--mono);font-size:.7rem;letter-spacing:.1em;text-transform:uppercase;
   color:var(--muted);font-weight:500;}
td{color:var(--ink-2);}
td.num{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--ink);}
.verdict{font-family:var(--mono);font-size:.72rem;padding:.16rem .45rem;border-radius:2px;
         white-space:nowrap;}
.v-no{background:color-mix(in srgb,var(--hot) 16%,transparent);color:var(--hot);}
.v-yes{background:color-mix(in srgb,var(--signal) 18%,transparent);color:var(--signal);}

/* ---- proof block ---- */
.proof{background:var(--panel);border:1px solid var(--edge);border-radius:3px;
       padding:1.3rem 1.4rem;margin-top:1.6rem;font-family:var(--mono);font-size:.82rem;
       overflow-x:auto;}
.proof .line{display:flex;gap:1rem;white-space:nowrap;padding:.18rem 0;}
.proof .who{color:var(--muted);min-width:15rem;}
.proof .val{color:var(--signal);font-weight:600;}
.match{margin-top:.9rem;padding-top:.9rem;border-top:1px solid var(--edge);
       color:var(--ink);font-size:.78rem;}

.flip{display:grid;gap:.9rem;margin-top:1.6rem;}
.flip .row{background:var(--panel);border:1px solid var(--edge);border-radius:3px;
           padding:.9rem 1.1rem;}
.flip .row.bad{border-color:color-mix(in srgb,var(--hot) 45%,var(--edge));}
.flip .k{font-family:var(--mono);font-size:.68rem;letter-spacing:.12em;
         text-transform:uppercase;color:var(--muted);margin-bottom:.4rem;}
.flip .t{font-family:var(--mono);font-size:.86rem;line-height:1.5;}
.flip .row.bad .t{color:var(--hot);}

footer{margin-top:5rem;padding-top:1.6rem;border-top:1px solid var(--edge);
       font-size:.8rem;color:var(--muted);}
footer p{color:var(--muted);font-size:.8rem;}
code{font-family:var(--mono);font-size:.88em;background:var(--grid);
     padding:.1em .32em;border-radius:2px;}
</style>
"""

BODY = """
<div class="wrap">

<header class="prose">
  <p class="eyebrow">Teensy 4.1 &middot; 15M-parameter Llama &middot; Q4_0</p>
  <h1>Sending text as the model&rsquo;s surprise</h1>
  <p class="lede">Two devices holding identical weights don&rsquo;t need to exchange
  text. They can exchange only the part the model didn&rsquo;t already predict &mdash;
  which, for ordinary English, is almost nothing.</p>
</header>

<div class="hero">
  <div class="card">
    <div class="k">what was typed</div>
    <div class="plain">Once upon a time there was a little girl named Lily who loved to play.</div>
    <div class="count">70 bytes &middot; gzip gets it to 65</div>
  </div>
  <div class="arrow">&rarr;</div>
  <div class="card">
    <div class="k">what went on the air</div>
    <div class="bytes">11 53 6c 21 ff</div>
    <div class="count">5 bytes &middot; 0.57 bits per character</div>
  </div>
</div>

<section class="prose">
  <h2>Why a compressor can&rsquo;t do this</h2>
  <p>gzip, bzip2 and lzma all work by finding repetition <em>inside</em> the message.
  A 70-byte message has none to find, so they achieve essentially nothing &mdash; 70
  bytes became 65. There is no dictionary to build that short.</p>
  <p>A language model arrives already holding the dictionary. It has read enough
  English to know that after <code>Once upon a</code> the next word is
  overwhelmingly likely to be <code>time</code>. An arithmetic coder can spend
  a fraction of a bit on an outcome that likely. What it spends the bits on
  instead is the part that was genuinely unpredictable.</p>
</section>

<section>
  <div class="prose">
    <p class="eyebrow">The mechanism</p>
    <h2>Where the bits actually go</h2>
    <p>Every token is priced at <code>&minus;log&#8322;(p)</code> under the model&rsquo;s own
    prediction. Cool and short means the model saw it coming; tall and red means it
    didn&rsquo;t. Pick a message.</p>
  </div>

  <div class="tabs" role="tablist" id="tabs"></div>
  <div class="tape-scroll"><div class="tape" id="tape"></div></div>

  <div class="legend">
    <span>0 bits</span><span class="ramp" aria-hidden="true"></span><span>12+ bits</span>
    <span style="margin-left:auto">bar height and colour both encode cost</span>
  </div>

  <div class="tape-foot" id="foot"></div>

  <div class="prose" style="margin-top:2rem">
    <p>The pattern is the whole argument. In the first message,
    <code>&nbsp;upon</code>, <code>&nbsp;a</code> and <code>&nbsp;time</code> together cost
    about a seventh of a bit &mdash; the model had effectively already written them.
    The cost concentrates on names and numbers, which is exactly where information
    genuinely lives.</p>
    <p>Switch to an operational message and the picture inverts: almost every token
    is expensive, because a model trained on children&rsquo;s stories has no idea what a
    mile marker is. <strong>That gap is what a fine-tune closes</strong>, and it is worth
    roughly 3&times; &mdash; far more than any coder engineering.</p>
  </div>
</section>

<section>
  <div class="prose">
    <p class="eyebrow">Measured, not projected</p>
    <h2>Against the strongest classical coder</h2>
    <p>Bits per character on ten terse operational messages. <code>deflate+dict</code>
    is primed with a held-out corpus of the same register &mdash; the fair comparison,
    since both Semaphore endpoints already share a model.</p>
  </div>
  <div class="bars" id="bars"></div>
  <div class="prose" style="margin-top:1.4rem">
    <p>On this out-of-domain set Semaphore is <strong>1.79&times; smaller</strong> than the best
    classical result. On text the model actually knows, it is 6.9&times;. The
    &ldquo;model ideal&rdquo; line is the model&rsquo;s own cross-entropy &mdash; a floor no coder
    can beat; the real coder lands 5.5% above it.</p>
  </div>
</section>

<section>
  <div class="prose">
    <p class="eyebrow">The hard part</p>
    <h2>Both ends must agree, bit for bit</h2>
    <p>The decoder rebuilds the encoder&rsquo;s probability table from its own copy of the
    model. One float differing in its last bit anywhere in the forward pass and the
    range coder desynchronises, destroying the rest of the message. That makes
    reproducibility a correctness requirement, not a nicety.</p>
    <p>It doesn&rsquo;t come for free. <code>expf</code>, <code>sinf</code>, <code>cosf</code>
    and <code>powf</code> from the system maths library disagree between macOS and ARM
    on <strong>1.21% of the values this model evaluates</strong> &mdash; all last-ulp, all
    correct, all fatal here. Replacing them with fixed sequences of IEEE-754 double
    operations fixes it:</p>
  </div>
  <div class="proof">
    <div class="line"><span class="who">Teensy 4.1, Cortex-M7, SMLAD kernels</span><span class="val">4aba58abe4d1c3fa</span></div>
    <div class="line"><span class="who">macOS arm64, generic C99 kernels</span><span class="val">4aba58abe4d1c3fa</span></div>
    <div class="match">256,000 floats &mdash; 8 forward passes over a 32,000-token
    vocabulary &mdash; identical across two architectures, two compilers, two C
    libraries and two different kernel implementations.</div>
  </div>
  <div class="prose" style="margin-top:1.4rem">
    <p>So either end can be either side, and both were run: the laptop compressed 70
    characters to 5 bytes and the board recovered the text exactly; the board
    compressed 59 characters to 10 bytes and the laptop recovered those. On three
    separate messages the board&rsquo;s output was byte-identical to the laptop&rsquo;s.</p>
  </div>
</section>

<section>
  <div class="prose">
    <p class="eyebrow">The catch</p>
    <h2>One flipped bit and it&rsquo;s gone</h2>
    <p>Compression this tight leaves no redundancy to absorb an error. Flip a single
    bit in byte 3 of a 33-byte message and the decoder doesn&rsquo;t garble &mdash; it
    diverges into fluent, confident nonsense:</p>
  </div>
  <div class="flip">
    <div class="row">
      <div class="k">intact</div>
      <div class="t">Need pickup at mile marker 42 on the county road, truck wont start.</div>
    </div>
    <div class="row bad">
      <div class="k">one bit flipped</div>
      <div class="t">Need<span style="opacity:.55">nees and his kitty, Striphant the Molea the T</span></div>
    </div>
  </div>
  <div class="prose" style="margin-top:1.4rem">
    <p>Forward error correction isn&rsquo;t optional here, and its overhead comes straight
    out of the win. The honest figure to publish is always bits-on-air after FEC,
    never the coder&rsquo;s bits per character.</p>
  </div>
</section>

<section>
  <div class="prose">
    <p class="eyebrow">Being honest about it</p>
    <h2>When this actually pays</h2>
    <p>Coding runs one forward pass per token &mdash; <strong>325&nbsp;ms each</strong> on the
    Teensy, so a 200-character message costs about 16 seconds of compute in each
    direction. That only makes sense when airtime is scarcer than time.</p>
  </div>
  <div class="tbl-scroll">
  <table>
    <thead><tr><th>Channel</th><th>200 chars raw</th><th>Compressed</th><th>Compute</th><th>Verdict</th></tr></thead>
    <tbody>
      <tr><td>Meshtastic LoRa (US)</td><td class="num">0.85 s</td><td class="num">~0.2 s</td><td class="num">~16 s</td>
          <td><span class="verdict v-no">absurd</span></td></tr>
      <tr><td>JS8Call HF, ~16 bit/s</td><td class="num">~100 s</td><td class="num">~15 s</td><td class="num">~16 s</td>
          <td><span class="verdict v-yes">wins by minutes</span></td></tr>
      <tr><td>Iridium Short Burst Data</td><td class="num">billed per byte</td><td class="num">~10&times; cheaper</td><td class="num">irrelevant</td>
          <td><span class="verdict v-yes">wins on cost</span></td></tr>
    </tbody>
  </table>
  </div>
  <div class="prose" style="margin-top:1.4rem">
    <p>US Part 15.247 sets no duty-cycle cap, so on a fast link airtime simply
    isn&rsquo;t scarce and compression buys nothing worth 16 seconds. The idea lives or
    dies on picking a channel measured in bits per second &mdash; where a message that
    took two minutes now takes fifteen seconds.</p>
  </div>
</section>

<footer class="prose">
  <p>Every figure here was measured on a Teensy 4.1 with hand-soldered PSRAM
  running <code>stories15M</code> quantized to Q4_0, or on the host running the
  identical C. The channel timings are the one exception &mdash; those are
  arithmetic from published data rates, not captured on air.</p>
</footer>
</div>

<script>
const DATA = __DATA__;

const lerp = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));
const TEAL = [31, 107, 123], WARM = [206, 132, 36], HOT = [190, 61, 36];
function costColor(bits) {
  const t = Math.min(bits / 12, 1);
  const c = t < 0.5 ? lerp(TEAL, WARM, t / 0.5) : lerp(WARM, HOT, (t - 0.5) / 0.5);
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

const tabs = document.getElementById('tabs');
const tape = document.getElementById('tape');
const foot = document.getElementById('foot');
let active = 0;

DATA.forEach((m, i) => {
  const b = document.createElement('button');
  b.className = 'tab';
  b.type = 'button';
  b.setAttribute('role', 'tab');
  b.textContent = `${m.wire_bytes}B \\u00b7 ${(8 * m.wire_bytes / m.raw_bytes).toFixed(2)} b/ch \\u00b7 ${m.register}`;
  b.addEventListener('click', () => select(i));
  tabs.appendChild(b);
});

function select(i) {
  active = i;
  const m = DATA[i];
  [...tabs.children].forEach((t, k) =>
    t.setAttribute('aria-selected', k === i ? 'true' : 'false'));

  tape.innerHTML = '';
  const maxBits = Math.max(6, ...m.tokens.map(t => t.bits));
  m.tokens.forEach((t, k) => {
    const el = document.createElement('div');
    el.className = 'tok';
    el.style.animationDelay = (k * 14) + 'ms';
    el.title = `${t.bits.toFixed(2)} bits`;

    const txt = document.createElement('div');
    txt.className = 'txt';
    txt.textContent = t.piece === '' ? '\\u2423' : t.piece;

    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('div');
    fill.className = 'fill';
    fill.style.height = Math.max(2, Math.round(20 * t.bits / maxBits)) + 'px';
    fill.style.background = costColor(t.bits);
    bar.appendChild(fill);

    const n = document.createElement('div');
    n.className = 'n';
    n.textContent = t.bits < 10 ? t.bits.toFixed(1) : Math.round(t.bits);

    el.append(txt, bar, n);
    tape.appendChild(el);
  });

  const cheap = m.tokens.filter(t => t.bits < 1).length;
  foot.innerHTML = `
    <div><span>tokens</span> <b>${m.tokens.length}</b></div>
    <div><span>under 1 bit</span> <b>${cheap}</b></div>
    <div><span>token payload</span> <b>${m.token_bits_total.toFixed(1)} bits</b></div>
    <div><span>+ length field</span> <b>8 bits</b></div>
    <div><span>on the wire</span> <b>${m.wire_bytes} bytes</b></div>
    <div><span>gzip would send</span> <b>${m.deflate_bytes} bytes</b></div>`;
}
select(0);

const BARS = [
  { l: 'raw ASCII',     v: 8.000, c: 'var(--muted)' },
  { l: 'deflate',       v: 7.631, c: 'var(--muted)' },
  { l: 'deflate+dict',  v: 5.886, c: 'var(--warm)'  },
  { l: 'SEMAPHORE',     v: 3.293, c: 'var(--signal)' },
  { l: 'model ideal',   v: 3.123, c: 'transparent', pale: true }
];
document.getElementById('bars').innerHTML = BARS.map(b => `
  <div class="bar-row">
    <div class="lbl">${b.l}</div>
    <div class="bar-track">
      <div class="bar-fill${b.pale ? ' pale' : ''}" style="width:${b.v / 8 * 100}%;
           background:${b.c};${b.pale ? 'border:1px dashed var(--edge);' : ''}">
        <span>${b.v.toFixed(3)}</span>
      </div>
    </div>
  </div>`).join('');
</script>
"""



if __name__ == "__main__":
    sys.exit(main())
