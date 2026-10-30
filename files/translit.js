// Udgam transliterator runner: romanised Hindi -> Devanagari, in the browser or node, no dependencies.
// Files (plan: translit/PLAN_R2.md): model.json + model.q4 (format udgam-translit-q4-v1, from r2_train.py export_q4),
// optional words.bin (wordpack.py pack or pack_v2) and rerank.json {lam, off, layout, beam}. Any file may be gzip; it is
// unpacked with DecompressionStream. Same maths as the Python scorer: Tiny encoder-decoder, beam 4 with the
// fairseq-like rules of indicxlit_torch.beam_search, then score = model log-prob / length + lam * (banded log p(word),
// or the list floor - off when the word is not in the list), over the finished candidates only.

const SQRT1_2 = Math.SQRT1_2;

function f16(h) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

// erf with |error| < 1.2e-7 (Numerical Recipes erfc, Chebyshev fit); GELU here is the exact (erf) form, as F.gelu
function erf(x) {
  const z = Math.abs(x), t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 +
    t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? 1 - r : r - 1;
}

async function bytesOf(src) {
  let b = src instanceof ArrayBuffer ? new Uint8Array(src) : src;
  if (typeof src === "string") b = new Uint8Array(await (await fetch(src)).arrayBuffer());
  if (b.length > 2 && b[0] === 0x1f && b[1] === 0x8b) {
    const s = new Blob([b]).stream().pipeThrough(new DecompressionStream("gzip"));
    b = new Uint8Array(await new Response(s).arrayBuffer());
  }
  return b;
}

function loadWeights(man, raw) {
  const W = {}, g = man.group, dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  for (const t of man.tensors) {
    const n = t.shape.reduce((a, b) => a * b, 1), w = new Float32Array(n);
    if (t.kind === "q4") {
      const cols = t.shape[1], per = cols / g, so = t.offset + n / 2;
      for (let i = 0; i < n; i++) {
        const byte = raw[t.offset + (i >> 1)], q = (i & 1 ? byte >> 4 : byte & 15) - 8;
        const r = (i / cols) | 0, c = i % cols;
        w[i] = q * f16(dv.getUint16(so + 2 * (r * per + ((c / g) | 0)), true));
      }
    } else {
      for (let i = 0; i < n; i++) w[i] = f16(dv.getUint16(t.offset + 2 * i, true));
    }
    W[t.name] = w;
  }
  return W;
}

// y = W x + b for a Linear with weight [out, in]; x may be a slice of a bigger array (xo = offset)
function linear(W, b, x, xo, nin, nout) {
  const y = new Float32Array(nout);
  for (let o = 0; o < nout; o++) {
    let s = b ? b[o] : 0;
    const wo = o * nin;
    for (let i = 0; i < nin; i++) s += W[wo + i] * x[xo + i];
    y[o] = s;
  }
  return y;
}

function layerNorm(x, w, b) {
  const n = x.length, y = new Float32Array(n);
  let m = 0, v = 0;
  for (let i = 0; i < n; i++) m += x[i];
  m /= n;
  for (let i = 0; i < n; i++) v += (x[i] - m) ** 2;
  const r = 1 / Math.sqrt(v / n + 1e-5);
  for (let i = 0; i < n; i++) y[i] = (x[i] - m) * r * w[i] + b[i];
  return y;
}

// one query row against S cached keys/values ([S, d] flattened), h heads, softmax(q k / sqrt(hd)) v
function attend(q, K, V, S, d, h) {
  const hd = d / h, out = new Float32Array(d), sc = new Float64Array(S), scale = 1 / Math.sqrt(hd);
  for (let hh = 0; hh < h; hh++) {
    const o = hh * hd;
    let mx = -Infinity;
    for (let s = 0; s < S; s++) {
      let a = 0;
      for (let i = 0; i < hd; i++) a += q[o + i] * K[s * d + o + i];
      sc[s] = a * scale; if (sc[s] > mx) mx = sc[s];
    }
    let z = 0;
    for (let s = 0; s < S; s++) { sc[s] = Math.exp(sc[s] - mx); z += sc[s]; }
    for (let s = 0; s < S; s++) {
      const p = sc[s] / z;
      for (let i = 0; i < hd; i++) out[o + i] += p * V[s * d + o + i];
    }
  }
  return out;
}

function ffn(W, p, x, d, f) {
  const y = layerNorm(x, W[p + "ln3.weight"], W[p + "ln3.bias"]);
  const u = linear(W[p + "fc1.weight"], W[p + "fc1.bias"], y, 0, d, f);
  for (let i = 0; i < f; i++) u[i] = 0.5 * u[i] * (1 + erf(u[i] * SQRT1_2));
  const v = linear(W[p + "fc2.weight"], W[p + "fc2.bias"], u, 0, f, d);
  for (let i = 0; i < d; i++) x[i] += v[i];
}

function add(x, y) { for (let i = 0; i < x.length; i++) x[i] += y[i]; }

class WordList {
  // decoded into full byte strings (one byte per Devanagari char, U+0900 + b - 0x80) with offsets; groups are sorted runs
  constructor(raw, layout) {
    const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
    const bands = dv.getUint32(0, true);
    let counts, p;
    if (layout === "v2") { counts = [dv.getUint32(4, true)]; p = 8; }
    else { counts = []; for (let i = 0; i < bands; i++) counts.push(dv.getUint32(4 + 4 * i, true)); p = 4 + 4 * bands; }
    this.centres = []; for (let i = 0; i < bands; i++) this.centres.push(dv.getFloat32(p + 4 * i, true));
    p += 4 * bands;
    const n = counts.reduce((a, b) => a + b, 0);
    this.off = new Uint32Array(n + 1); this.band = new Uint8Array(n); this.groups = [];
    const buf = []; let len = 0, k = 0, chunk = new Uint8Array(1 << 20), cl = 0;
    const flush = () => { buf.push(chunk.subarray(0, cl)); chunk = new Uint8Array(1 << 20); cl = 0; };
    let prev = new Uint8Array(0);
    for (let gi = 0; gi < counts.length; gi++) {
      const gs = k;
      for (let j = 0; j < counts[gi]; j++, k++) {
        const pre = raw[p++]; let e = p; while (raw[e] !== 0) e++;
        const w = new Uint8Array(pre + e - p); w.set(prev.subarray(0, pre)); w.set(raw.subarray(p, e), pre); p = e + 1;
        if (cl + w.length > chunk.length) flush();
        chunk.set(w, cl); cl += w.length; this.off[k] = len; len += w.length; prev = w;
        if (layout !== "v2") this.band[k] = gi;
      }
      this.groups.push([gs, k]);
      prev = new Uint8Array(0);
    }
    flush(); this.off[n] = len;
    this.bytes = new Uint8Array(len); let o = 0; for (const c of buf) { this.bytes.set(c, o); o += c.length; }
    if (layout === "v2") this.band.set(raw.subarray(p, p + n));
    let fl = Infinity; for (let i = 0; i < n; i++) fl = Math.min(fl, this.centres[this.band[i]]);
    this.floor = fl; this.size = n;
  }
  cmp(i, w) {
    const a = this.off[i], b = this.off[i + 1], n = Math.min(b - a, w.length);
    for (let j = 0; j < n; j++) { const d = this.bytes[a + j] - w[j]; if (d) return d; }
    return (b - a) - w.length;
  }
  // banded log p(word), or undefined when absent (or not all Devanagari)
  logp(s) {
    const w = new Uint8Array(s.length);
    for (let i = 0; i < s.length; i++) { const c = s.charCodeAt(i); if (c < 0x900 || c > 0x97f) return undefined; w[i] = c - 0x900 + 0x80; }
    for (const [gs, ge] of this.groups) {
      let lo = gs, hi = ge - 1;
      while (lo <= hi) {
        const m = (lo + hi) >> 1, c = this.cmp(m, w);
        if (c === 0) return this.centres[this.band[m]];
        if (c < 0) lo = m + 1; else hi = m - 1;
      }
    }
    return undefined;
  }
}

export class Transliterator {
  constructor(man, raw, list, rerank) {
    this.man = man; this.c = man.config; this.W = loadWeights(man, raw); this.list = list; this.rr = rerank || {};
    this.sidx = new Map(man.src.map((ch, i) => [ch, i])); this.tgt = man.tgt; this.V = man.tgt.length;
    const d = this.c.d, half = d / 2, e = Math.log(10000) / (half - 1);
    this.pos = new Float32Array(64 * d);
    for (let p = 0; p < 64; p++) for (let i = 0; i < half; i++) {
      if (p === 1) continue;
      this.pos[p * d + i] = Math.sin(p * Math.exp(-i * e)); this.pos[p * d + half + i] = Math.cos(p * Math.exp(-i * e));
    }
  }

  static async load({ model, json, words, rerank }) {
    const man = json instanceof Uint8Array || json instanceof ArrayBuffer || typeof json === "string"
      ? JSON.parse(new TextDecoder().decode(await bytesOf(json))) : json;
    const raw = await bytesOf(model);
    let rr = rerank;
    if (typeof rr === "string") rr = await (await fetch(rr)).json();
    const list = words ? new WordList(await bytesOf(words), (rr && rr.layout) || "v1") : null;
    return new Transliterator(man, raw, list, rr);
  }

  embed(table, tok, p) {
    const d = this.c.d, x = new Float32Array(d), E = this.W[table], s = Math.sqrt(d);
    for (let i = 0; i < d; i++) x[i] = E[tok * d + i] * s + this.pos[p * d + i];
    return x;
  }

  encode(toks) {
    const { d, h, f, le, ld } = this.c, W = this.W, S = toks.length;
    let X = toks.map((t, i) => this.embed("se.weight", t, i + 2));
    for (let l = 0; l < le; l++) {
      const p = `enc.${l}.`, K = new Float32Array(S * d), V = new Float32Array(S * d), Q = [];
      for (let s = 0; s < S; s++) {
        const y = layerNorm(X[s], W[p + "ln1.weight"], W[p + "ln1.bias"]);
        const qkv = linear(W[p + "qkv.weight"], W[p + "qkv.bias"], y, 0, d, 3 * d);
        Q.push(qkv.subarray(0, d)); K.set(qkv.subarray(d, 2 * d), s * d); V.set(qkv.subarray(2 * d), s * d);
      }
      X = X.map((x, s) => {
        const z = Float32Array.from(x);
        add(z, linear(W[p + "o.weight"], W[p + "o.bias"], attend(Q[s], K, V, S, d, h), 0, d, d));
        ffn(W, p, z, d, f); return z;
      });
    }
    const enc = new Float32Array(S * d);
    X.forEach((x, s) => enc.set(layerNorm(x, W["eln.weight"], W["eln.bias"]), s * d));
    // cross-attention keys and values, once per decoder layer
    const cross = [];
    for (let l = 0; l < ld; l++) {
      const p = `dec.${l}.`, K = new Float32Array(S * d), V = new Float32Array(S * d);
      for (let s = 0; s < S; s++) {
        const kv = linear(W[p + "kv2.weight"], W[p + "kv2.bias"], enc, s * d, d, 2 * d);
        K.set(kv.subarray(0, d), s * d); V.set(kv.subarray(d), s * d);
      }
      cross.push([K, V]);
    }
    return { cross, S };
  }

  // one decoder step for one hypothesis: appends this token's self-attention keys/values to its cache, returns log-probs
  step(tok, L, cache, enc) {
    const { d, h, f, ld } = this.c, W = this.W;
    const x = this.embed("te.weight", tok, L + 1);
    for (let l = 0; l < ld; l++) {
      const p = `dec.${l}.`, c = cache[l];
      const y = layerNorm(x, W[p + "ln1.weight"], W[p + "ln1.bias"]);
      const qkv = linear(W[p + "qkv.weight"], W[p + "qkv.bias"], y, 0, d, 3 * d);
      c.K.set(qkv.subarray(d, 2 * d), (L - 1) * d); c.V.set(qkv.subarray(2 * d), (L - 1) * d);
      add(x, linear(W[p + "o.weight"], W[p + "o.bias"], attend(qkv.subarray(0, d), c.K, c.V, L, d, h), 0, d, d));
      const y2 = layerNorm(x, W[p + "ln2.weight"], W[p + "ln2.bias"]);
      const q2 = linear(W[p + "q2.weight"], W[p + "q2.bias"], y2, 0, d, d);
      add(x, linear(W[p + "o2.weight"], W[p + "o2.bias"], attend(q2, enc.cross[l][0], enc.cross[l][1], enc.S, d, h), 0, d, d));
      ffn(W, p, x, d, f);
    }
    const z = layerNorm(x, W["dln.weight"], W["dln.bias"]), V = this.V, E = this.W["te.weight"], lg = new Float64Array(V);
    let mx = -Infinity;
    for (let v = 0; v < V; v++) { let s = 0; for (let i = 0; i < d; i++) s += z[i] * E[v * d + i]; lg[v] = s; if (s > mx) mx = s; }
    let sum = 0; for (let v = 0; v < V; v++) sum += Math.exp(lg[v] - mx);
    const lz = mx + Math.log(sum);
    for (let v = 0; v < V; v++) lg[v] -= lz;
    return lg;
  }

  // beam search, same rules as the reference: returns the finished hypotheses [{text, score}], best model score first
  nbest(word, beam = 4, maxLen = 40) {
    const { d, ld } = this.c, V = this.V, K = beam, EOS = 2, PAD = 1;
    const toks = [...word.slice(0, 61)].map((ch) => this.sidx.get(ch) ?? 3); toks.push(EOS);
    const enc = this.encode(toks);
    const newCache = () => Array.from({ length: ld }, () => ({ K: new Float32Array((maxLen + 1) * d), V: new Float32Array((maxLen + 1) * d) }));
    const copyCache = (c) => c.map((x) => ({ K: Float32Array.from(x.K), V: Float32Array.from(x.V) }));
    let hyps = [{ seq: [EOS], score: 0, cache: newCache() }];
    for (let k = 1; k < K; k++) hyps.push({ seq: [EOS], score: -Infinity, cache: null });
    const fin = [];
    for (let step = 0; step < maxLen; step++) {
      const L = hyps[0].seq.length, cand = [];
      for (let r = 0; r < K; r++) {
        const hy = hyps[r];
        if (hy.score === -Infinity) continue;
        const lp = this.step(hy.seq[L - 1], L, hy.cache, enc);
        lp[PAD] = -Infinity; lp[0] = -Infinity; lp[3] = -Infinity;
        if (step === 0) lp[EOS] = -Infinity;
        for (let v = 0; v < V; v++) if (lp[v] > -Infinity) cand.push([hy.score + lp[v], r * V + v]);
      }
      // top 2K by score; equal scores keep the lower flat index first (as torch.topk does in practice)
      cand.sort((a, b) => b[0] - a[0] || a[1] - b[1]);
      const top = cand.slice(0, 2 * K), next = [];
      for (let j = 0; j < top.length; j++) {
        const [sc, idx] = top[j], r = (idx / V) | 0, t = idx % V;
        if (t === EOS) {
          if (j < K && fin.length < K) fin.push({ tokens: hyps[r].seq.slice(1), score: sc / L });
          continue;
        }
        if (next.length < K) next.push({ r, t, sc });
      }
      if (fin.length >= K || !next.length) break;
      const used = new Set();
      hyps = next.map(({ r, t, sc }) => {
        const c = used.has(r) ? copyCache(hyps[r].cache) : hyps[r].cache; used.add(r);
        return { seq: [...hyps[r].seq, t], score: sc, cache: c };
      });
      while (hyps.length < K) hyps.push({ seq: [...hyps[0].seq.slice(0, -1), PAD], score: -Infinity, cache: null });
    }
    fin.sort((a, b) => b.score - a.score);
    return fin.map((h) => ({ text: norm(h.tokens.filter((t) => t > 3).map((t) => this.tgt[t]).join("")), score: h.score }));
  }

  // best Devanagari spelling for one romanised word, plus the ranked candidates
  word(w) {
    const src = w.trim().toLowerCase();
    if (!src) return { best: "", nbest: [], candidates: [] };
    const { lam = 0, off = 0, beam = 4 } = this.rr, nb = this.nbest(src, beam);
    for (const c of nb) {
      const lp = this.list ? this.list.logp(c.text) : undefined;
      c.final = this.list && lam ? c.score + lam * (lp === undefined ? this.list.floor - off : lp) : c.score;
      c.inList = lp !== undefined;
    }
    let best = null;
    for (const c of nb) if (!best || c.final > best.final) best = c;
    return { best: best ? best.text : "", nbest: nb, candidates: [...nb].sort((a, b) => b.final - a.final) };
  }

  // a line of text: runs of Latin letters are transliterated, everything else is kept
  text(s) {
    return s.replace(/[A-Za-z]+/g, (w) => this.word(w).best || w);
  }
}

function norm(s) {
  return s.normalize("NFC").replace(/[‌‍]/g, "");
}
