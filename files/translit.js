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

// GELU from a table of the exact (erf) values every 1/1024 on [-6, 6], linear in between: within 2.4e-7 of the erf
// formula (float32 rounding of the table near |x| = 6). Outside that range GELU(x) is 0 or x to within 1e-8
const GELU_X = 6, GELU_N = 1024;
const GELU_T = Float32Array.from({ length: 2 * GELU_X * GELU_N + 2 }, (_, i) => {
  const x = i / GELU_N - GELU_X;
  return 0.5 * x * (1 + erf(x * SQRT1_2));
});
function gelu(x) {
  if (x <= -GELU_X) return 0;
  if (x >= GELU_X) return x;
  const p = (x + GELU_X) * GELU_N, i = p | 0, a = GELU_T[i];
  return a + (p - i) * (GELU_T[i + 1] - a);
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

// every tensor as f32; alloc(n) gives the array to fill (a view into the WebAssembly memory when SIMD is used)
function loadWeights(man, raw, alloc = (n) => new Float32Array(n)) {
  const W = {}, g = man.group, dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  for (const t of man.tensors) {
    const n = t.shape.reduce((a, b) => a * b, 1), w = alloc(n);
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

// The kernels: lin is Y = X W^T + b for a Linear with weight [out, in] on B rows (X is [B, nin] flattened, Y is
// [B, nout]; with acc the result is added to Y instead, for the residual connections), ln is LayerNorm of B rows and gelu
// is GELU in place. These are the plain-JS versions, used when WebAssembly SIMD is not available; the WebAssembly ones
// below take the same arguments
function linearRows(W, b, X, Y, B, nin, nout, acc) {
  for (let o = 0; o < nout; o++) {
    const wo = o * nin, s = b ? b[o] : 0;
    let r = 0;
    for (; r + 4 <= B; r += 4) {
      const x0 = r * nin, x1 = x0 + nin, x2 = x1 + nin, x3 = x2 + nin;
      let s0 = s, s1 = s, s2 = s, s3 = s;
      for (let i = 0; i < nin; i++) {
        const w = W[wo + i];
        s0 += w * X[x0 + i]; s1 += w * X[x1 + i]; s2 += w * X[x2 + i]; s3 += w * X[x3 + i];
      }
      put(Y, r * nout + o, s0, acc); put(Y, (r + 1) * nout + o, s1, acc);
      put(Y, (r + 2) * nout + o, s2, acc); put(Y, (r + 3) * nout + o, s3, acc);
    }
    for (; r < B; r++) {
      const xo = r * nin;
      let s0 = s;
      for (let i = 0; i < nin; i++) s0 += W[wo + i] * X[xo + i];
      put(Y, r * nout + o, s0, acc);
    }
  }
}

// Y[k] = s, or with acc Y[k] + s rounded to f32 first (as adding a stored output would)
function put(Y, k, s, acc) { Y[k] = acc ? Y[k] + Math.fround(s) : s; }

function layerNormRows(X, w, b, Y, B, n) {
  for (let r = 0; r < B; r++) {
    const o = r * n;
    let m = 0, v = 0;
    for (let i = 0; i < n; i++) m += X[o + i];
    m /= n;
    for (let i = 0; i < n; i++) { const t = X[o + i] - m; v += t * t; }
    const s = 1 / Math.sqrt(v / n + 1e-5);
    for (let i = 0; i < n; i++) Y[o + i] = (X[o + i] - m) * s * w[i] + b[i];
  }
}

function geluRows(U, n) { for (let i = 0; i < n; i++) U[i] = gelu(U[i]); }

// The same three kernels in WebAssembly, assembled here byte by byte (no binary blob, no build step). Pointers are byte
// offsets into the shared memory, and b = 0 means no bias. lin takes two outputs and four rows at a time: eight f32x4
// sums run over the input 4 floats at a time (each weight vector serves four rows, each input vector two outputs), then
// each sum's lanes are added as (0 + 1) + (2 + 3), plus the bias. Leftover rows go one at a time, an odd last output on
// its own; nin must be a multiple of 4. ln and gelu go one value at a time in f64, the same sums in the same order as
// the JS versions
function kernelBytes() {
  const leb = (n) => { const o = []; do { let b = n & 127; n >>>= 7; if (n) b |= 128; o.push(b); } while (n); return o; };
  const sleb = (n) => {
    const o = [];
    for (;;) { const b = n & 127; n >>= 7; if ((n === 0 && !(b & 64)) || (n === -1 && b & 64)) return [...o, b]; o.push(b | 128); }
  };
  const vec = (a) => [...leb(a.length), ...a], sec = (id, a) => [id, ...vec(a)];
  const name = (s) => vec([...s].map((c) => c.charCodeAt(0))), range = (n) => [...Array(n).keys()];
  const c32 = (n) => [0x41, ...sleb(n)], f64c = (x) => {
    const dv = new DataView(new ArrayBuffer(8)); dv.setFloat64(0, x, true); return [0x44, ...new Uint8Array(dv.buffer)];
  };
  const iadd = [0x6a], imul = [0x6c], geu = [0x4f], gtu = [0x4b], fadd = [0x92], select = [0x1b];
  const f32load = [0x2a, 2, 0], f32store = [0x38, 2, 0], promote = [0xbb], demote = [0xb6];
  const dadd = [0xa0], dsub = [0xa1], dmul = [0xa2], ddiv = [0xa3], dsqrt = [0x9f];
  const simd = (op, ...imm) => [0xfd, ...leb(op), ...imm];
  const vload = simd(0, 4, 0), vadd = simd(228), vmul = simd(230), lane = (k) => simd(31, k), vzero = simd(12, ...Array(16).fill(0));
  // block { loop { if exit: break; body; continue } }
  const loop = (exit, ...body) => [0x02, 0x40, 0x03, 0x40, ...exit, 0x0d, 1, ...body.flat(), 0x0c, 0, 0x0b, 0x0b];
  // a function's variables: the parameters, then named locals of each type; gives get/set/tee by name and the declarations
  const vars = (params, groups) => {
    const L = {}, decl = [groups.length];
    let n = 0;
    for (const p of params) L[p] = n++;
    for (const [type, names] of groups) { decl.push(names.length, type); for (const v of names) L[v] = n++; }
    return { decl, get: (v) => [0x20, ...leb(L[v])], set: (v) => [0x21, ...leb(L[v])], tee: (v) => [0x22, ...leb(L[v])] };
  };

  const lin = (() => {
    const { decl, get, set, tee } = vars(["W", "b", "X", "Y", "B", "nin", "nout", "acc"], [
      [0x7f, ["o", "r", "nb", "i", "x0", "x1", "x2", "x3", "w0", "w1", "ya"]], [0x7d, ["bo0", "bo1", "t"]],
      [0x7b, ["a00", "a01", "a02", "a03", "a10", "a11", "a12", "a13", "wv0", "wv1", "xv"]]]);
    const a = (j, k) => `a${j}${k}`;
    // x_k = X + (r + k) * nb
    const rowsAt = (nr) => [...get("X"), ...get("r"), ...get("nb"), ...imul, ...iadd,
      ...range(nr).flatMap((k) => [...(k ? [...get("nb"), ...iadd] : []), ...(k < nr - 1 ? tee : set)("x" + k)])];
    // a_jk = sum over i of w_j[i] * x_k[i], 4 floats at a time
    const dots = (nr, no) => [...range(no).flatMap((j) => range(nr).flatMap((k) => [...vzero, ...set(a(j, k))])),
      ...c32(0), ...set("i"),
      ...loop([...get("i"), ...get("nb"), ...geu],
        range(no).flatMap((j) => [...get("w" + j), ...get("i"), ...iadd, ...vload, ...set("wv" + j)]),
        range(nr).flatMap((k) => [...get("x" + k), ...get("i"), ...iadd, ...vload, ...set("xv"),
          ...range(no).flatMap((j) => [...get(a(j, k)), ...get("wv" + j), ...get("xv"), ...vmul, ...vadd, ...set(a(j, k))])]),
        [...get("i"), ...c32(16), ...iadd, ...set("i")])];
    // t = lanes of a_jk added up + bias; Y[r + k, o + j] = acc ? Y[...] + t : t
    const stores = (nr, no) => range(nr).flatMap((k) => range(no).flatMap((j) => {
      const v = a(j, k);
      return [...get("Y"), ...get("r"), ...c32(k), ...iadd, ...get("nout"), ...imul, ...get("o"), ...c32(j), ...iadd, ...iadd,
        ...c32(4), ...imul, ...iadd, ...tee("ya"),
        ...get(v), ...lane(0), ...get(v), ...lane(1), ...fadd, ...get(v), ...lane(2), ...get(v), ...lane(3), ...fadd, ...fadd,
        ...get("bo" + j), ...fadd, ...tee("t"), ...get("ya"), ...f32load, ...fadd, ...get("t"), ...get("acc"), ...select, ...f32store];
    }));
    const bias = (j) => [...get("b"), 0x04, 0x7d, ...get("b"), ...get("o"), ...c32(j), ...iadd, ...c32(4), ...imul, ...iadd,
      ...f32load, 0x05, 0x43, 0, 0, 0, 0, 0x0b, ...set("bo" + j)];
    // outputs o .. o + no - 1 for all rows, then o += no
    const outputs = (no) => [...get("W"), ...get("o"), ...get("nb"), ...imul, ...iadd,
      ...(no > 1 ? [...tee("w0"), ...get("nb"), ...iadd, ...set("w1")] : set("w0")), ...range(no).flatMap(bias),
      ...c32(0), ...set("r"),
      ...loop([...get("r"), ...c32(4), ...iadd, ...get("B"), ...gtu], rowsAt(4), dots(4, no), stores(4, no), [...get("r"), ...c32(4), ...iadd, ...set("r")]),
      ...loop([...get("r"), ...get("B"), ...geu], rowsAt(1), dots(1, no), stores(1, no), [...get("r"), ...c32(1), ...iadd, ...set("r")]),
      ...get("o"), ...c32(no), ...iadd, ...set("o")];
    return [...decl, ...get("nin"), ...c32(4), ...imul, ...set("nb"),
      ...loop([...get("o"), ...c32(2), ...iadd, ...get("nout"), ...gtu], outputs(2)),
      ...loop([...get("o"), ...get("nout"), ...geu], outputs(1)), 0x0b];
  })();

  const ln = (() => {
    const { decl, get, set, tee } = vars(["X", "w", "b", "Y", "B", "n"], [[0x7f, ["r", "i", "xr", "yr", "nb"]],
      [0x7c, ["m", "v", "rs", "t", "nf"]]]);
    const x = [...get("xr"), ...get("i"), ...iadd, ...f32load, ...promote];
    const each = (body) => [...c32(0), ...set("i"), ...loop([...get("i"), ...get("nb"), ...geu], body, [...get("i"), ...c32(4), ...iadd, ...set("i")])];
    const at = (p) => [...get(p), ...get("i"), ...iadd, ...f32load, ...promote];
    return [...decl, ...get("n"), ...c32(4), ...imul, ...set("nb"), ...get("n"), 0xb7, ...set("nf"),
      ...loop([...get("r"), ...get("B"), ...geu],
        [...get("X"), ...get("r"), ...get("nb"), ...imul, ...iadd, ...set("xr"), ...get("Y"), ...get("r"), ...get("nb"), ...imul, ...iadd, ...set("yr")],
        // m = mean, v = sum of squared deviations, rs = 1 / sqrt(v / n + 1e-5)
        [...f64c(0), ...set("m")], each([...get("m"), ...x, ...dadd, ...set("m")]), [...get("m"), ...get("nf"), ...ddiv, ...set("m")],
        [...f64c(0), ...set("v")], each([...x, ...get("m"), ...dsub, ...tee("t"), ...get("t"), ...dmul, ...get("v"), ...dadd, ...set("v")]),
        [...f64c(1), ...get("v"), ...get("nf"), ...ddiv, ...f64c(1e-5), ...dadd, ...dsqrt, ...ddiv, ...set("rs")],
        each([...get("yr"), ...get("i"), ...iadd, ...x, ...get("m"), ...dsub, ...get("rs"), ...dmul, ...at("w"), ...dmul, ...at("b"), ...dadd,
          ...demote, ...f32store]),
        [...get("r"), ...c32(1), ...iadd, ...set("r")]), 0x0b];
  })();

  // GELU from the table T, as the JS gelu
  const gelu = (() => {
    const { decl, get, set, tee } = vars(["U", "n", "T"], [[0x7f, ["i", "nb", "k", "ka"]], [0x7c, ["x", "p", "a"]]]);
    return [...decl, ...get("n"), ...c32(4), ...imul, ...set("nb"),
      ...loop([...get("i"), ...get("nb"), ...geu],
        [...get("U"), ...get("i"), ...iadd,
          ...get("U"), ...get("i"), ...iadd, ...f32load, ...promote, ...tee("x"), ...f64c(-GELU_X), 0x65, 0x04, 0x7c, ...f64c(0), 0x05,
          ...get("x"), ...f64c(GELU_X), 0x66, 0x04, 0x7c, ...get("x"), 0x05,
          // p = (x + 6) * 1024, k = trunc(p), a = T[k]: a + (p - k) * (T[k + 1] - a)
          ...get("x"), ...f64c(GELU_X), ...dadd, ...f64c(GELU_N), ...dmul, ...tee("p"), 0xfc, 2, ...tee("k"),
          ...c32(4), ...imul, ...get("T"), ...iadd, ...tee("ka"), ...f32load, ...promote, ...tee("a"),
          ...get("p"), ...get("k"), 0xb7, ...dsub, ...get("ka"), 0x2a, 2, 4, ...promote, ...get("a"), ...dsub, ...dmul, ...dadd,
          0x0b, 0x0b, ...demote, ...f32store],
        [...get("i"), ...c32(4), ...iadd, ...set("i")]), 0x0b];
  })();

  const type = (n) => [0x60, n, ...Array(n).fill(0x7f), 0];   // (n x i32) -> ()
  return new Uint8Array([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0,
    ...sec(1, [3, ...type(8), ...type(6), ...type(3)]),
    ...sec(2, [1, ...name("e"), ...name("m"), 2, 0, 1]),   // import memory "e" "m", at least 1 page
    ...sec(3, [3, 0, 1, 2]),
    ...sec(7, [3, ...name("lin"), 0, 0, ...name("ln"), 0, 1, ...name("gelu"), 0, 2]),
    ...sec(10, [3, ...vec(lin), ...vec(ln), ...vec(gelu)])]);
}

// One Float32Array holds the GELU table, the scratch rows and every weight; alloc hands out 16-byte-aligned views of it.
// The scratch rows: X the residual stream, N a LayerNorm output, A attention outputs, Q cross-attention queries, H the
// wider Linear outputs (qkv, fc1, logits). With SIMD the array is the WebAssembly memory and the kernels run there
const CAP = 64;   // the most rows in one call: the letters of a word + 1, or the beam
const pad4 = (n) => (n + 3) & ~3;

function arenaSize(man) {
  const { d, f } = man.config, wide = Math.max(3 * d, f, man.tgt.length);
  return 4 + pad4(GELU_T.length) + 4 * pad4(CAP * d) + pad4(CAP * wide) +
    man.tensors.reduce((a, t) => a + pad4(t.shape.reduce((p, q) => p * q, 1)), 0);
}

// the WebAssembly memory and kernels, or null when the platform has no WebAssembly SIMD
async function simdModule(man) {
  if (typeof WebAssembly !== "object" || man.config.d % 4 || man.config.f % 4) return null;
  const bytes = kernelBytes();
  if (!WebAssembly.validate(bytes)) return null;
  const mem = new WebAssembly.Memory({ initial: Math.ceil(arenaSize(man) * 4 / 65536) });
  return { mem, fn: (await WebAssembly.instantiate(bytes, { e: { m: mem } })).instance.exports };
}

function kernels(man, wasm) {
  const { d, f } = man.config, wide = Math.max(3 * d, f, man.tgt.length);
  const F = wasm ? new Float32Array(wasm.mem.buffer) : new Float32Array(arenaSize(man));
  let top = 4;   // in floats; offset 0 stays free, so that b = 0 can mean no bias
  const alloc = (n) => { const w = F.subarray(top, top + n); top += pad4(n); return w; };
  const T = alloc(GELU_T.length); T.set(GELU_T);
  const buf = { X: alloc(CAP * d), N: alloc(CAP * d), A: alloc(CAP * d), Q: alloc(CAP * d), H: alloc(CAP * wide) };
  if (!wasm) return { alloc, buf, simd: false, lin: linearRows, ln: layerNormRows, gelu: geluRows };
  const k = wasm.fn, p = (a) => a.byteOffset;
  return {
    alloc, buf, simd: true,
    lin: (W, b, X, Y, B, nin, nout, acc) => k.lin(p(W), b ? p(b) : 0, p(X), p(Y), B, nin, nout, acc ? 1 : 0),
    ln: (X, w, b, Y, B, n) => k.ln(p(X), p(w), p(b), p(Y), B, n),
    gelu: (U, n) => k.gelu(p(U), n, p(T)),
  };
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
  constructor(man, raw, list, rerank, wasm) {
    this.man = man; this.c = man.config; this.k = kernels(man, wasm || null); this.W = loadWeights(man, raw, this.k.alloc);
    this.simd = this.k.simd; this.list = list; this.rr = rerank || {};
    this.cache = new Map(); this.cacheSize = 2000;
    this.sidx = new Map(man.src.map((ch, i) => [ch, i])); this.tgt = man.tgt; this.V = man.tgt.length;
    const d = this.c.d, half = d / 2, e = Math.log(10000) / (half - 1);
    this.pos = new Float32Array(64 * d);
    for (let p = 0; p < 64; p++) for (let i = 0; i < half; i++) {
      if (p === 1) continue;
      this.pos[p * d + i] = Math.sin(p * Math.exp(-i * e)); this.pos[p * d + half + i] = Math.cos(p * Math.exp(-i * e));
    }
  }

  // simd: false keeps everything in plain JS even where WebAssembly SIMD is available
  static async load({ model, json, words, rerank, simd = true }) {
    const man = json instanceof Uint8Array || json instanceof ArrayBuffer || typeof json === "string"
      ? JSON.parse(new TextDecoder().decode(await bytesOf(json))) : json;
    const raw = await bytesOf(model);
    let rr = rerank;
    if (typeof rr === "string") rr = await (await fetch(rr)).json();
    const list = words ? new WordList(await bytesOf(words), (rr && rr.layout) || "v1") : null;
    return new Transliterator(man, raw, list, rr, simd ? await simdModule(man) : null);
  }

  // token embedding * sqrt(d) + position encoding, written to out at offset o
  embed(table, tok, p, out, o) {
    const d = this.c.d, E = this.W[table], s = Math.sqrt(d);
    for (let i = 0; i < d; i++) out[o + i] = E[tok * d + i] * s + this.pos[p * d + i];
  }

  // N = LayerNorm p of the first B rows of X
  norm(p, B) {
    const { X, N } = this.k.buf;
    this.k.ln(X, this.W[p + ".weight"], this.W[p + ".bias"], N, B, this.c.d);
  }

  // feed-forward block of layer p on the first B rows of X, added to X
  ffn(p, B) {
    const { d, f } = this.c, W = this.W, k = this.k, { X, N, H } = k.buf;
    this.norm(p + "ln3", B);
    k.lin(W[p + "fc1.weight"], W[p + "fc1.bias"], N, H, B, d, f, false);
    k.gelu(H, B * f);
    k.lin(W[p + "fc2.weight"], W[p + "fc2.bias"], H, X, B, f, d, true);
  }

  // all S positions go through each layer together, as the beam does in step
  encode(toks) {
    const { d, h, le, ld } = this.c, W = this.W, k = this.k, { X, N, A, H } = k.buf, S = toks.length;
    toks.forEach((t, s) => this.embed("se.weight", t, s + 2, X, s * d));
    // keys and values out of the rows of H (w wide, keys at column c and values right after)
    const split = (w, c) => {
      const K = new Float32Array(S * d), V = new Float32Array(S * d);
      for (let s = 0; s < S; s++) {
        K.set(H.subarray(s * w + c, s * w + c + d), s * d); V.set(H.subarray(s * w + c + d, s * w + c + 2 * d), s * d);
      }
      return [K, V];
    };
    for (let l = 0; l < le; l++) {
      const p = `enc.${l}.`;
      this.norm(p + "ln1", S);
      k.lin(W[p + "qkv.weight"], W[p + "qkv.bias"], N, H, S, d, 3 * d, false);
      const [K, V] = split(3 * d, d);
      for (let s = 0; s < S; s++) A.set(attend(H.subarray(s * 3 * d, s * 3 * d + d), K, V, S, d, h), s * d);
      k.lin(W[p + "o.weight"], W[p + "o.bias"], A, X, S, d, d, true);
      this.ffn(p, S);
    }
    this.norm("eln", S);
    // cross-attention keys and values, once per decoder layer
    const cross = [];
    for (let l = 0; l < ld; l++) {
      k.lin(W[`dec.${l}.kv2.weight`], W[`dec.${l}.kv2.bias`], N, H, S, d, 2 * d, false);
      cross.push(split(2 * d, 0));
    }
    return { cross, S };
  }

  // one decoder step for all B live hypotheses at once (toks[r] is hypothesis r's last token, caches[r] its cache; all
  // have length L): appends each token's self-attention keys/values to its cache, returns one log-prob array per
  // hypothesis. The Linear layers run on all B rows together, so each weight row is read once per step, not B times
  step(toks, L, caches, enc) {
    const { d, h, ld } = this.c, W = this.W, k = this.k, { X, N, A, Q, H } = k.buf, B = toks.length, V = this.V;
    toks.forEach((t, r) => this.embed("te.weight", t, L + 1, X, r * d));
    for (let l = 0; l < ld; l++) {
      const p = `dec.${l}.`;
      this.norm(p + "ln1", B);
      k.lin(W[p + "qkv.weight"], W[p + "qkv.bias"], N, H, B, d, 3 * d, false);
      for (let r = 0; r < B; r++) {
        const c = caches[r][l], o = r * 3 * d;
        c.K.set(H.subarray(o + d, o + 2 * d), (L - 1) * d); c.V.set(H.subarray(o + 2 * d, o + 3 * d), (L - 1) * d);
        A.set(attend(H.subarray(o, o + d), c.K, c.V, L, d, h), r * d);
      }
      k.lin(W[p + "o.weight"], W[p + "o.bias"], A, X, B, d, d, true);
      this.norm(p + "ln2", B);
      k.lin(W[p + "q2.weight"], W[p + "q2.bias"], N, Q, B, d, d, false);
      for (let r = 0; r < B; r++) A.set(attend(Q.subarray(r * d, (r + 1) * d), enc.cross[l][0], enc.cross[l][1], enc.S, d, h), r * d);
      k.lin(W[p + "o2.weight"], W[p + "o2.bias"], A, X, B, d, d, true);
      this.ffn(p, B);
    }
    // output layer (tied to the target embedding): logits for all rows, then log-softmax per row
    this.norm("dln", B);
    k.lin(W["te.weight"], null, N, H, B, d, V, false);
    const lgs = toks.map((_, r) => new Float64Array(H.subarray(r * V, (r + 1) * V)));
    for (const lg of lgs) {
      let mx = -Infinity; for (let v = 0; v < V; v++) if (lg[v] > mx) mx = lg[v];
      let sum = 0; for (let v = 0; v < V; v++) sum += Math.exp(lg[v] - mx);
      const lz = mx + Math.log(sum);
      for (let v = 0; v < V; v++) lg[v] -= lz;
    }
    return lgs;
  }

  // beam search, same rules as the reference: returns the finished hypotheses [{text, score}], best model score first
  nbest(word, beam = 4, maxLen = 40) {
    const { d, ld } = this.c, V = this.V, K = beam, EOS = 2, PAD = 1;
    if (K > CAP) throw new RangeError(`beam ${K} is more than ${CAP}`);
    const toks = [...word.slice(0, 61)].map((ch) => this.sidx.get(ch) ?? 3); toks.push(EOS);
    const enc = this.encode(toks);
    const newCache = () => Array.from({ length: ld }, () => ({ K: new Float32Array((maxLen + 1) * d), V: new Float32Array((maxLen + 1) * d) }));
    // only the first n floats (the rows filled so far) are copied; the next step writes the row after them
    const copyCache = (c, n) => c.map((x) => {
      const y = { K: new Float32Array(x.K.length), V: new Float32Array(x.V.length) };
      y.K.set(x.K.subarray(0, n)); y.V.set(x.V.subarray(0, n)); return y;
    });
    let hyps = [{ seq: [EOS], score: 0, cache: newCache() }];
    for (let k = 1; k < K; k++) hyps.push({ seq: [EOS], score: -Infinity, cache: null });
    const fin = [], topS = new Float64Array(2 * K), topI = new Int32Array(2 * K);
    for (let step = 0; step < maxLen; step++) {
      const L = hyps[0].seq.length, live = [];
      for (let r = 0; r < K; r++) if (hyps[r].score !== -Infinity) live.push(r);
      const lps = this.step(live.map((r) => hyps[r].seq[L - 1]), L, live.map((r) => hyps[r].cache), enc);
      // top 2K by score, kept sorted as the candidates arrive in flat-index order; an equal score goes after the ones
      // already there, so equal scores keep the lower flat index first (as torch.topk does in practice)
      let n = 0;
      for (let j = 0; j < live.length; j++) {
        const r = live[j], lp = lps[j], sc = hyps[r].score;
        lp[PAD] = -Infinity; lp[0] = -Infinity; lp[3] = -Infinity;
        if (step === 0) lp[EOS] = -Infinity;
        for (let v = 0; v < V; v++) {
          if (!(lp[v] > -Infinity)) continue;
          const s = sc + lp[v];
          if (n === 2 * K && !(s > topS[n - 1])) continue;
          let q = n < 2 * K ? n++ : n - 1;
          while (q > 0 && topS[q - 1] < s) { topS[q] = topS[q - 1]; topI[q] = topI[q - 1]; q--; }
          topS[q] = s; topI[q] = r * V + v;
        }
      }
      const next = [];
      for (let j = 0; j < n; j++) {
        const sc = topS[j], idx = topI[j], r = (idx / V) | 0, t = idx % V;
        if (t === EOS) {
          if (j < K && fin.length < K) fin.push({ tokens: hyps[r].seq.slice(1), score: sc / L });
          continue;
        }
        if (next.length < K) next.push({ r, t, sc });
      }
      if (fin.length >= K || !next.length) break;
      const used = new Set();
      hyps = next.map(({ r, t, sc }) => {
        const c = used.has(r) ? copyCache(hyps[r].cache, L * d) : hyps[r].cache; used.add(r);
        return { seq: [...hyps[r].seq, t], score: sc, cache: c };
      });
      while (hyps.length < K) hyps.push({ seq: [...hyps[0].seq.slice(0, -1), PAD], score: -Infinity, cache: null });
    }
    fin.sort((a, b) => b.score - a.score);
    return fin.map((h) => ({ text: norm(h.tokens.filter((t) => t > 3).map((t) => this.tgt[t]).join("")), score: h.score }));
  }

  // best Devanagari spelling for one romanised word, plus the ranked candidates. The results for the last cacheSize
  // distinct words are kept (set tr.cacheSize = 0 to turn this off), so a repeated word is a lookup; the returned
  // object may be shared with earlier calls, so treat it as read-only
  word(w) {
    const src = w.trim().toLowerCase();
    if (!src) return { best: "", nbest: [], candidates: [] };
    const hit = this.cache.get(src);
    if (hit) { this.cache.delete(src); this.cache.set(src, hit); return hit; }   // most recently used goes last
    const res = this.rank(src);
    if (this.cacheSize > 0) {
      this.cache.set(src, res);
      while (this.cache.size > this.cacheSize) this.cache.delete(this.cache.keys().next().value);
    }
    return res;
  }

  rank(src) {
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
