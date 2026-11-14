![Lekhani · Udgam Labs](./assets/banner.png)

**UDGAM LABS** · Transliterator · 12 languages, under 1 MB each, runs in the browser

# Lekhani

Lekhani turns words typed in Latin letters into their own script: `kaise ho` becomes कैसे हो in Hindi, `vanakkam`
becomes வணக்கம் in Tamil. It covers twelve languages: Hindi, Tamil, Marathi, Bengali, Urdu, Kannada, Telugu, Malayalam,
Gujarati, Punjabi, Odia and Assamese. Each language is one small model (0.8M or 1.2M parameters at 4 bits) and a list of common words,
and everything a page loads for one language, runner included, is under 1 MB. It runs in a web page on your own
device, so nothing you type is sent anywhere.

On the Aksharantar test it gets the first guess right on 50% (Odia) to 81% (Telugu) of words, and it is ahead of
IndicXlit's model on its own in all twelve languages, by 4.2 to 14.8 points. IndicXlit's full Python package, which
re-ranks with word lists of millions of words held in memory, is level with Lekhani or ahead in several languages.

*Lekhani* (लेखनी) is Hindi for "pen".

[Try it in your browser](https://huggingface.co/spaces/udgamlabs/Lekhani) · [Model card](https://huggingface.co/udgamlabs/Lekhani) · [udgamlabs.com](https://udgamlabs.com/models/lekhani)

| | |
|---|---|
| Developer | Udgam Labs |
| Models | One per language: encoder-decoder transformer, width 128, 2 + 2 layers (0.81M parameters) or 3 + 3 layers (1.2M), 4-bit weights, trained from random weights |
| Files | `files/`: one runner, `translit.js`, and four files per language (Hindi at the root of `files/`, the others in a folder each); 0.71 to 1.0 MB per language in all (gzip) |
| Input / output | Latin letters → the language's script, one word at a time; everything that is not Latin letters is kept as it is |
| Runtime | JavaScript (ES module) with WebAssembly SIMD where the browser has it, plain JavaScript otherwise: any modern browser, or Node 18+. No server, no WebGPU, no dependencies |
| Licence | Apache-2.0 (models and runner); CC0 (word lists) |

## Get started

**In a web page.** Serve the repository from any static host and open the example, which has a language picker:

```bash
git clone https://github.com/udgamlabs/lekhani && cd lekhani
python3 -m http.server 8000      # or any static server
# open http://localhost:8000/examples/browser/
```

Or add it to your own page: copy `files/translit.js` and your language's four files next to it.

| Language | Folder in `files/` | Script |
|---|---|---|
| Hindi | (root) | Devanagari |
| Tamil | `tamil/` | Tamil |
| Marathi | `marathi/` | Devanagari |
| Bengali | `bengali/` | Bengali |
| Urdu | `urdu/` | Urdu (Arabic script, right to left) |
| Kannada | `kannada/` | Kannada |
| Telugu | `telugu/` | Telugu |
| Malayalam | `malayalam/` | Malayalam |
| Gujarati | `gujarati/` | Gujarati |
| Punjabi | `punjabi/` | Gurmukhi |
| Odia | `odia/` | Odia |
| Assamese | `assamese/` | Bengali script, Assamese letters (ৰ, ৱ) |

```html
<input id="in" placeholder="vanakkam">
<p id="out" lang="ta"></p>
<script type="module">
  import { Transliterator } from "./translit.js";
  const dir = "tamil/";   // "" for Hindi
  const tr = await Transliterator.load({
    model: dir + "model.q4.gz", json: dir + "model.json.gz", words: dir + "words.bin.gz", rerank: dir + "rerank.json",
  });
  const box = document.getElementById("in");
  box.addEventListener("input", () => {
    document.getElementById("out").textContent = tr.text(box.value);
  });
</script>
```

**In Node 18 or newer:**

```bash
node examples/node/transliterate.mjs "aaj office mein meeting hai"     # Hindi: आज ऑफिस में मीटिंग है
node examples/node/transliterate.mjs --lang tamil "vanakkam"
```

- `tr.text(s)` writes every run of Latin letters in the language's script and keeps everything else (spaces, digits,
  punctuation, text already in the script) as it is.
- `tr.word(w)` returns `best` and `candidates`: the 8 spellings the model considered, best first, each with `inList`
  (whether it is one of the listed common words). Show the runners-up as suggestions if you build a keyboard.
- Nothing per language goes in the call: each folder's `rerank.json` tells the runner which script its word list uses.
  For Urdu, set `dir="rtl"` on the element that shows the output.
- `translit.js` is a minified build; `translit.src.js` is the same code, readable, with comments.
- Load several languages on one page by calling `Transliterator.load` once per language.

Check the files with `cd files && shasum -a 256 -c ../SHA256SUMS`. The same files are on
[Hugging Face](https://huggingface.co/udgamlabs/Lekhani).

## Results

**Aksharantar test** (AI4Bharat), one language at a time: every romanised word in the test, top-1 exact match against
any reference spelling, the same words and the same scorer for every system. 95% intervals from 2,000 bootstrap
resamples, paired for gaps. Each language's settings were chosen on the validation set and the test was scored once.
Measured by us between 28 October and 4 November 2026, and re-scored with the 1.3.0 runner on 11 November: the same
first guess as the scored predictions on every test word, except 11 Marathi words that the cut-short-word fix changes
(one more right; Marathi is 70.9% either way). Odia and Assamese were trained, scored and re-scored with the same
runner on 13 November: the same first guess on every test word.

| Language | Test words | **Lekhani**, first guess right | IndicXlit's model on its own | Gap (paired 95% interval) | IndicXlit's full package | Download, gzip bytes |
|---|---:|---:|---:|---:|---:|---:|
| Hindi | 9,991 | **65.9%** (64.9 to 66.8) | 58.9% | **+7.0** (+6.1 to +8.0) | 68.7% | 872,565 |
| Tamil | 11,405 | **74.2%** (73.4 to 75.1) | 64.2% | **+10.0** (+9.3 to +10.7) | 73.9% | 708,999 |
| Marathi | 12,059 | **70.9%** (70.2 to 71.8) | 63.2% | **+7.8** (+7.0 to +8.6) | 72.7% | 958,520 |
| Bengali | 13,367 | **70.2%** (69.4 to 71.0) | 55.4% | **+14.8** (+14.0 to +15.6) | 67.4% | 999,891 |
| Urdu | 14,617 | **57.6%** (56.8 to 58.4) | 44.4% | **+13.2** (+12.4 to +14.0) | 48.2% | 908,841 |
| Kannada | 11,165 | **79.4%** (78.7 to 80.2) | 72.7% | **+6.7** (+6.0 to +7.4) | 80.9% | 981,214 |
| Telugu | 9,936 | **81.2%** (80.5 to 81.9) | 72.6% | **+8.5** (+7.7 to +9.3) | 81.1% | 964,906 |
| Malayalam | 12,343 | **67.1%** (66.3 to 68.0) | 59.5% | **+7.6** (+6.9 to +8.4) | 69.2% | 956,061 |
| Gujarati | 17,402 | **72.8%** (72.1 to 73.5) | 61.2% | **+11.6** (+10.9 to +12.3) | 72.7% | 983,327 |
| Punjabi | 10,949 | **55.7%** (54.8 to 56.6) | 44.5% | **+11.2** (+10.3 to +12.0) | 55.2% | 932,530 |
| Odia | 4,201 | **49.9%** (48.4 to 51.4) | 45.6% | **+4.2** (+3.0 to +5.5) | 52.0% | 978,975 |
| Assamese | 5,476 | **58.6%** (57.3 to 59.9) | 53.1% | **+5.5** (+4.4 to +6.7) | 60.4% | 982,294 |

- **IndicXlit's model on its own** (11.5M parameters, beam 4, its default) is the comparison we set for every language
  before training. Lekhani is ahead of it in all twelve, and the paired intervals are clear of zero.
- **IndicXlit's full package** adds a word list of millions of words for each language (for Hindi, 4.2 million words,
  358 MB in memory) and re-ranks with it. Against the package Lekhani is ahead in Bengali (+2.8) and
  Urdu (+9.4), level in Tamil, Telugu, Gujarati and Punjabi (intervals that include zero), and behind in Hindi (-2.8),
  Marathi (-1.8), Kannada (-1.4), Malayalam (-2.1), Odia (-2.1) and Assamese (-1.8).
- **Download** is every file a page loads for that language, as gzip bytes: the model, its word list, the two small
  settings files and the runner (6,625 bytes). Each is under 1,000,000. Hindi is 0.87 MB, the others 0.71 to 1.0 MB.
- In Hindi, our own 122-word chat set (everyday chat words, English words used in Hindi, brands and names, checked
  word by word by a native speaker): Lekhani 113, IndicXlit's package 112, IndicXlit's model 102.

## Where it falls short

- **Names are weak, except in Hindi.** In the other eleven languages, Aksharantar's name rows (AK-NEF and AK-NEI)
  get 25% to 60% right at top-1, against 58% to 88% on its Dakshina rows (words as people type them) and about 70% on
  common words in Odia and Assamese. Odia and Malayalam names are the weakest (Odia 25% and 30%, Malayalam 30% and
  37%). In Odia and Assamese, names are 1 to 4 points below IndicXlit's model; the whole gain there is on common words.
- **Odia, Punjabi, Urdu and Assamese are weak in absolute terms.** Odia gets 50% of test words right on the first
  guess, Punjabi 56%, Urdu 58% and Assamese 59%, even though each is ahead of IndicXlit's model. Odia's test is about
  half names, which is part of why it is the lowest.
- **Odia and Assamese have no real-typing test.** Dakshina has neither, so their scores cover the test's
  dictionary-style romanisation only, not words as people type them in chat.
- **IndicXlit's full package, with its multi-million-word lists, is level or ahead in ten of the twelve languages** on
  this test (ahead in Hindi, Marathi, Kannada, Malayalam, Odia and Assamese; level in Tamil, Telugu, Gujarati and Punjabi). On our
  real-typing check (4,000 words per language from Dakshina's romanised sentences, nine languages) it is ahead in six.
  If you can hold its word lists in memory, it is the more accurate choice there.
- **No native-speaker review of the misses yet**, except for Hindi's chat set. Some misses may be acceptable spellings
  that are missing from the test's answer lists; some may be real errors a native reader would catch at once.
- **No sentence context.** It writes one word at a time, so words spelled alike in Latin letters get the same output:
  in Hindi, `main` ("I") and `mein` ("in") both become में.
- **Some common words come out wrong.** In Hindi, `kal` → काल (कल is its second choice) and `delhi` → देल्ही (should
  be दिल्ली).
- **One direction only**: Latin letters to each script. It does not translate.
- **Speed was measured on one Mac mini** (2 to 4 ms a word in Node with WebAssembly SIMD). Phones and older laptops are
  not measured yet.

## How it was trained

Each language has its own small encoder-decoder transformer (2 + 2 layers, 0.81M parameters, or 3 + 3 layers, 1.2M
parameters; the size was chosen on validation), trained from random weights and stored at 4 bits. The training pairs
of romanised and native words come from Aksharantar (its CC0 and CC-BY parts; not its share-alike part) and from a
teacher: IndicXlit wrote the ways people type common words from IndicCorp v2, and each language's spellings of English
words. At run time the model proposes 8 spellings and a list of 125,000 to 300,000 common words, built from IndicCorp v2
word counts, moves a common real word up when the model is unsure. Words that are not in the list still work from the
model alone. No test or validation word was in the training pairs.

## Intended use and limitations

For typing these languages in Latin letters on websites, apps and tools: search boxes, forms, chat, notes, a
keyboard's suggestions. It works one word at a time and does not translate. Show users the runners-up so they can fix
a wrong word, and keep their own spelling if they do. It is not meant for names on legal or official documents without
a human check.

## Changes

- **Eight more languages (14 Nov 2026):** Marathi, Bengali, Urdu, Malayalam, Gujarati, Punjabi, Odia and Assamese, each in
  its own folder, so twelve in all. The runner (1.3.0) and the Hindi, Tamil, Telugu and Kannada files are unchanged.
  The 1.2.0 fix for words cut short (a bare virama at the end) applies to Devanagari, so to Hindi and Marathi; in
  Marathi it changes 11 of 12,059 test words, one more of them right.
- **Kannada (13 Nov 2026):** Kannada, in `kannada/`. The runner (1.3.0) and the Hindi, Tamil and Telugu files are
  unchanged.
- **Tamil (12 Nov 2026):** Tamil, in `tamil/`. The runner (1.3.0) and the Hindi and Telugu files are unchanged.
- **1.3.0 (11 Nov 2026):** Telugu, in its own folder. One runner serves both languages: a language's `rerank.json` now
  names the first letter of its script's block, so its word list can pack any script. Hindi is unchanged:
  the same files, and the same best word, the same 8 candidates and the same scores on all 9,991 Aksharantar Hindi test
  words as 1.2.0. `translit.js` is now a minified build (6.6 KB gzip, was 10.9 KB), with the readable source as
  `translit.src.js`.
- **1.2.0 (4 Nov 2026):** common Hindi words cut short are fixed (`acha` and `achha` now give अच्छा). Test score
  unchanged at 65.9%; the Hinglish chat set rises from 112 to 113.
- **1.1.0 (2 Nov 2026):** the runner is about 10 times faster (batched beam search, WebAssembly SIMD kernels with a
  plain-JavaScript fallback, a GELU table, leaner beam search and a 2,000-word cache), contributed by Manit Roy
  ([GitHub #1](https://github.com/udgamlabs/lekhani/pull/1)).
- **1.0.0 (30 Oct 2026):** first release, Hindi.

## Licence

Model weights and the runner (`translit.js`, `translit.src.js`): Apache-2.0. The word lists (`words.bin.gz`): CC0. See
`NOTICE` for the credits: Aksharantar (AI4Bharat; CC-BY hand-collected part, CC0 mined parts), IndicXlit (AI4Bharat,
MIT) as the teacher, IndicCorp v2 (AI4Bharat, CC0) for the word counts, and the CMU Pronouncing Dictionary
(BSD-style) for English words.

## Citation

```bibtex
@misc{udgamlabs2026lekhani,
  title  = {Lekhani: in-browser transliteration, under 1 MB per language},
  author = {Prasad, Jitesh K},
  year   = {2026},
  howpublished = {Udgam Labs},
  url    = {https://huggingface.co/udgamlabs/Lekhani}
}
```

Please also cite Aksharantar (Madhani et al., 2022, arXiv:2205.03018), whose data and IndicXlit model made this
possible.

---

**About Udgam Labs.** Udgam Labs is an independent AI research lab. We work from first principles: before we build, we understand why things work, whether we are training a model, fine-tuning one or applying AI to a new problem. Udgam (उद्गम) is Sanskrit for origin. [udgamlabs.com](https://udgamlabs.com)
