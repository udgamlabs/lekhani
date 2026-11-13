// Lekhani in Node 18+: transliterate the text given on the command line (or a few examples) in one language.
// Run from the repository root:
//   node examples/node/transliterate.mjs "aaj office mein meeting hai"            (Hindi, the default)
//   node examples/node/transliterate.mjs --lang kannada "naale sigona"
//   node examples/node/transliterate.mjs --lang tamil "eppadi irukkeenga"
//   node examples/node/transliterate.mjs --lang telugu "meeru ela unnaru"
// Languages: hindi kannada tamil telugu
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Transliterator } from "../../files/translit.js";

const args = process.argv.slice(2);
const at = args.indexOf("--lang");
const lang = at >= 0 ? args.splice(at, 2)[1] : "hindi";
// Hindi's files are at the root of files/; every other language has its own folder
const dir = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "files", lang === "hindi" ? "" : lang);
if (!fs.existsSync(path.join(dir, "model.q4.gz"))) throw new Error(`no files for "${lang}" in ${dir}`);
const read = (name) => fs.readFileSync(path.join(dir, name));

const tr = await Transliterator.load({
  model: read("model.q4.gz"),
  json: read("model.json.gz"),
  words: read("words.bin.gz"),
  rerank: JSON.parse(read("rerank.json").toString("utf8")),
});

const examples = {
  hindi: ["kaise ho", "kya haal hai bhai", "aaj office mein meeting hai"],
  kannada: ["naale sigona", "hegiddira", "dhanyavadagalu"],
  tamil: ["eppadi irukkeenga", "romba nandri", "naan nalla irukken"],
  telugu: ["meeru ela unnaru", "nenu bagunnanu", "dhanyavadalu"],
};
for (const line of args.length ? args : examples[lang] ?? ["namaste"]) console.log(tr.text(line));
// The runners-up for one word, best first (show them as suggestions in a keyboard):
const w = (args.length ? args.join(" ") : (examples[lang] ?? ["namaste"])[0]).split(/\s+/)[0];
console.log(w, "->", tr.word(w).candidates.slice(0, 3).map((c) => c.text).join(", "));
