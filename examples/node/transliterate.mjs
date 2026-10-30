// Lekhani in Node 18+: transliterate the words given on the command line (or a few examples).
// Run from the repository root:  node examples/node/transliterate.mjs "aaj office mein meeting hai"
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Transliterator } from "../../files/translit.js";

const files = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "files");
const read = (name) => fs.readFileSync(path.join(files, name));

const tr = await Transliterator.load({
  model: read("model.q4.gz"),
  json: read("model.json.gz"),
  words: read("words.bin.gz"),
  rerank: JSON.parse(read("rerank.json").toString("utf8")),
});

const inputs = process.argv.slice(2);
for (const line of inputs.length ? inputs : ["kaise ho", "kya haal hai bhai", "aaj office mein meeting hai"]) {
  console.log(tr.text(line));
}
// The runners-up for one word, best first (show them as suggestions in a keyboard):
console.log("main ->", tr.word("main").candidates.slice(0, 3).map((c) => c.text).join(", "));
