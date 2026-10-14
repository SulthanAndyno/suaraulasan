// Port FeatureUnion(TfidfVectorizer kata 1-2 + char_wb 2-5, sublinear_tf, l2) + LogisticRegression dari sklearn.
// ponytail: tokenisasi kata meniru token_pattern default (\w{2,}); beda kecil di karakter Unicode langka.
function block(vocab, grams) {
  const tf = {};
  for (const g of grams) if (g in vocab) tf[g] = (tf[g] || 0) + 1;
  const w = Object.entries(tf).map(([g, n]) => [g, (1 + Math.log(n)) * vocab[g][0]]);
  const norm = Math.hypot(...w.map(([, v]) => v)) || 1;
  return w.map(([g, v]) => [g, (v / norm) * vocab[g][1]]);
}

function charGrams(text) {
  const out = [];
  for (const word of text.split(/\s+/).filter(Boolean)) {
    const w = Array.from(" " + word + " ");
    for (let n = 2; n <= 5; n++) {
      let o = 0;
      out.push(w.slice(0, n).join(""));
      while (o + n < w.length) out.push(w.slice(++o, o + n).join(""));
      if (o === 0) break; // kata pendek dihitung sekali, sama seperti sklearn
    }
  }
  return out;
}

function predict(model, text) {
  const lower = text.toLowerCase();
  const toks = lower.match(/[\p{L}\p{N}_]{2,}/gu) || [];
  const terms = block(model.word, toks.concat(toks.slice(1).map((t, i) => toks[i] + " " + t)));
  const chars = block(model.char, charGrams(lower));
  const score = [...terms, ...chars].reduce((s, [, c]) => s + c, model.b);
  // terms: kontribusi per kata untuk penjelasan di UI (n-gram karakter tidak ditampilkan)
  return { label: score > 0 ? "positif" : "negatif", prob: 1 / (1 + Math.exp(-score)), score, terms };
}
if (typeof module !== "undefined") module.exports = predict;
