// Port TfidfVectorizer(ngram 1-2, sublinear_tf, norm l2) + LogisticRegression dari sklearn.
// ponytail: tokenisasi meniru token_pattern default (\w{2,}); beda kecil di karakter Unicode langka.
function predict(model, text) {
  const toks = text.toLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) || [];
  const grams = toks.concat(toks.slice(1).map((t, i) => toks[i] + " " + t));
  const tf = {};
  for (const g of grams) if (g in model.vocab) tf[g] = (tf[g] || 0) + 1;
  const w = Object.entries(tf).map(([g, n]) => [g, (1 + Math.log(n)) * model.vocab[g][0]]);
  const norm = Math.hypot(...w.map(([, v]) => v)) || 1;
  const terms = w.map(([g, v]) => [g, (v / norm) * model.vocab[g][1]]);
  const score = terms.reduce((s, [, c]) => s + c, model.b);
  return { label: score > 0 ? "positif" : "negatif", prob: 1 / (1 + Math.exp(-score)), score, terms };
}
if (typeof module !== "undefined") module.exports = predict;
