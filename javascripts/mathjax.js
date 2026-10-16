// Configure MathJax so math renders both in Markdown pages ( \( ... \) ) and in notebook cells ( $ ... $ ).
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"], ["$", "$"]],
    displayMath: [["\\[", "\\]"], ["$$", "$$"]],
    processEscapes: true,
  },
  options: { ignoreHtmlClass: "no-mathjax" },
};
