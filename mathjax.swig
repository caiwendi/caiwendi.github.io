{% if theme.mathjax.enable %}
<!-- MathJax Start -->
<!-- MathJax documentation: http://docs.mathjax.org/en/latest/index.html -->
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {  // tex2jax preprocessor
      inlineMath: [ ['$','$'] ],  // delimiters for in-line math
      displayMath: [ ['$$','$$'] ],  // delimiters for displayed equations
      processEscapes: true,  // enable \$ to represent a single dollar sign
      skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']  // MathJax will not process contents inside these tags 
    },
    TeX: {  // TeX/LaTeX input processor
      equationNumbers: { autoNumber: "all" },  // only number those equations in specific AMSmath environments
      //extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"]  // introduce AMS extensions and suppress generating error messages 
    },
    "HTML-CSS": {  // HTML-CSS output processor (this is the default output of MathJax)
      scale: 110,  // The scaling factor of math with respect to the surrounding text
      linebreaks: { automatic: true } // automatically breaks the line if necessary
    },
    SVG: {  // SVG output processor
      scale: 110,  // The scaling factor of math with respect to the surrounding text
      linebreaks: { automatic: true } // automatically breaks the line if necessary
    },
    menuSettings: {  // settings for the mathematics contextual menu
      zoom: "false"  // set equation zooming to be triggered by a single mouse click
    }
  });
 
  MathJax.Hub.Queue(function() { // Fix <code> tags after MathJax finishes running, this is a hack to overcome a shortcoming of Markdown
      var all = MathJax.Hub.getAllJax(), i;
      for(i=0; i < all.length; i += 1) {
          all[i].SourceElement().parentNode.className += ' has-jax';
      }
  });
</script>
<script type="text/javascript"
   //src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML">  // link to the MathJax CDN
</script>
<!-- MathJax End -->

{% endif %}