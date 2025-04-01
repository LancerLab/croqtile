// co.js
(function() {
  if (window.hljs) {
    console.log('Highlight.js is loaded and available.');
  } else {
    console.error('Highlight.js is not loaded.');
  }

  var hljs = window.hljs;

  // Extend the C++ language definition
  var choreo = hljs.getLanguage('cpp');

  // Add custom keywords
  choreo.keywords = {
    keyword: 'chunkat mdspan with parallel ituple by in foreach shared local global where after call wait inthreads',
    literal: 'true false null'
  };

  // Define custom modes for attributes, types, and operations
  choreo.contains.push(
    {
      className: 'attribute',
      begin: /\b(__co__|__cok__|__global__|__device__|shared|local|global|.async)\b/,
      relevance: 10
    },
    {
      className: 'type',
      begin: /\b(f32|s32|u32|f16|bf16|s16|u16|s8|u8|half|bfp16|half8|event)\b/,
      relevance: 10
    },
    {
      className: 'operation',
      begin: /\b(call|dma.copy|dma.copy|=>|select|trigger|wait|sync)\b/,
      relevance: 10
    },
  );

//  choreo.contains.concat(cpp.contains) // Include all C++ modes

  // Register the new language
  hljs.registerLanguage('choreo', function(hljs) {
    return choreo;
  });
  hljs.highlightAll();
})();
