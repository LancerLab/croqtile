'use strict';

const EXAMPLES = {
  hello: `// Hello World in Choreo
__co__ void hello() {
  println("Hello from Choreo!");
}

int main() {
  hello();
  std::cout << "Run Complete." << std::endl;
}
`,
  dma: `// DMA Copy Example
__co__ auto dma_copy(f32 [4, 64] input) {
  shared f32 [4, 64] buf;
  dma(input, buf);
  return buf;
}

int main() {
  std::vector<float> data(4 * 64, 1.0f);
  auto result = dma_copy(data.data());
  std::cout << "Run Complete." << std::endl;
}
`,
  matmul: `// Simple Matrix Multiply
__co__ auto matmul(f16 [16, 16] A, f16 [16, 16] B) {
  shared f16 [16, 16] sA;
  shared f16 [16, 16] sB;
  dma(A, sA);
  dma(B, sB);
  local f32 [16, 16] C;
  mma(sA, sB, C);
  return C;
}
`
};

let editor = null;
let choreoWorker = null;
let moduleReady = false;
let gpuServerUrl = null;  // Set when GPU server is detected

function initMonaco() {
  require.config({
    paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs' }
  });

  require(['vs/editor/editor.main'], function () {
    monaco.languages.register({ id: 'choreo' });

    monaco.languages.setMonarchTokensProvider('choreo', {
      keywords: [
        'auto', 'bool', 'break', 'case', 'char', 'class', 'const',
        'continue', 'default', 'do', 'double', 'else', 'enum', 'extern',
        'float', 'for', 'goto', 'if', 'inline', 'int', 'long', 'namespace',
        'new', 'operator', 'private', 'protected', 'public', 'register',
        'return', 'short', 'signed', 'sizeof', 'static', 'struct',
        'switch', 'template', 'this', 'throw', 'try', 'typedef', 'typename',
        'union', 'unsigned', 'using', 'virtual', 'void', 'volatile', 'while',
        'true', 'false', 'nullptr', 'std'
      ],
      choreoKeywords: [
        '__co__', '__cok__', 'dma', 'tma', 'mma', 'exec',
        'shared', 'local', 'global', 'inthreads', 'ingroups', 'inblocks',
        'foreach', 'pipeline', 'event', 'signal', 'wait', 'arrive',
        'println', 'print', 'assert_true'
      ],
      typeKeywords: [
        'f16', 'f32', 'f64', 'bf16', 'i8', 'i16', 'i32', 'i64',
        'u8', 'u16', 'u32', 'u64', 'half', 'size_t',
        'f8_e4m3', 'f8_e5m2'
      ],
      operators: [
        '=', '>', '<', '!', '~', '?', ':', '==', '<=', '>=', '!=',
        '&&', '||', '++', '--', '+', '-', '*', '/', '&', '|', '^', '%',
        '<<', '>>', '+=', '-=', '*=', '/=', '&=', '|=', '^=',
        '%=', '<<=', '>>='
      ],
      symbols: /[=><!~?:&|+\-*/^%]+/,
      escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

      tokenizer: {
        root: [
          [/[a-z_$][\w$]*/, {
            cases: {
              '@choreoKeywords': 'keyword.choreo',
              '@typeKeywords': 'type',
              '@keywords': 'keyword',
              '@default': 'identifier'
            }
          }],
          [/__\w+__/, 'keyword.choreo'],
          [/[A-Z][\w$]*/, 'type.identifier'],

          { include: '@whitespace' },

          [/[{}()[\]]/, '@brackets'],
          [/[<>](?!@symbols)/, '@brackets'],
          [/@symbols/, {
            cases: {
              '@operators': 'operator',
              '@default': ''
            }
          }],

          [/\d*\.\d+([eE][-+]?\d+)?[fFdD]?/, 'number.float'],
          [/0[xX][0-9a-fA-F]+/, 'number.hex'],
          [/\d+[lL]?/, 'number'],

          [/[;,.]/, 'delimiter'],
          [/"([^"\\]|\\.)*$/, 'string.invalid'],
          [/"/, { token: 'string.quote', bracket: '@open', next: '@string' }],
          [/'[^\\']'/, 'string'],
          [/'/, 'string.invalid']
        ],
        string: [
          [/[^\\"]+/, 'string'],
          [/@escapes/, 'string.escape'],
          [/\\./, 'string.escape.invalid'],
          [/"/, { token: 'string.quote', bracket: '@close', next: '@pop' }]
        ],
        whitespace: [
          [/[ \t\r\n]+/, 'white'],
          [/\/\*/, 'comment', '@comment'],
          [/\/\/.*$/, 'comment']
        ],
        comment: [
          [/[^/*]+/, 'comment'],
          [/\*\//, 'comment', '@pop'],
          [/[/*]/, 'comment']
        ]
      }
    });

    monaco.editor.defineTheme('choreo-dark', {
      base: 'vs-dark',
      inherit: true,
      rules: [
        { token: 'keyword.choreo', foreground: 'f38ba8', fontStyle: 'bold' },
        { token: 'type', foreground: 'fab387' },
        { token: 'type.identifier', foreground: 'a6e3a1' },
        { token: 'keyword', foreground: '89b4fa' },
        { token: 'comment', foreground: '6c7086', fontStyle: 'italic' },
        { token: 'string', foreground: 'a6e3a1' },
        { token: 'number', foreground: 'fab387' },
        { token: 'operator', foreground: '89dceb' }
      ],
      colors: {
        'editor.background': '#1e1e2e',
        'editor.foreground': '#cdd6f4',
        'editorLineNumber.foreground': '#6c7086',
        'editorLineNumber.activeForeground': '#cdd6f4',
        'editor.selectionBackground': '#45475a',
        'editor.lineHighlightBackground': '#252537'
      }
    });

    editor = monaco.editor.create(document.getElementById('editor-container'), {
      value: EXAMPLES.hello,
      language: 'choreo',
      theme: 'choreo-dark',
      fontSize: 14,
      fontFamily: "'JetBrains Mono', 'Fira Code', Consolas, monospace",
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      automaticLayout: true,
      tabSize: 2,
      lineNumbers: 'on',
      renderLineHighlight: 'line',
      padding: { top: 8 }
    });

    initWorker();
    initUI();
  });
}

function initWorker() {
  choreoWorker = new Worker('choreo-worker.js');

  choreoWorker.onmessage = function (e) {
    const { type, data } = e.data;

    switch (type) {
      case 'ready':
        moduleReady = true;
        setStatus('Ready', 'ready');
        document.getElementById('version-badge').textContent = 'v' + (data.version || '?');
        enableButtons(true);
        break;

      case 'compile-result':
        showResult(data);
        enableButtons(true);
        setStatus('Ready', 'ready');
        break;

      case 'error':
        showError(data.message);
        enableButtons(true);
        setStatus('Error', 'error');
        break;
    }
  };

  choreoWorker.onerror = function (e) {
    setStatus('Worker error: ' + e.message, 'error');
    enableButtons(true);
  };
}

function initUI() {
  document.getElementById('btn-compile').addEventListener('click', doCompile);
  document.getElementById('btn-run').addEventListener('click', doMockRun);
  document.getElementById('btn-gpu').addEventListener('click', doGPURun);
  document.getElementById('btn-ast').addEventListener('click', doDumpAST);

  // Try to detect GPU server
  detectGPUServer();

  document.getElementById('example-select').addEventListener('change', function () {
    const val = this.value;
    if (val && EXAMPLES[val]) {
      editor.setValue(EXAMPLES[val]);
      this.value = '';
    }
  });

  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', function () {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      this.classList.add('active');
      document.getElementById('tab-' + this.dataset.tab).classList.add('active');
    });
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', function (e) {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      doCompile();
    }
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'Enter') {
      e.preventDefault();
      doMockRun();
    }
  });
}

async function detectGPUServer() {
  const candidates = [
    'http://localhost:8081',
    window.location.origin.replace(/:\d+$/, ':8081')
  ];

  for (const url of candidates) {
    try {
      const resp = await fetch(url + '/api/status', { signal: AbortSignal.timeout(2000) });
      if (resp.ok) {
        const data = await resp.json();
        if (data.status === 'ok') {
          gpuServerUrl = url;
          const btn = document.getElementById('btn-gpu');
          btn.disabled = false;
          const gpuName = data.gpus?.[0]?.name || 'GPU';
          btn.title = `Run on ${gpuName} (${data.arch})`;
          return;
        }
      }
    } catch (e) {
      // Server not available
    }
  }
}

async function doGPURun() {
  if (!gpuServerUrl) return;
  const source = editor.getValue();
  enableButtons(false);
  setStatus('Running on GPU...', 'working');

  try {
    const resp = await fetch(gpuServerUrl + '/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source })
    });
    const data = await resp.json();
    showResult(data);
  } catch (e) {
    showError('GPU server error: ' + e.message);
  } finally {
    enableButtons(true);
    setStatus('Ready', 'ready');
  }
}

function doCompile() {
  if (!moduleReady) return;
  const source = editor.getValue();
  const target = document.getElementById('target-select').value;
  enableButtons(false);
  setStatus('Compiling...', 'working');
  choreoWorker.postMessage({ type: 'compile', source, target, flags: '' });
}

function doMockRun() {
  if (!moduleReady) return;
  const source = editor.getValue();
  enableButtons(false);
  setStatus('Running interpreter (in-browser)...', 'working');
  choreoWorker.postMessage({ type: 'mockRun', source });
}

function doDumpAST() {
  if (!moduleReady) return;
  const source = editor.getValue();
  enableButtons(false);
  setStatus('Dumping AST...', 'working');
  choreoWorker.postMessage({ type: 'dumpAST', source });
}

function showResult(data) {
  document.getElementById('output-content').textContent = data.output || '(no output)';
  document.getElementById('errors-content').textContent = data.errors || '(no errors)';

  if (data.ast) {
    document.getElementById('ast-content').textContent = data.ast;
  }

  // Switch to appropriate tab
  if (data.errors && !data.success) {
    switchTab('errors');
  } else if (data.ast) {
    switchTab('ast');
  } else {
    switchTab('output');
  }
}

function showError(message) {
  document.getElementById('errors-content').textContent = message;
  switchTab('errors');
}

function switchTab(tabName) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  const tab = document.querySelector(`.tab[data-tab="${tabName}"]`);
  if (tab) tab.classList.add('active');
  const content = document.getElementById('tab-' + tabName);
  if (content) content.classList.add('active');
}

function enableButtons(enabled) {
  document.getElementById('btn-compile').disabled = !enabled;
  document.getElementById('btn-run').disabled = !enabled;
  document.getElementById('btn-ast').disabled = !enabled;
}

function setStatus(text, type) {
  const el = document.getElementById('status');
  el.textContent = text;
  el.className = type ? 'status-' + type : '';
}

// Boot
enableButtons(false);
initMonaco();
