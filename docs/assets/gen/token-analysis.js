import { writeFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outDir = join(__dirname, '..');

const W = 640;
const H = 220;
const margin = { top: 32, right: 24, bottom: 44, left: 80 };
const w = W - margin.left - margin.right;
const h = H - margin.top - margin.bottom;

const budget = 8000;

const data = [
  { label: 'Croqtile', tokens: 500, color: '#2a9d8f' },
  { label: 'CUDA+CuTe', tokens: 2200, color: '#457b9d' },
  { label: 'CUTLASS', tokens: 4000, color: '#264653' },
];

const barH = 28;
const gap = 14;
const totalH = data.length * barH + (data.length - 1) * gap;

function sx(t) { return (w * t) / budget; }

let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" font-family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif">\n`;
svg += `<defs>\n`;
data.forEach((d, i) => {
  svg += `  <linearGradient id="grad${i}" x1="0%" y1="0%" x2="100%" y2="0%">\n`;
  svg += `    <stop offset="0%" stop-color="${d.color}" stop-opacity="0.95"/>\n`;
  svg += `    <stop offset="100%" stop-color="${d.color}" stop-opacity="0.7"/>\n`;
  svg += `  </linearGradient>\n`;
});
svg += `</defs>\n`;
svg += `<rect width="${W}" height="${H}" fill="#fafbfc" rx="8"/>\n`;
svg += `<g transform="translate(${margin.left},${margin.top})">\n`;

const startY = (h - totalH) / 2;

data.forEach((d, i) => {
  const y = startY + i * (barH + gap);
  const bw = sx(d.tokens);

  svg += `<rect x="0" y="${y}" width="${bw}" height="${barH}" fill="url(#grad${i})" rx="4"/>\n`;

  svg += `<text x="-8" y="${y + barH / 2 + 5}" text-anchor="end" font-size="12" font-weight="500" fill="#24292f">${d.label}</text>\n`;

  const labelX = bw + 8;
  svg += `<text x="${labelX}" y="${y + barH / 2 + 5}" font-size="11" font-weight="600" fill="${d.color}">${d.tokens.toLocaleString()} tok</text>\n`;
});

const axisY = startY + totalH + 20;
svg += `<line x1="0" x2="${w}" y1="${axisY}" y2="${axisY}" stroke="#d1d5da" stroke-width="0.8"/>\n`;
for (let t = 0; t <= budget; t += 2000) {
  const x = sx(t);
  svg += `<line x1="${x}" x2="${x}" y1="${axisY - 3}" y2="${axisY + 3}" stroke="#6e7681" stroke-width="0.8"/>\n`;
  svg += `<text x="${x}" y="${axisY + 16}" text-anchor="middle" font-size="9" fill="#6e7681">${(t / 1000).toFixed(0)}K</text>\n`;
}
svg += `<text x="${w / 2}" y="${axisY + 32}" text-anchor="middle" font-size="10" fill="#57606a">tokens per kernel representation</text>\n`;

const budgetX = sx(budget);
svg += `<line x1="${budgetX}" x2="${budgetX}" y1="${startY - 8}" y2="${axisY}" stroke="#cf222e" stroke-width="1.2" stroke-dasharray="4,3" opacity="0.7"/>\n`;

svg += `</g>\n</svg>`;

writeFileSync(join(outDir, 'token-analysis.svg'), svg);
console.log('  -> token-analysis.svg written to docs/assets/');
