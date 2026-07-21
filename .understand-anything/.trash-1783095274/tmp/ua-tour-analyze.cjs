#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

function fail(msg) {
  process.stderr.write('ERROR: ' + msg + '\n');
  process.exit(1);
}

const inputPath = process.argv[2];
const outputPath = process.argv[3];

if (!inputPath || !outputPath) {
  fail('Usage: node ua-tour-analyze.js <input.json> <output.json>');
}

let raw;
try {
  raw = fs.readFileSync(inputPath, 'utf8');
} catch (e) {
  fail('Could not read input file: ' + e.message);
}

let data;
try {
  data = JSON.parse(raw);
} catch (e) {
  fail('Could not parse input JSON: ' + e.message);
}

const nodes = Array.isArray(data.nodes) ? data.nodes : [];
const allEdges = Array.isArray(data.edges) ? data.edges : [];
const layers = Array.isArray(data.layers) ? data.layers : [];

if (nodes.length === 0) {
  fail('No nodes found in input.');
}

const nodeById = new Map();
for (const n of nodes) {
  nodeById.set(n.id, n);
}

// Only keep edges where both endpoints exist among our (file-level) nodes.
// The raw edge set may reference function/class nodes that are out of scope here.
const edges = allEdges.filter((e) => nodeById.has(e.source) && nodeById.has(e.target));

// ---------- A. Fan-In Ranking ----------
const fanIn = new Map();
const fanOut = new Map();
for (const n of nodes) {
  fanIn.set(n.id, 0);
  fanOut.set(n.id, 0);
}
for (const e of edges) {
  fanOut.set(e.source, (fanOut.get(e.source) || 0) + 1);
  fanIn.set(e.target, (fanIn.get(e.target) || 0) + 1);
}

const fanInRanking = [...fanIn.entries()]
  .map(([id, count]) => ({ id, fanIn: count, name: nodeById.get(id)?.name }))
  .sort((a, b) => b.fanIn - a.fanIn)
  .slice(0, 20);

const fanOutRanking = [...fanOut.entries()]
  .map(([id, count]) => ({ id, fanOut: count, name: nodeById.get(id)?.name }))
  .sort((a, b) => b.fanOut - a.fanOut)
  .slice(0, 20);

// ---------- C. Entry Point Candidates ----------
const ENTRY_FILENAMES = new Set([
  'index.ts', 'index.js', 'main.ts', 'main.js', 'app.ts', 'app.js',
  'server.ts', 'server.js', 'mod.rs', 'main.go', 'main.py', 'main.rs',
  'manage.py', 'app.py', 'wsgi.py', 'asgi.py', 'run.py', '__main__.py',
  'Application.java', 'Main.java', 'Program.cs', 'config.ru', 'index.php',
  'App.swift', 'Application.kt', 'main.cpp', 'main.c',
]);

function pathDepth(filePath) {
  if (!filePath) return Infinity;
  return filePath.split('/').filter(Boolean).length;
}

const fanInValues = [...fanIn.values()];
const sortedFanIn = [...fanInValues].sort((a, b) => a - b);
function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  const idx = Math.floor(p * (sorted.length - 1));
  return sorted[idx];
}
const fanOutValues = [...fanOut.values()];
const sortedFanOutDesc = [...fanOutValues].sort((a, b) => b - a);
const top10PctFanOutThreshold = percentile(sortedFanOutDesc, 0.10);
const bottom25PctFanInThreshold = percentile(sortedFanIn, 0.25);

const entryScores = [];
for (const n of nodes) {
  let score = 0;
  const base = path.basename(n.filePath || n.name || '');
  const isRoot = n.filePath && !n.filePath.includes('/');
  const isOneLevelDeep = n.filePath && n.filePath.split('/').filter(Boolean).length <= 2;

  if (n.type === 'document') {
    if (base === 'README.md' && isRoot) {
      score += 5;
    } else if (base.endsWith('.md') && isRoot) {
      score += 2;
    }
  } else {
    if (ENTRY_FILENAMES.has(base)) {
      score += 3;
    }
    if (isRoot || isOneLevelDeep) {
      score += 1;
    }
    const fo = fanOut.get(n.id) || 0;
    const fi = fanIn.get(n.id) || 0;
    if (fo >= top10PctFanOutThreshold && fo > 0) {
      score += 1;
    }
    if (fi <= bottom25PctFanInThreshold) {
      score += 1;
    }
  }

  if (score > 0) {
    entryScores.push({ id: n.id, score, name: n.name, summary: n.summary });
  }
}

entryScores.sort((a, b) => b.score - a.score);
const entryPointCandidates = entryScores.slice(0, 5);

// ---------- D. BFS from top code entry point ----------
// Skip documentation nodes for BFS start (they have no imports/calls edges).
const codeEntryCandidates = entryScores.filter((c) => nodeById.get(c.id)?.type !== 'document');
const bfsStart = codeEntryCandidates.length > 0 ? codeEntryCandidates[0] : (entryScores[0] || null);

const adjacency = new Map();
for (const n of nodes) adjacency.set(n.id, []);
for (const e of edges) {
  if (e.type === 'imports' || e.type === 'calls') {
    adjacency.get(e.source).push(e.target);
  }
}

let bfsTraversal = { startNode: null, order: [], depthMap: {}, byDepth: {} };
if (bfsStart) {
  const startId = bfsStart.id;
  const visited = new Set([startId]);
  const order = [startId];
  const depthMap = { [startId]: 0 };
  const queue = [startId];
  while (queue.length > 0) {
    const cur = queue.shift();
    const depth = depthMap[cur];
    const neighbors = adjacency.get(cur) || [];
    for (const nb of neighbors) {
      if (!visited.has(nb)) {
        visited.add(nb);
        depthMap[nb] = depth + 1;
        order.push(nb);
        queue.push(nb);
      }
    }
  }
  const byDepth = {};
  for (const [id, depth] of Object.entries(depthMap)) {
    const key = String(depth);
    if (!byDepth[key]) byDepth[key] = [];
    byDepth[key].push(id);
  }
  bfsTraversal = { startNode: startId, order, depthMap, byDepth };
}

// ---------- E. Non-Code File Inventory ----------
const nonCodeFiles = {
  documentation: [],
  infrastructure: [],
  data: [],
  config: [],
};

for (const n of nodes) {
  const entry = { id: n.id, name: n.name, type: n.type, summary: n.summary };
  if (n.type === 'document') {
    nonCodeFiles.documentation.push(entry);
  } else if (n.type === 'service' || n.type === 'pipeline' || n.type === 'resource') {
    nonCodeFiles.infrastructure.push(entry);
  } else if (n.type === 'table' || n.type === 'schema' || n.type === 'endpoint') {
    nonCodeFiles.data.push(entry);
  } else if (n.type === 'config') {
    nonCodeFiles.config.push(entry);
  }
}

// ---------- F. Tightly Coupled Clusters ----------
// Build a mutual-edge graph: A->B and B->A (any edge type among filtered edges)
const edgeSet = new Set(edges.map((e) => `${e.source}=>${e.target}`));
const mutualPairs = [];
for (const e of edges) {
  const rev = `${e.target}=>${e.source}`;
  if (edgeSet.has(rev) && e.source < e.target) {
    mutualPairs.push([e.source, e.target]);
  }
}

// Union-Find to group mutual pairs into initial clusters
const parent = new Map();
function find(x) {
  if (!parent.has(x)) parent.set(x, x);
  if (parent.get(x) !== x) parent.set(x, find(parent.get(x)));
  return parent.get(x);
}
function union(a, b) {
  const ra = find(a);
  const rb = find(b);
  if (ra !== rb) parent.set(ra, rb);
}
for (const [a, b] of mutualPairs) {
  union(a, b);
}

const clusterGroups = new Map();
for (const [a, b] of mutualPairs) {
  const root = find(a);
  if (!clusterGroups.has(root)) clusterGroups.set(root, new Set());
  clusterGroups.get(root).add(a);
  clusterGroups.get(root).add(b);
}

// Expand clusters: add nodes connecting to 2+ existing cluster members, cap size at 5
function countEdgesWithinSet(nodeSet) {
  let count = 0;
  for (const e of edges) {
    if (nodeSet.has(e.source) && nodeSet.has(e.target)) count++;
  }
  return count;
}

const rawClusters = [];
for (const [, memberSet] of clusterGroups) {
  const members = new Set(memberSet);
  // try expansion
  let expanded = true;
  while (expanded && members.size < 5) {
    expanded = false;
    const connectionCount = new Map();
    for (const e of edges) {
      if (members.has(e.source) && !members.has(e.target)) {
        connectionCount.set(e.target, (connectionCount.get(e.target) || 0) + 1);
      } else if (members.has(e.target) && !members.has(e.source)) {
        connectionCount.set(e.source, (connectionCount.get(e.source) || 0) + 1);
      }
    }
    let bestCandidate = null;
    let bestCount = 0;
    for (const [cand, count] of connectionCount) {
      if (count >= 2 && count > bestCount) {
        bestCandidate = cand;
        bestCount = count;
      }
    }
    if (bestCandidate && members.size < 5) {
      members.add(bestCandidate);
      expanded = true;
    }
  }
  if (members.size >= 2) {
    rawClusters.push({
      nodes: [...members],
      edgeCount: countEdgesWithinSet(members),
    });
  }
}

rawClusters.sort((a, b) => b.edgeCount - a.edgeCount);
const clusters = rawClusters.slice(0, 10);

// ---------- G. Layer List ----------
const layerOutput = {
  count: layers.length,
  list: layers.map((l) => ({ id: l.id, name: l.name, description: l.description })),
};

// ---------- H. Node Summary Index ----------
const nodeSummaryIndex = {};
for (const n of nodes) {
  nodeSummaryIndex[n.id] = { name: n.name, type: n.type, summary: n.summary };
}

// ---------- Final Output ----------
const result = {
  scriptCompleted: true,
  entryPointCandidates,
  fanInRanking,
  fanOutRanking,
  bfsTraversal,
  nonCodeFiles,
  clusters,
  layers: layerOutput,
  nodeSummaryIndex,
  totalNodes: nodes.length,
  totalEdges: edges.length,
};

try {
  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
} catch (e) {
  fail('Could not write output file: ' + e.message);
}

process.exit(0);
