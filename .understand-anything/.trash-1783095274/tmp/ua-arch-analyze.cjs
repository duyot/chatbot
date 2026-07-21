#!/usr/bin/env node
'use strict';

const fs = require('fs');
const path = require('path');

function fail(msg) {
  console.error('ua-arch-analyze error: ' + msg);
  process.exit(1);
}

const inputPath = process.argv[2];
const outputPath = process.argv[3];
if (!inputPath || !outputPath) fail('usage: ua-arch-analyze.js <input.json> <output.json>');

let input;
try {
  input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
} catch (e) {
  fail('failed to read/parse input: ' + e.message);
}

const fileNodes = input.fileNodes || [];
const importEdges = input.importEdges || [];
const allEdges = input.allEdges || [];

try {
  // ---------- A. Directory Grouping ----------
  const filePaths = fileNodes.map((n) => n.filePath).filter(Boolean);

  function commonPrefix(paths) {
    if (paths.length === 0) return '';
    const splitPaths = paths.map((p) => p.split('/'));
    const first = splitPaths[0];
    let prefixLen = 0;
    for (let i = 0; i < first.length - 1; i++) {
      // -1: don't count the filename segment itself as prefix
      const seg = first[i];
      if (splitPaths.every((sp) => sp[i] === seg)) {
        prefixLen = i + 1;
      } else {
        break;
      }
    }
    return prefixLen > 0 ? first.slice(0, prefixLen).join('/') + '/' : '';
  }

  const prefix = commonPrefix(filePaths);

  function extPattern(filePath) {
    const base = path.basename(filePath);
    if (/\.test\./.test(base) || /\.spec\./.test(base) || /^test_/.test(base) || /_test\.go$/.test(base)) return 'test';
    if (/\.config\./.test(base)) return 'config';
    const ext = path.extname(base).replace('.', '') || 'noext';
    return ext;
  }

  function firstDirSegment(filePath, pfx) {
    let rel = filePath;
    if (pfx && rel.startsWith(pfx)) rel = rel.slice(pfx.length);
    const parts = rel.split('/');
    if (parts.length > 1) return parts[0];
    return null; // flat - no subdirectory
  }

  // Determine if flat structure: check whether ANY file (post common-prefix) has a subdirectory
  let hasAnySubdir = false;
  for (const p of filePaths) {
    if (firstDirSegment(p, prefix)) {
      hasAnySubdir = true;
      break;
    }
  }

  const directoryGroups = {};
  const idToGroup = {};

  for (const node of fileNodes) {
    if (!node.filePath) continue;
    let group;
    if (hasAnySubdir) {
      const seg = firstDirSegment(node.filePath, prefix);
      if (seg) {
        group = seg;
      } else {
        // file directly under prefix with no subdir -> group by 'root' relative to prefix
        group = prefix ? '(root)' : (node.filePath.includes('/') ? node.filePath.split('/')[0] : '(root)');
      }
    } else {
      group = extPattern(node.filePath);
    }
    if (!directoryGroups[group]) directoryGroups[group] = [];
    directoryGroups[group].push(node.id);
    idToGroup[node.id] = group;
  }

  // ---------- B. Node Type Grouping ----------
  const nodeTypeGroups = {};
  const idToType = {};
  for (const node of fileNodes) {
    const t = node.type || 'unknown';
    if (!nodeTypeGroups[t]) nodeTypeGroups[t] = [];
    nodeTypeGroups[t].push(node.id);
    idToType[node.id] = t;
  }

  // ---------- C. Import Adjacency Matrix ----------
  const fanOut = {};
  const fanIn = {};
  const groupImportsTo = {}; // group -> Set(group)
  const groupImportedBy = {}; // group -> Set(group)

  for (const edge of importEdges) {
    fanOut[edge.source] = (fanOut[edge.source] || 0) + 1;
    fanIn[edge.target] = (fanIn[edge.target] || 0) + 1;

    const sg = idToGroup[edge.source];
    const tg = idToGroup[edge.target];
    if (sg && tg) {
      if (!groupImportsTo[sg]) groupImportsTo[sg] = new Set();
      groupImportsTo[sg].add(tg);
      if (!groupImportedBy[tg]) groupImportedBy[tg] = new Set();
      groupImportedBy[tg].add(sg);
    }
  }

  // ---------- D. Cross-Category Dependency Analysis ----------
  const crossCategoryMap = {};
  for (const edge of allEdges) {
    if (edge.type === 'imports') continue; // imports are file<->file, handled elsewhere; but still could cross type in theory
    const st = idToType[edge.source];
    const tt = idToType[edge.target];
    if (!st || !tt) continue;
    if (st === tt) continue; // only cross-category
    const key = `${st}|${tt}|${edge.type}`;
    crossCategoryMap[key] = (crossCategoryMap[key] || 0) + 1;
  }
  const crossCategoryEdges = Object.entries(crossCategoryMap).map(([key, count]) => {
    const [fromType, toType, edgeType] = key.split('|');
    return { fromType, toType, edgeType, count };
  });

  // ---------- E. Inter-Group Import Frequency ----------
  const interGroupMap = {};
  for (const edge of importEdges) {
    const sg = idToGroup[edge.source];
    const tg = idToGroup[edge.target];
    if (!sg || !tg) continue;
    if (sg === tg) continue; // handled in intra-group
    const key = `${sg}|${tg}`;
    interGroupMap[key] = (interGroupMap[key] || 0) + 1;
  }
  const interGroupImports = Object.entries(interGroupMap).map(([key, count]) => {
    const [from, to] = key.split('|');
    return { from, to, count };
  }).sort((a, b) => b.count - a.count);

  // ---------- F. Intra-Group Import Density ----------
  const intraGroupDensity = {};
  const groupTotalEdges = {};
  const groupInternalEdges = {};

  for (const edge of importEdges) {
    const sg = idToGroup[edge.source];
    const tg = idToGroup[edge.target];
    if (sg) groupTotalEdges[sg] = (groupTotalEdges[sg] || 0) + 1;
    if (tg && tg !== sg) groupTotalEdges[tg] = (groupTotalEdges[tg] || 0) + 1;
    if (sg && tg && sg === tg) {
      groupInternalEdges[sg] = (groupInternalEdges[sg] || 0) + 1;
    }
  }

  for (const group of Object.keys(directoryGroups)) {
    const internal = groupInternalEdges[group] || 0;
    const total = groupTotalEdges[group] || 0;
    intraGroupDensity[group] = {
      internalEdges: internal,
      totalEdges: total,
      density: total > 0 ? Number((internal / total).toFixed(3)) : 0,
    };
  }

  // ---------- G. Directory Pattern Matching ----------
  const dirPatternTable = [
    [['routes', 'api', 'controllers', 'endpoints', 'handlers', 'controller', 'routers', 'blueprints'], 'api'],
    [['services', 'core', 'lib', 'domain', 'logic', 'signals', 'composables', 'mailers', 'jobs', 'channels', 'internal'], 'service'],
    [['models', 'db', 'data', 'persistence', 'repository', 'entities', 'migrations', 'entity', 'sql', 'database', 'schema', 'repositories'], 'data'],
    [['components', 'views', 'pages', 'ui', 'layouts', 'screens'], 'ui'],
    [['middleware', 'plugins', 'interceptors', 'guards'], 'middleware'],
    [['utils', 'helpers', 'common', 'shared', 'tools', 'templatetags', 'pkg'], 'utility'],
    [['config', 'constants', 'env', 'settings', 'management', 'commands'], 'config'],
    [['__tests__', 'test', 'tests', 'spec', 'specs'], 'test'],
    [['types', 'interfaces', 'schemas', 'contracts', 'dtos', 'dto', 'request', 'response'], 'types'],
    [['hooks'], 'hooks'],
    [['store', 'state', 'reducers', 'actions', 'slices'], 'state'],
    [['assets', 'static', 'public'], 'assets'],
    [['cmd'], 'entry'],
    [['bin'], 'entry'],
    [['docs', 'documentation', 'wiki'], 'documentation'],
    [['deploy', 'deployment', 'infra', 'infrastructure'], 'infrastructure'],
    [['.github', '.gitlab', '.circleci'], 'ci-cd'],
    [['k8s', 'kubernetes', 'helm', 'charts'], 'infrastructure'],
    [['terraform', 'tf'], 'infrastructure'],
    [['docker'], 'infrastructure'],
    [['auth'], 'service'],
    [['api'], 'api'],
    [['scripts'], 'utility'],
    [['evals'], 'test'],
    [['alembic'], 'data'],
    [['workers'], 'service'],
    [['ocr-service'], 'service'],
    [['features_planning'], 'documentation'],
    [['uploads'], 'assets'],
    [['logs'], 'infrastructure'],
    [['styles'], 'ui'],
  ];

  function matchDirPattern(dirName) {
    const lower = dirName.toLowerCase();
    for (const [names, label] of dirPatternTable) {
      if (names.includes(lower)) return label;
    }
    return null;
  }

  const patternMatches = {};
  for (const group of Object.keys(directoryGroups)) {
    const m = matchDirPattern(group);
    if (m) patternMatches[group] = m;
  }

  // File-level pattern overrides (informational, not altering directoryGroups)
  const fileLevelPatterns = {};
  for (const node of fileNodes) {
    const fp = node.filePath || '';
    const base = path.basename(fp);
    let label = null;
    if (/\.test\.|\.spec\.|^test_.*\.py$|_test\.go$|Test\.java$|_spec\.rb$|Test\.php$|Tests\.cs$/.test(base)) label = 'test';
    else if (/\.d\.ts$/.test(base)) label = 'types';
    else if (base === 'index.ts' || base === 'index.js' || base === '__init__.py') label = 'entry';
    else if (base === 'manage.py') label = 'entry';
    else if (base === 'wsgi.py' || base === 'asgi.py') label = 'config';
    else if (base === 'main.go' && /cmd\//.test(fp)) label = 'entry';
    else if ((base === 'main.rs' || base === 'lib.rs') && /src\//.test(fp)) label = 'entry';
    else if (base === 'Application.java' || base === 'Program.cs') label = 'entry';
    else if (base === 'config.ru') label = 'entry';
    else if (['Cargo.toml', 'go.mod', 'Gemfile', 'pom.xml', 'build.gradle', 'composer.json'].includes(base)) label = 'config';
    else if (base === 'Dockerfile' || /^docker-compose\./.test(base)) label = 'infrastructure';
    else if (/\.tf$|\.tfvars$/.test(base)) label = 'infrastructure';
    else if (/^\.github\/workflows\//.test(fp) || base === '.gitlab-ci.yml' || base === 'Jenkinsfile') label = 'ci-cd';
    else if (/\.sql$/.test(base)) label = 'data';
    else if (/\.graphql$|\.gql$|\.proto$/.test(base)) label = 'types';
    else if (/\.md$|\.rst$/.test(base)) label = 'documentation';
    else if (base === 'Makefile') label = 'infrastructure';
    if (label) fileLevelPatterns[node.id] = label;
  }

  // ---------- H. Deployment Topology Detection ----------
  const infraFiles = [];
  let hasDockerfile = false, hasCompose = false, hasK8s = false, hasTerraform = false, hasCI = false;
  for (const node of fileNodes) {
    const fp = node.filePath || '';
    const base = path.basename(fp);
    if (base.startsWith('Dockerfile')) { hasDockerfile = true; infraFiles.push(fp); }
    if (/^docker-compose.*\.ya?ml$/.test(base)) { hasCompose = true; infraFiles.push(fp); }
    if (/k8s|kubernetes|helm/.test(fp.toLowerCase())) { hasK8s = true; infraFiles.push(fp); }
    if (/\.tf$|\.tfvars$|terraform/.test(fp.toLowerCase())) { hasTerraform = true; infraFiles.push(fp); }
    if (/^\.github\/workflows\//.test(fp) || base === '.gitlab-ci.yml' || base === 'Jenkinsfile' || /\.circleci/.test(fp)) { hasCI = true; infraFiles.push(fp); }
  }
  const deploymentTopology = {
    hasDockerfile,
    hasCompose,
    hasK8s,
    hasTerraform,
    hasCI,
    infraFiles: Array.from(new Set(infraFiles)),
  };

  // ---------- I. Data Pipeline Detection ----------
  const schemaFiles = [];
  const migrationFiles = [];
  const dataModelFiles = [];
  const apiHandlerFiles = [];
  for (const node of fileNodes) {
    const fp = node.filePath || '';
    const lower = fp.toLowerCase();
    if (/\.sql$/.test(lower) && !/migrations?\//.test(lower)) schemaFiles.push(fp);
    if (/\.graphql$|\.gql$|\.proto$/.test(lower)) schemaFiles.push(fp);
    if (/migrations?\//.test(lower) || /alembic\/versions\//.test(lower)) migrationFiles.push(fp);
    if (/models?\.py$|models?\//.test(lower) || (node.tags || []).includes('data-model')) dataModelFiles.push(fp);
    if (/routers?\//.test(lower) || /routes?\//.test(lower) || (node.tags || []).includes('api-handler')) apiHandlerFiles.push(fp);
  }
  const dataPipeline = {
    schemaFiles: Array.from(new Set(schemaFiles)),
    migrationFiles: Array.from(new Set(migrationFiles)),
    dataModelFiles: Array.from(new Set(dataModelFiles)),
    apiHandlerFiles: Array.from(new Set(apiHandlerFiles)),
  };

  // ---------- J. Documentation Coverage ----------
  const docFiles = fileNodes.filter((n) => n.type === 'document' || /\.md$|\.rst$/i.test(n.filePath || ''));
  const groupsWithDocsSet = new Set();
  for (const doc of docFiles) {
    const fp = (doc.filePath || '').toLowerCase();
    for (const group of Object.keys(directoryGroups)) {
      if (fp.includes('/' + group.toLowerCase() + '/') || fp.startsWith(group.toLowerCase() + '/')) {
        groupsWithDocsSet.add(group);
      }
    }
    if (path.basename(fp) === 'readme.md') {
      // root readme covers all groups minimally; not counted per-group unless nested
    }
  }
  const totalGroups = Object.keys(directoryGroups).length;
  const groupsWithDocs = groupsWithDocsSet.size;
  const undocumentedGroups = Object.keys(directoryGroups).filter((g) => !groupsWithDocsSet.has(g));
  const docCoverage = {
    groupsWithDocs,
    totalGroups,
    coverageRatio: totalGroups > 0 ? Number((groupsWithDocs / totalGroups).toFixed(2)) : 0,
    undocumentedGroups,
  };

  // ---------- K. Dependency Direction ----------
  const dependencyDirection = [];
  const seenPairs = new Set();
  for (const { from, to, count } of interGroupImports) {
    const pairKey = [from, to].sort().join('|');
    if (seenPairs.has(pairKey)) continue;
    const reverseCount = interGroupMap[`${to}|${from}`] || 0;
    if (count > reverseCount) {
      dependencyDirection.push({ dependent: from, dependsOn: to });
    } else if (reverseCount > count) {
      dependencyDirection.push({ dependent: to, dependsOn: from });
    }
    seenPairs.add(pairKey);
  }

  // ---------- File Stats ----------
  const filesPerGroup = {};
  for (const [group, ids] of Object.entries(directoryGroups)) filesPerGroup[group] = ids.length;

  const nodeTypeCounts = {};
  for (const [t, ids] of Object.entries(nodeTypeGroups)) nodeTypeCounts[t] = ids.length;

  const fileStats = {
    totalFileNodes: fileNodes.length,
    filesPerGroup,
    nodeTypeCounts,
  };

  // ---------- Assemble Output ----------
  const result = {
    scriptCompleted: true,
    commonPrefix: prefix,
    flatStructure: !hasAnySubdir,
    directoryGroups,
    nodeTypeGroups,
    crossCategoryEdges,
    interGroupImports,
    intraGroupDensity,
    patternMatches,
    fileLevelPatterns,
    deploymentTopology,
    dataPipeline,
    docCoverage,
    dependencyDirection,
    fileStats,
    fileFanIn: fanIn,
    fileFanOut: fanOut,
  };

  fs.writeFileSync(outputPath, JSON.stringify(result, null, 2));
  process.exit(0);
} catch (e) {
  fail(e.stack || e.message);
}
