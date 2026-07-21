import fs from "fs";

const AG = "/Users/duyot/Development/Projects/chatbot/.understand-anything/intermediate/assembled-graph.json";
const OUT = "/Users/duyot/Development/Projects/chatbot/.understand-anything/tmp";

const g = JSON.parse(fs.readFileSync(AG, "utf8"));
const fileLevelTypes = new Set(["file", "config", "document", "service", "pipeline", "table", "schema", "resource", "endpoint"]);

const fileNodes = g.nodes
  .filter((n) => fileLevelTypes.has(n.type))
  .map((n) => ({ id: n.id, type: n.type, name: n.name, filePath: n.filePath, summary: n.summary, tags: n.tags }));

fs.writeFileSync(`${OUT}/arch-file-nodes.json`, JSON.stringify(fileNodes));

const importEdges = g.edges.filter((e) => e.type === "imports");
fs.writeFileSync(`${OUT}/arch-import-edges.json`, JSON.stringify(importEdges));

fs.writeFileSync(`${OUT}/arch-all-edges.json`, JSON.stringify(g.edges));

console.log("fileNodes:", fileNodes.length);
console.log("importEdges:", importEdges.length);
console.log("allEdges:", g.edges.length);
console.log("allNodes:", g.nodes.length);
