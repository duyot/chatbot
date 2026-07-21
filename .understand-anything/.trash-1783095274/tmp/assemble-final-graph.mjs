import fs from "fs";

const ROOT = "/Users/duyot/Development/Projects/chatbot";
const INTER = `${ROOT}/.understand-anything/intermediate`;

const assembled = JSON.parse(fs.readFileSync(`${INTER}/assembled-graph.json`, "utf8"));
const layers = JSON.parse(fs.readFileSync(`${INTER}/layers.json`, "utf8"));
const tour = JSON.parse(fs.readFileSync(`${INTER}/tour.json`, "utf8"));
const scan = JSON.parse(fs.readFileSync(`${INTER}/scan-result.json`, "utf8"));

const graph = {
  version: "1.0.0",
  project: {
    name: "chatbot",
    languages: scan.languages || [],
    frameworks: scan.frameworks || [],
    description:
      "A full-stack agentic RAG chatbot: React 19 + Vite frontend, FastAPI backend running a LangGraph state-machine RAG pipeline (hybrid pgvector + Postgres FTS retrieval with RRF fusion, OpenRouter reranking/embeddings/chat), PostgreSQL, Celery + Redis background jobs, JWT auth, and a standalone PaddleOCR microservice.",
    analyzedAt: "2026-07-03T16:12:31.000Z",
    gitCommitHash: "22bb20519fad98e0de4bf328eaf03519ec22802d",
  },
  nodes: assembled.nodes,
  edges: assembled.edges,
  layers,
  tour,
};

fs.writeFileSync(`${INTER}/assembled-graph.json`, JSON.stringify(graph));
console.log("Final graph assembled:", graph.nodes.length, "nodes,", graph.edges.length, "edges,", graph.layers.length, "layers,", graph.tour.length, "tour steps");
