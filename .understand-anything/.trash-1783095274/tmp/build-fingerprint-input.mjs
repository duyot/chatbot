import fs from "fs";

const ROOT = "/Users/duyot/Development/Projects/chatbot";
const scan = JSON.parse(fs.readFileSync(`${ROOT}/.understand-anything/intermediate/scan-result.json`, "utf8"));
const files = (scan.files || scan.fileList || []).map((f) => f.path || f.filePath);

const input = {
  projectRoot: ROOT,
  sourceFilePaths: files,
  gitCommitHash: "22bb20519fad98e0de4bf328eaf03519ec22802d",
};

fs.writeFileSync(`${ROOT}/.understand-anything/intermediate/fingerprint-input.json`, JSON.stringify(input));
console.log("Wrote fingerprint-input.json with", files.length, "source file paths");
