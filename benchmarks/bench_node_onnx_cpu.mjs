#!/usr/bin/env node
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import os from "node:os";
import { performance } from "node:perf_hooks";

const require = createRequire(import.meta.url);
const ort = require("onnxruntime-node");

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function argValue(name, fallback) {
  const index = process.argv.indexOf(name);
  return index >= 0 && index + 1 < process.argv.length ? process.argv[index + 1] : fallback;
}

function parseNpyF32(buffer) {
  const magic = buffer.subarray(0, 6).toString("binary");
  if (magic !== "\x93NUMPY") throw new Error("Expected .npy input");
  const major = buffer[6];
  const headerLen = major === 1 ? buffer.readUInt16LE(8) : buffer.readUInt32LE(8);
  const headerStart = major === 1 ? 10 : 12;
  const header = buffer.subarray(headerStart, headerStart + headerLen).toString("latin1");
  if (!header.includes("'descr': '<f4'") && !header.includes('"descr": "<f4"')) {
    throw new Error(`Expected little-endian float32 .npy, got header ${header}`);
  }
  const shapeMatch = header.match(/\(([^)]*)\)/);
  if (!shapeMatch) throw new Error(`Could not parse .npy shape from ${header}`);
  const shape = shapeMatch[1].split(",").map((part) => part.trim()).filter(Boolean).map(Number);
  const byteOffset = headerStart + headerLen;
  const bytes = buffer.subarray(byteOffset);
  const data = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  return { shape, data: new Float32Array(data) };
}

function summarize(samples) {
  if (samples.length === 0) {
    return {
      count: 0,
      min_ms: null,
      median_ms: null,
      max_ms: null,
    };
  }
  const ordered = [...samples].sort((a, b) => a - b);
  return {
    count: samples.length,
    min_ms: ordered[0],
    median_ms: ordered[Math.floor(ordered.length / 2)],
    max_ms: ordered[ordered.length - 1],
  };
}

function topK(logits, count = 5) {
  return Array.from(logits)
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value)
    .slice(0, count);
}

const modelPath = resolve(root, argValue("--onnx", "benchmarks/artifacts/resnet50_imagenet.onnx"));
const inputPath = resolve(root, argValue("--input-npy", "benchmarks/artifacts/resnet50_input.npy"));
const labelsPath = resolve(root, argValue("--labels", "network_data/resnet50/imagenet_classes.json"));
const runs = Number(argValue("--runs", "20"));
const warmup = Number(argValue("--warmup", "5"));
const resultPath = resolve(root, "benchmarks/results/nodejs_cpu.json");

const input = parseNpyF32(await readFile(inputPath));
let labels = [];
try {
  labels = JSON.parse(await readFile(labelsPath, "utf8"));
} catch {
  labels = [];
}
const session = await ort.InferenceSession.create(modelPath, { executionProviders: ["cpu"] });
const inputName = session.inputNames[0];
const outputName = session.outputNames[0];
const feeds = { [inputName]: new ort.Tensor("float32", input.data, input.shape) };

const warmupTimings = [];
for (let i = 0; i < warmup; i += 1) {
  const start = performance.now();
  await session.run(feeds);
  warmupTimings.push(performance.now() - start);
}

const timings = [];
let output;
for (let i = 0; i < runs; i += 1) {
  const start = performance.now();
  output = await session.run(feeds);
  timings.push(performance.now() - start);
}

const logits = output[outputName].data;
const top = topK(logits);
const prediction = top[0].index;
const payload = {
  name: "nodejs_cpu",
  created_unix: Date.now() / 1000,
  backend: "onnxruntime-node:cpu",
  warmup_summary: summarize(warmupTimings),
  summary: summarize(timings),
  prediction,
  class_label: labels[prediction] || "",
  top5: top.map((item) => `${item.index} ${labels[item.index] || ""} ${item.value.toPrecision(6)}`.trim()).join("\n"),
  node: process.version,
  onnxruntime_node: ort.version || "unknown",
  system: {
    platform: `${os.type()} ${os.release()} ${os.arch()}`,
    processor: os.cpus()[0]?.model || "",
  },
};

await mkdir(dirname(resultPath), { recursive: true });
await writeFile(resultPath, `${JSON.stringify(payload, null, 2)}\n`);
console.log(resultPath);
console.log(`nodejs_cpu: median=${payload.summary.median_ms.toFixed(3)} ms prediction=${prediction}`);
