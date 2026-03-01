#!/usr/bin/env bun

import { file, fetch } from "bun";
import { join, extname } from "node:path";
import { exists, unlink, mkdir } from "node:fs/promises";

const ROOT_DIR = process.cwd();
const OUTPUT_DIR = join(ROOT_DIR, "server", "public", "models");
const MANIFEST_PATH = join(OUTPUT_DIR, ".manifest.json");
const MANIFEST_VERSION = "1.0.0";

const MIN_FILE_SIZE = 100 * 1024;
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 5000;
const HF_BASE = "https://huggingface.co";

interface ModelConfig {
  hfRepo: string;
  filePathInRepo: string;
  filename: string;
  fallbackGithubUrl: string;
  expectedSha256: string;
}

interface ManifestModel {
  name: string;
  md5: string;
  size_mb: number;
  format: string;
  updated_at: string;
}

interface Manifest {
  version: string;
  updated_at: string;
  models: ManifestModel[];
}

type ModelResult =
  | { status: "downloaded" | "skipped"; entry: ManifestModel }
  | { status: "failed" };

const MODELS: ModelConfig[] = [
  {
    hfRepo: "opencv/face_detection_yunet",
    filePathInRepo: "face_detection_yunet_2023mar.onnx",
    filename: "face_detection_yunet_2023mar.onnx",
    fallbackGithubUrl:
      "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    expectedSha256:
      "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
  },
  {
    hfRepo: "opencv/face_recognition_sface",
    filePathInRepo: "face_recognition_sface_2021dec.onnx",
    filename: "face_recognition_sface_2021dec.onnx",
    fallbackGithubUrl:
      "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    expectedSha256:
      "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
  },
];

const sleep = (ms: number) => new Promise<void>(r => setTimeout(r, ms));

const cleanupFile = async (p: string) => unlink(p).catch(() => { });

const hash = (algorithm: "sha256" | "md5", buf: ArrayBuffer): string => {
  const hasher = new Bun.CryptoHasher(algorithm);
  hasher.update(buf);
  return hasher.digest("hex");
};

const sha256 = (buf: ArrayBuffer) => hash("sha256", buf);
const md5 = (buf: ArrayBuffer) => hash("md5", buf);

const sizeMb = (bytes: number) => parseFloat((bytes / 1024 / 1024).toFixed(2));

const loadManifest = async (): Promise<Manifest | null> => {
  if (!await exists(MANIFEST_PATH)) return null;
  try {
    return JSON.parse(await file(MANIFEST_PATH).text()) as Manifest;
  } catch {
    console.warn("manifest.json exists but is malformed, will overwrite.");
    return null;
  }
};

const writeManifest = async (entries: Map<string, ManifestModel>) => {
  const manifest: Manifest = {
    version: MANIFEST_VERSION,
    updated_at: new Date().toISOString(),
    models: Array.from(entries.values()),
  };
  await file(MANIFEST_PATH).write(JSON.stringify(manifest, null, 2));
  console.log(`\nManifest written: ${MANIFEST_PATH}`);
};

const downloadFile = async (url: string, outputPath: string): Promise<{ sha256: string; md5: string }> => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);

  const buffer = await res.arrayBuffer();
  const [fileSha256, fileMd5] = [sha256(buffer), md5(buffer)];
  await file(outputPath).write(new Uint8Array(buffer));

  console.log(`Downloaded: ${url}`);
  return { sha256: fileSha256, md5: fileMd5 };
};

const downloadWithRetry = async (
  url: string,
  outputPath: string
): Promise<{ sha256: string; md5: string } | null> => {
  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    console.log(`Attempt ${attempt}/${MAX_RETRIES}: ${url}`);
    try {
      return await downloadFile(url, outputPath);
    } catch (e) {
      console.warn(`Failed: ${e}`);
      await cleanupFile(outputPath);
      if (attempt < MAX_RETRIES) {
        console.warn(`Retrying in ${RETRY_DELAY_MS / 1000}s...`);
        await sleep(RETRY_DELAY_MS);
      }
    }
  }
  return null;
};

interface VerifyResult {
  valid: boolean;
  md5?: string;
  size_mb?: number;
}

const verifyExisting = async (filePath: string, expectedHash: string): Promise<VerifyResult> => {
  const f = file(filePath);
  if (f.size < MIN_FILE_SIZE) {
    console.warn(`File too small (< ${MIN_FILE_SIZE} bytes)`);
    return { valid: false };
  }

  const buf = await f.arrayBuffer();
  const actual = sha256(buf);
  const fileMd5 = md5(buf);

  if (actual !== expectedHash) {
    console.error(`SHA256 mismatch — expected: ${expectedHash}, got: ${actual}`);
    return { valid: false };
  }

  return { valid: true, md5: fileMd5, size_mb: sizeMb(f.size) };
};

const downloadModel = async (model: ModelConfig): Promise<ModelResult> => {
  const { hfRepo, filePathInRepo, filename, fallbackGithubUrl, expectedSha256 } = model;
  const filepath = join(OUTPUT_DIR, filename);
  const format = extname(filename).slice(1);

  console.log(`\n--- ${filename} ---`);

  if (await exists(filepath)) {
    console.log("File exists, verifying...");
    const result = await verifyExisting(filepath, expectedSha256);
    if (result.valid) {
      console.log("✓ Valid.");
      return {
        status: "skipped",
        entry: {
          name: filename,
          md5: result.md5!,
          size_mb: result.size_mb!,
          format,
          updated_at: new Date().toISOString(),
        },
      };
    }
    console.error("Verification failed. File kept as-is, skipping download.");
    return { status: "failed" };
  }

  const hfUrl = `${HF_BASE}/${hfRepo}/resolve/main/${filePathInRepo}`;
  let hashes = await downloadWithRetry(hfUrl, filepath);

  if (!hashes) {
    console.warn("HF failed, trying GitHub fallback...");
    hashes = await downloadWithRetry(fallbackGithubUrl, filepath);
  }

  if (!hashes) {
    console.error(`All sources failed: ${filename}`);
    return { status: "failed" };
  }

  if (hashes.sha256 !== expectedSha256) {
    console.error(`SHA256 mismatch for ${filename}`);
    await cleanupFile(filepath);
    return { status: "failed" };
  }

  console.log(`✓ Verified: ${filename}`);
  return {
    status: "downloaded",
    entry: {
      name: filename,
      md5: hashes.md5,
      size_mb: sizeMb(file(filepath).size),
      format,
      updated_at: new Date().toISOString(),
    },
  };
};

export async function downloadModels() {
  console.log("=== Model Download & Verify ===");
  console.log(`Time: ${new Date().toISOString()}`);
  console.log(`Output: ${OUTPUT_DIR}\n`);

  await mkdir(OUTPUT_DIR, { recursive: true });

  // 加载已有 manifest 作为基准，失败的模型保留旧条目
  const existingManifest = await loadManifest();
  const entries = new Map(
    existingManifest?.models.map(m => [m.name, m]) ?? []
  );

  const results = await Promise.allSettled(
    MODELS.map(m => downloadModel(m).catch(() => ({ status: "failed" as const })))
  );

  const counts = { downloaded: 0, skipped: 0, failed: 0 };

  for (const r of results) {
    const result = r.status === "fulfilled" ? r.value : { status: "failed" as const };
    counts[result.status]++;
    // 校验通过：upsert；失败：保留旧条目（如有）
    if (result.status !== "failed") {
      entries.set(result.entry.name, result.entry);
    }
  }

  await writeManifest(entries);

  console.log("=== Summary ===");
  console.log(`Downloaded: ${counts.downloaded}`);
  console.log(`Skipped:    ${counts.skipped}`);
  console.log(`Failed:     ${counts.failed}`);

  if (counts.failed > 0) {
    console.error(`${counts.failed} model(s) failed.`);
    process.exit(1);
  }
  console.log("All models OK!");
}
