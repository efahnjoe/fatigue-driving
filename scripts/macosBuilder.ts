#!/usr/bin/env bun

import { spawn } from "bun";
import { mkdir } from "node:fs/promises"
import { client_path, client_outdir } from "./lib/client";
import { server_path, server_name } from "./lib/server";
import { getOutDir } from "./lib/getOutDir";
import { copy } from "./lib/copy";
import { buildLauncher } from "./buildLauncher";
import { clean } from "./clean";

const args = process.argv.slice(2);

const isRelease = args.includes("--release");

const type = isRelease ? "release" : "debug";

console.log("Building Client:", isRelease ? "Release" : "Debug");

const buildClientCmd = isRelease
  ? ["dx", "build", "--platform=macos", "--release"]
  : ["dx", "build", "--platform=macos"];

// const buildServerCmd = isRelease
//   ? ["./.venv/bin/python", "-m", "PyInstaller", "--noconfirm", "--windowed", "--strip", "--noupx", "--python-option=O", "--hidden-import", "cv2", "--hidden-import", "onnxruntime", "--name", server_name, "src/main.py"]
//   : ["./.venv/bin/python", "-m", "PyInstaller", "--noconfirm", "--debug=all", "--console", "--hidden-import", "cv2", "--hidden-import", "onnxruntime", "--name", server_name, "src/main.py"];

const buildServerCmd = isRelease
  ? ["./.venv/bin/python", "-m", "PyInstaller", "--noconfirm", `${server_name}.release.spec`]
  : ["./.venv/bin/python", "-m", "PyInstaller", "--noconfirm", `${server_name}.debug.spec`];

async function buildClient() {
  try {
    console.log("🦀 Building client in:", client_path);
    const proc = spawn({
      cmd: buildClientCmd,
      cwd: client_path,
      stdout: "inherit",
      stderr: "inherit"
    });

    const status = await proc.exited;

    if (status !== 0) {
      console.error(`❌ Client build failed with exit code ${status}`);
      process.exit(1);
    }

    const outDir = getOutDir(type, "macos", "client");

    await clean(outDir);
    await mkdir(outDir, { recursive: true });

    await copy(client_outdir, outDir);
  } catch (error) {
    console.error("Build client fail: ", error);
    process.exit(1);
  }
}

async function buildServer() {
  try {
    console.log("🐍 Building server in:", server_path);
    const proc = spawn({
      cmd: buildServerCmd,
      cwd: server_path,
      stdout: "inherit",
      stderr: "inherit"
    });

    const status = await proc.exited;
    if (status !== 0) {
      console.error(`❌ Server build failed with exit code ${status}`);
      process.exit(1);
    }

    const outDir = getOutDir(type, "macos", "server");

    await clean(outDir);
    await mkdir(outDir, { recursive: true });

    await copy(client_outdir, outDir);
  } catch (error) {
    console.error("Build server fail: ", error);
    process.exit(1);
  }
}

async function main() {
  await buildClient();
  await buildServer();
  await buildLauncher("macos");

  console.log("✅ Build completed!");
}

await main();
