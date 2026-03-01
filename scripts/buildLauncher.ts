import { spawn } from "bun";
import { config_path, config_resources } from "./lib/config";

const args = process.argv.slice(2);

const isRelease = args.includes("--release");

const type = isRelease ? "release" : "debug";

console.log("Building Client:", isRelease ? "Release" : "Debug");

function buildLauncherCmd(os: string) {
  return isRelease
    ? ["bun", "build", "./launcher/index.ts", "--compile", "--outfile", `${config_resources}/${type}/${os}/launcher`]
    : ["bun", "build", "./launcher/index.ts", "--compile", "--outfile", `${config_resources}/${type}/${os}/launcher`];
}

export async function buildLauncher(os: string) {
  try {
    console.log("🚀 Building launcher");
    const proc = spawn({
      cmd: buildLauncherCmd(os),
      cwd: config_path,
      stdout: "inherit",
      stderr: "inherit"
    });

    const status = await proc.exited;
    if (status !== 0) {
      console.error(`❌ Launcher build failed with exit code ${status}`);
    }
  } catch (error) {
    console.error("Build launcher fail: ", error);
  }
}