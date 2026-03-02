import { rimraf } from "rimraf";
import { resolve } from "node:path";

export async function clean(path: string): Promise<boolean> {
  try {
    console.log(`🧹 Clean last time output in: ${path}`);

    return await rimraf(resolve(path), { preserveRoot: true });
  } catch (error) {
    throw error;
  }
}
