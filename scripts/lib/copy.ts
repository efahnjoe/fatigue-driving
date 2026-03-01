import { cp } from "fs/promises";

export async function copy(src: string, dest: string) {
  try {
    await cp(src, dest, { recursive: true });
    console.log(`💾 Copy ${src} to ${dest} successful.`);
  } catch (error) {
    throw new Error(`Copy ${src} to ${dest} failed.`);
  }
}
