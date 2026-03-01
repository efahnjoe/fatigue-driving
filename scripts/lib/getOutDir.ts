import { join } from "node:path"
import { random } from "./random"
import { config_outdir } from "./config"

/**
 * 
 * @param type `release` \ `debug`
 * @param os `linux` \ `windows` \ `macos`
 * @param mode `client` \ `server`
 * @returns 
 */
export function getOutDir(type: string, os: string, mode: string): string {
  // const out = join(process.cwd(), config_outdir, mode, os, version, `${name}-${random()}`)

  return join(process.cwd(), config_outdir, type, os, mode)
}