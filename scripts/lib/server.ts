import { join } from "node:path"
import data from "../../server/pyproject.toml"

const ROOT_DIR = process.cwd();
const SERVER_PROJECT_DIR = join(ROOT_DIR, "server");

export const server_path = SERVER_PROJECT_DIR;
export const server_name = data.project.name;
export const server_version = data.project.version;

export const server_outdir = join(SERVER_PROJECT_DIR, "dist");