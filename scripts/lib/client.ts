import { join } from "node:path";
import data from "../../client/Cargo.toml";

const ROOT_DIR = process.cwd();
const CLIENT_PROJECT_DIR = join(ROOT_DIR, "client");

export const client_path = CLIENT_PROJECT_DIR;
export const client_name = data.package.name;
export const client_version = data.package.version;

export const client_outdir = join(
  CLIENT_PROJECT_DIR,
  "target",
  "dx",
  client_name,
);
