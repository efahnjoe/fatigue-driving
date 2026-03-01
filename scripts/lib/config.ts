import { join } from "node:path"
import { NAME, VERSION, RESOURCES_DIR } from "../../config";

const ROOT_DIR = process.cwd();
const PROJECT_DIR = join(ROOT_DIR);

export const config_path = PROJECT_DIR;
export const config_name = NAME;
export const config_version = VERSION;
export const config_resources = RESOURCES_DIR;
