import { file, write, semver, TOML } from "bun";
import { join } from "path";

const CONFIG_JSON_PATH = join(process.cwd(), "config.json");
const CLIENT_TOML_PATH = join(process.cwd(), "client", "Cargo.toml");
const SERVER_TOML_PATH = join(process.cwd(), "server", "pyproject.toml");

function checkNewVersion(newVersion: string, oldVersion: string): void {
  if (semver.order(newVersion, oldVersion) === -1) {
    throw new Error(
      `New version ${newVersion} is lower than old version ${oldVersion}`,
    );
  }
}

export async function updateConfigVersion(newVersion: string): Promise<void> {
  const content = await file(CONFIG_JSON_PATH).text();
  const parsed = JSON.parse(content) as { version: string };
  checkNewVersion(newVersion, parsed.version);

  parsed.version = newVersion;
  const updated = JSON.stringify(parsed, null, 2);

  await write(CONFIG_JSON_PATH, updated + "\n");

  console.log(`Update config version in: ${CONFIG_JSON_PATH}`);
}

export async function updateClientVersion(newVersion: string): Promise<void> {
  const content = await file(CLIENT_TOML_PATH).text();
  const parsed = TOML.parse(content) as { package: { version: string } };
  checkNewVersion(newVersion, parsed.package.version);

  const updated = content.replace(
    /^version = ".*"$/m,
    `version = "${newVersion}"`,
  );
  await write(CLIENT_TOML_PATH, updated);

  console.log(`Update client version in: ${CLIENT_TOML_PATH}`);
}

export async function updateServerVersion(newVersion: string): Promise<void> {
  const content = await file(SERVER_TOML_PATH).text();
  const parsed = TOML.parse(content) as { project: { version: string } };
  checkNewVersion(newVersion, parsed.project.version);

  const updated = content.replace(
    /^version = ".*"$/m,
    `version = "${newVersion}"`,
  );
  await write(SERVER_TOML_PATH, updated);

  console.log(`Update server version in: ${SERVER_TOML_PATH}`);
}

await updateConfigVersion("1.0.0");
// await updateClientVersion("1.0.0")
// await updateServerVersion("1.0.0")
