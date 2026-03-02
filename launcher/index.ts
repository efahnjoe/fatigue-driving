import { spawn } from "bun";
import { dirname, join } from "node:path";
import { platform } from "os";
import { client_name } from "../scripts/lib/client";
import { server_name } from "../scripts/lib/server";

const isWin = platform() === "win32";
const appRoot = dirname(process.execPath);

const clientExe = isWin ? `${client_name}.exe` : client_name;
const serverExe = isWin ? `${server_name}.exe` : server_name;

const clientPath = join(appRoot, "client", "app", clientExe);
const serverPath = join(appRoot, "server", serverExe);

const serverProc = spawn({
  cmd: [serverPath],
  ipc({ message }) {
    console.log("[Server]", message.trim());
    if (message.includes("ready")) {
      console.log("Server is ready");
    }
  },
  cwd: join(appRoot, "server"),
  stdio: ["inherit", "inherit", "inherit"],
});

const clientProc = spawn({
  cmd: [clientPath],
  cwd: join(appRoot, "client"),
  ipc({ message }) {
    console.log("[Client]", message.trim());
    if (message.includes("ready")) {
      console.log("Client is ready");
    }
  },
  stdio: ["inherit", "inherit", "inherit"],
});

const [serverStatus, clientStatus] = await Promise.all([
  serverProc.exited,
  clientProc.exited,
]);

if (clientStatus !== 0 || serverStatus !== 0) {
  console.error(
    `App run failed with exit code: client=${clientStatus}, server=${serverStatus}`,
  );
  process.exit(1);
}

function shutdown() {
  console.log("\nClosing...");
  serverProc.kill();
  clientProc.kill();
  process.exit(0);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

Promise.race([serverProc.exited, clientProc.exited]).then(() => {
  shutdown();
});
