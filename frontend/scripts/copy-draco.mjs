// Copy the Draco decoder next to the app so optimized (Draco-compressed)
// GLBs load without any CDN dependency.
import { chmodSync, cpSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, "..", "node_modules", "three", "examples", "jsm", "libs", "draco", "gltf");
const target = join(here, "..", "public", "draco");
mkdirSync(target, { recursive: true });
cpSync(source, target, { recursive: true });
// npm can mark the encoder executable on POSIX; these are served assets,
// so keep generated file modes deterministic across developer machines.
for (const name of readdirSync(target)) {
  if (/\.(js|wasm)$/.test(name)) chmodSync(join(target, name), 0o644);
}
console.log(`copied draco decoder -> ${target}`);
