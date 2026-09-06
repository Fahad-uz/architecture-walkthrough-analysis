// Copy the Draco decoder next to the app so optimized (Draco-compressed)
// GLBs load without any CDN dependency.
import { cpSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const source = join(here, "..", "node_modules", "three", "examples", "jsm", "libs", "draco", "gltf");
const target = join(here, "..", "public", "draco");
mkdirSync(target, { recursive: true });
cpSync(source, target, { recursive: true });
console.log(`copied draco decoder -> ${target}`);
