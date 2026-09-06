// Per-slot GLB optimization.
//
// The CLI's blanket `optimize` lossy-compresses and resizes EVERY texture —
// including baked lightmaps, whose smooth gradients band and blur under lossy
// WebP. Lightmaps ride the emissive slot in our baked GLBs, so they get
// lossless WebP at native resolution; everything else is compressed and
// resized aggressively to stay near the size target.
//
// Usage: node optimize.mjs <input.glb> <output.glb> [maxTextureSize]

import { NodeIO } from "@gltf-transform/core";
import { ALL_EXTENSIONS } from "@gltf-transform/extensions";
import { dedup, draco, prune, textureCompress } from "@gltf-transform/functions";
import draco3d from "draco3dgltf";
import sharp from "sharp";

const [, , input, output, sizeArg] = process.argv;
if (!input || !output) {
  console.error("usage: node optimize.mjs <input.glb> <output.glb> [maxTextureSize]");
  process.exit(2);
}
const maxSize = Number.parseInt(sizeArg ?? "2048", 10);

const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
  "draco3d.encoder": await draco3d.createEncoderModule(),
  "draco3d.decoder": await draco3d.createDecoderModule(),
});

const document = await io.read(input);

await document.transform(
  dedup(),
  prune(),
  // Baked lightmaps (emissive slot): lossless, never resized below bake res.
  textureCompress({ encoder: sharp, targetFormat: "webp", slots: /emissive/i, lossless: true }),
  // Everything else: lossy WebP + resize toward the size budget.
  textureCompress({
    encoder: sharp,
    targetFormat: "webp",
    slots: /^(?!.*emissive).*$/i,
    quality: 80,
    resize: [maxSize, maxSize],
  }),
  draco(),
);

await io.write(output, document);
const { statSync } = await import("node:fs");
console.log(
  `optimized ${input} -> ${output}: ${(statSync(input).size / 1e6).toFixed(2)} MB -> ${(statSync(output).size / 1e6).toFixed(2)} MB`,
);
