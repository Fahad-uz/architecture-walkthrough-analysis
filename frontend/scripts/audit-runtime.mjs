import { spawnSync } from "node:child_process";

// React Router's RSC action advisory does not apply to this client-only
// HashRouter SPA. Keep the exception scoped to the exact advisory so any
// unrelated present or future runtime finding still fails CI.
const acceptedAdvisories = new Map([
  [
    "https://github.com/advisories/GHSA-qwww-vcr4-c8h2",
    "The application uses a client-only HashRouter and has no RSC server actions.",
  ],
]);

const npmExecutable = process.env.npm_execpath;
const command = npmExecutable ? process.execPath : process.platform === "win32" ? "npm.cmd" : "npm";
const args = npmExecutable
  ? [npmExecutable, "audit", "--omit=dev", "--json"]
  : ["audit", "--omit=dev", "--json"];
const result = spawnSync(command, args, {
  encoding: "utf8",
  shell: !npmExecutable && process.platform === "win32",
});

if (result.error) {
  console.error(`Unable to run npm audit: ${result.error.message}`);
  process.exit(1);
}

let report;
try {
  report = JSON.parse(result.stdout);
} catch {
  console.error(result.stdout || result.stderr || "npm audit returned invalid JSON.");
  process.exit(1);
}

const vulnerabilities = report.vulnerabilities ?? {};

function rootAdvisories(name, seen = new Set()) {
  if (seen.has(name)) {
    return [];
  }
  const nextSeen = new Set(seen).add(name);
  const vulnerability = vulnerabilities[name];
  if (!vulnerability) {
    return [];
  }

  return (vulnerability.via ?? []).flatMap((via) =>
    typeof via === "string" ? rootAdvisories(via, nextSeen) : [via],
  );
}

const unaccepted = Object.keys(vulnerabilities).filter((name) => {
  const advisories = rootAdvisories(name);
  return (
    advisories.length === 0 ||
    advisories.some((advisory) => !acceptedAdvisories.has(advisory.url))
  );
});

if (unaccepted.length > 0) {
  console.error(result.stdout);
  console.error(`Unaccepted runtime vulnerabilities: ${unaccepted.join(", ")}`);
  process.exit(1);
}

for (const [url, reason] of acceptedAdvisories) {
  const present = Object.keys(vulnerabilities).some((name) =>
    rootAdvisories(name).some((advisory) => advisory.url === url),
  );
  if (present) {
    console.log(`Accepted non-applicable advisory ${url}: ${reason}`);
  }
}

console.log("No applicable runtime vulnerabilities found.");
