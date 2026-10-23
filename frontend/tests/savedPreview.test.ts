import assert from "node:assert/strict";
import test from "node:test";
import { readSavedPreview, type SavedPreview } from "../src/savedPreview.ts";
import type { FloorPlanModel, JobRecord } from "../src/types.ts";

function job(version = 1, updates: Partial<JobRecord> = {}): JobRecord {
  return { job_id: "job-a", status: "needs_review", message: "Saved preview ready",
    glb_url: "/jobs/job-a/artifacts/building.glb", glb_source: "preview", glb_version: version, ...updates };
}

function plan(label: string): FloorPlanModel {
  return { coordinate_system: "metres", schema_version: "3.0", walls: [], doors: [], windows: [], rooms: [],
    metadata: { label }, validation_issues: [] };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((complete) => { resolve = complete; });
  return { promise, resolve };
}

test("initial preview publishes its plan only after a matching second status read", async () => {
  const first = job();
  const confirmed = job(1, { message: "Confirmed status" });
  const savedPlan = plan("saved");
  const verification = deferred<JobRecord>();
  const calls: string[] = [];
  let complete = false;
  const pending = readSavedPreview(null, {
    getJob: async () => {
      calls.push("job");
      return calls.length === 1 ? first : verification.promise;
    },
    getPlan: async () => { calls.push("plan"); return { model: savedPlan }; },
  }).then((result) => { complete = true; return result; });
  await new Promise<void>((resolve) => setImmediate(resolve));
  assert.deepEqual(calls, ["job", "plan", "job"]);
  assert.equal(complete, false, "the unverified plan must not be published");
  verification.resolve(confirmed);
  const result = await pending;
  assert.equal(result.snapshot!.plan, savedPlan);
  assert.equal(result.snapshot!.job, confirmed);
  assert.equal(result.latestJob, confirmed);
});

test("a concurrent completed save retains the previous pair and retries on the next poll", async () => {
  const previous: SavedPreview = { job: job(1), plan: plan("previous") };
  const original = structuredClone(previous);
  const fetchedPlan = deferred<{ model: FloorPlanModel }>();
  const currentPlan = plan("newest");
  let currentJob = job(2);
  let statusReads = 0;
  let planReads = 0;
  const pending = readSavedPreview(previous, {
    getJob: async () => { statusReads += 1; return currentJob; },
    getPlan: () => { planReads += 1; return fetchedPlan.promise; },
  });
  await Promise.resolve();
  currentJob = job(3);
  fetchedPlan.resolve({ model: currentPlan });
  const changed = await pending;
  assert.equal(changed.snapshot, previous, "a plan from revision 3 must not accompany GLB revision 2");
  assert.equal(changed.latestJob.glb_version, 3);
  assert.equal(statusReads, 2);
  assert.equal(planReads, 1, "a changed revision is deferred without an immediate retry loop");
  assert.deepEqual(previous, original);

  const next = await readSavedPreview(changed.snapshot, {
    getJob: async () => currentJob,
    getPlan: async () => ({ model: currentPlan }),
  });
  assert.equal(next.snapshot!.job.glb_version, 3);
  assert.equal(next.snapshot!.plan, currentPlan);
});

test("a save or analysis beginning during the plan read prevents publication even before its version changes", async () => {
  for (const status of ["saving_corrections", "processing"]) {
    const previous: SavedPreview = { job: job(1), plan: plan("previous") };
    let reads = 0;
    const result = await readSavedPreview(previous, {
      getJob: async () => job(2, { status: ++reads === 1 ? "needs_review" : status }),
      getPlan: async () => ({ model: plan("being replaced") }),
    });
    assert.equal(result.snapshot, previous, status);
    assert.equal(result.latestJob.status, status);
    assert.equal(reads, 2);
  }
});

test("active plan writes leave the previous pair intact and do not request edit data", async () => {
  for (const status of ["saving_corrections", "processing"]) {
    for (const previous of [null, { job: job(1), plan: plan("previous") }]) {
      let reads = 0;
      const result = await readSavedPreview(previous, {
        getJob: async () => { reads += 1; return job(2, { status }); },
        getPlan: async () => { assert.fail("an active plan write cannot supply verified edit data"); },
      });
      assert.equal(result.snapshot, previous);
      assert.equal(result.latestJob.status, status);
      assert.equal(reads, 1, "wait for the caller's next poll");
    }
  }
});

test("unchanged GLB polls update status while retaining plan identity and camera focus", async () => {
  const previous: SavedPreview = { job: job(), plan: plan("same saved plan") };
  let reads = 0;
  const result = await readSavedPreview(previous, {
    getJob: async () => { reads += 1; return job(1, { status: "generating", message: "Blender build running" }); },
    getPlan: async () => { assert.fail("an unchanged GLB needs no plan refresh"); },
  });
  assert.equal(result.snapshot!.plan, previous.plan);
  assert.equal(result.snapshot!.job.status, "generating");
  assert.equal(reads, 1);
});

test("a missing saved GLB is represented explicitly without requesting a plan", async () => {
  let reads = 0;
  const result = await readSavedPreview(null, {
    getJob: async () => { reads += 1; return job(0, { glb_url: null, status: "failed" }); },
    getPlan: async () => { assert.fail("an unavailable model has no matching plan to load"); },
  });
  assert.equal(result.snapshot!.job.glb_url, null);
  assert.equal(result.snapshot!.plan, null);
  assert.equal(result.latestJob.status, "failed");
  assert.equal(reads, 1);
});

test("changed or removed artifact URLs during verification do not receive the fetched plan", async () => {
  const previous: SavedPreview = { job: job(1), plan: plan("previous") };
  for (const glb_url of [null, "/jobs/job-a/artifacts/replacement.glb"]) {
    let reads = 0;
    const result = await readSavedPreview(previous, {
      getJob: async () => ++reads === 1 ? job(2) : job(2, { glb_url }),
      getPlan: async () => ({ model: plan("unverified") }),
    });
    assert.equal(result.snapshot, previous);
    assert.equal(result.latestJob.glb_url, glb_url);
    assert.equal(reads, 2);
  }
});

test("transient read failures leave caller-owned saved state untouched for a later poll", async () => {
  const previous: SavedPreview = { job: job(1), plan: plan("previous") };
  const original = structuredClone(previous);
  for (const failure of ["initial status", "plan", "verification"]) {
    let reads = 0;
    await assert.rejects(readSavedPreview(previous, {
      getJob: async () => {
        reads += 1;
        if (failure === "initial status" || (failure === "verification" && reads === 2)) throw new Error(failure);
        return job(2);
      },
      getPlan: async () => {
        if (failure === "plan") throw new Error(failure);
        return { model: plan("new") };
      },
    }), new RegExp(failure));
    assert.deepEqual(previous, original);
  }
});
