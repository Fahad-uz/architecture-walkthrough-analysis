# Editor warning milestone progress

Status: actionable 2D review warnings implemented and ready for root review/publication.

Owned files: `frontend/src/pages/EditorPage.tsx`, `frontend/src/editorHistory.ts`, new warning helper/tests, and this handoff note. Root owns all commits and backend changes.

Plan:

1. Check draft recovery with real saved models and correct any confirmed recovery or stale-response bug separately.
2. Build stable warning identities from source issue data, locate explicit element IDs or source-image positions, and retain whole-plan issues without fabricated locations.
3. Add focus/highlight, edit guidance, and explicit review acknowledgements. Store acknowledgement fingerprints in model metadata; geometry changes make them stale. Acknowledgement must never change quality scores, validation issues, or export gates.
4. Run focused helper tests, all frontend tests, typecheck/build, and provide exact limitations and handoff.

Backend inspection: correction saving copies general metadata, then recomputes validation and quality independently. The proposed `editor_review_acknowledgements` metadata can persist without a backend change. It will contain only issue IDs, fingerprints, and timestamps, not quality overrides.

No new downloads. Five-hour allowance checked at stage start: 94% remaining.

Inspection findings:

- Both the reviewed and automatic saved real-plan models pass the draft parser unchanged.
- Save responses already have session/revision guards. Generation polling now has the same session guard for navigating away and back to the same job.
- Structured `element_id` exists on validation issues. Gap and ambiguous-opening warnings currently provide wall IDs only inside stable backend-authored text; the frontend will recognize only those exact formats and verify every referenced wall exists. Other unlocated issues will stay whole-plan findings.
- Confirmation is an acknowledgement, not resolution. Its saved fingerprint will cover the relevant objects plus geometry dependencies; source-position or whole-plan findings conservatively cover all layout geometry. Confirmations remain visible as stale after edits, while the warning and original quality gate remain intact.

## Ready handoff

- Changed: `frontend/src/pages/EditorPage.tsx`.
- Added: `frontend/src/editorWarnings.ts`, `frontend/tests/editorWarnings.test.ts`.
- Updated: this progress note. `editorHistory.ts` did not need further changes.
- Findings focus and highlight explicit target objects or normalized source locations. Legacy gap, crossing and ambiguous-opening message formats resolve only verified existing wall IDs. Unlocated findings explicitly explain that no exact source location is available.
- Review acknowledgements persist under `metadata.editor_review_acknowledgements`, work with existing undo/redo and local drafts, and become stale when relevant geometry or scale changes. The compact fingerprint is a change detector, not an authentication digest.
- Eight new regression tests cover stable identities, target resolution, location deduplication, acknowledgement round trips, unchanged quality gates, dependency invalidation and element reordering.
- Final checks: `git diff --check`, frontend typecheck, all 29 frontend tests, and production build passed on the current files after resuming. No new downloads or commits were made by this agent.
- Browser smoke testing remains for root. This milestone is 2D only; 3D warning focus remains future work. Backend metadata persistence was inspected, but a full API/browser acknowledgement save-and-reload test has not been run.
- Conservative invalidation: wall/room/source/whole-plan findings recheck broad structural context; targeted openings track their host wall and sibling openings. Unrelated materials/camera edits do not invalidate structural review.

Next action: root reviews these exact files, performs any needed UI smoke test, and publishes the milestone as a separate commit. No other feature work is in progress in this agent's owned files.

Root browser verification passed: selecting the wall-junction warning zoomed from30% to91%, selected wallw001, and exposed correction guidance. Reviewed acknowledgement survived Save changes and page refresh; quality remained review_required39%. Used fresh testjob6e00fb89eafc435087f4b43c65296840, not the reviewed reference model.
