# Editor warning milestone progress

Status: draft inspection complete; implementing actionable 2D review warnings.

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
- Save responses already have session/revision guards. Generation polling still needs the same session guard for navigating away and back to the same job.
- Structured `element_id` exists on validation issues. Gap and ambiguous-opening warnings currently provide wall IDs only inside stable backend-authored text; the frontend will recognize only those exact formats and verify every referenced wall exists. Other unlocated issues will stay whole-plan findings.
- Confirmation is an acknowledgement, not resolution. Its saved fingerprint will cover the relevant objects plus geometry dependencies; source-position or whole-plan findings conservatively cover all layout geometry. Confirmations remain visible as stale after edits, while the warning and original quality gate remain intact.
