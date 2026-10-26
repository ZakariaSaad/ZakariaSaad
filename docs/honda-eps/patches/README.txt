Prepared external SunnyLink frontend follow-up

Source: https://github.com/sunnypilot/sunnylink-frontend
Reviewed base: 7edbc84f605e36dcf1f761401b3e3065e6b64e11
Patch: sunnylink-model-selection-frontend.patch

This patch is prepared for the hosted frontend maintainers. It has NOT been
published to or deployed on sunnylink.ai. Updating the car's branch does not
deploy somebody else's website. Apply it to the reviewed frontend base with:
  git apply --check sunnylink-model-selection-frontend.patch
  git apply sunnylink-model-selection-frontend.patch

Device-only work already supports the current frontend:
- Device-provided Steering schema supplies the Insight EPS toggle/status.
- Models uses its existing cached-catalog fallback to read an ordinary catalog
  plus the truthful corrected Bigv2 local entry. This is an RPC response overlay;
  neither the real upstream catalog URL nor ordinary model slots is rewritten.
- Existing DownloadRef/default-bundle writes become guarded asynchronous requests.
- Device status distinguishes pending/preparing/selected/loading/active/failed.
- Device checks genuine off-road/disengaged status and file admission. Browser
  Force Offroad does not override device guards.

Why an external patch remains useful:
Current hosted code sends a bare ref, empty cancellation, or default {} with no
request generation. The device cannot determine whether identical bytes encode
fresh intent or a stale browser action. It fences superseded in-flight jobs, but
cannot reconstruct missing client history. This patch sends the device revision
observed by that client with the same settings write. It never fetches a new
revision merely to bless stale user intent. The device rejects mismatches.

Patch behavior:
- Refresh every key in the device-provided Models schema, including status.
- Read/cache ModelManager_SelectionRevision from real device model responses.
- Include that revision for model selection, default selection, and cancellation.
- Capture revision when NavigationModelDrivingEnabled is explicitly toggled,
  then retain it through the existing settings debounce.
- Show RPC refusal and do not optimistically label queued default/model requests
  as already selected. The actual device readback is authoritative.
- Older devices that do not expose the optional revision retain legacy behavior.

The normal general settings checkbox still represents requested state, not an
assertion that a model or controller is active. Use the adjacent device status.
Current production hosted code can briefly show optimistic model/toggle state;
strict version rejection and the model-picker acknowledgement presentation need
this frontend patch deployment. No backend API change is required by the wire
format used here (an additional ordinary settings key in the same batch).

Focused verification:
- Prettier on changed Svelte/TypeScript files.
- svelte-kit sync + svelte-check: 0 errors, 0 warnings.
- PUBLIC_LOGTO_ENDPOINT=https://example.invalid and PUBLIC_LOGTO_APP_ID=check-only
  supplied only to compile locally; no real auth credentials used.
- git diff --check passes.
- Device bridge: 25 Python tests, including actual RPC + persistent Params.

No live hosted session/device browser test was performed. Deployment requires
write/deploy permission on sunnypilot/sunnylink-frontend or its maintainer-run
hosting; permission on ZakariaSaad/ZakariaSaad does not grant that access.

Relevant upstream interfaces (at the reviewed base):
- src/routes/dashboard/models/+page.svelte: cached catalog fallback, model refs,
  ActiveBundleChestnut default writes, cancel writes, current model readback.
- src/lib/types/models.ts: display catalog permits absent generation and models [].
- src/lib/api/device.ts: getParamsMetadata schema hydration and settings RPC API.
- src/lib/stores/batchPush.svelte.ts: debounced general settings writes.
