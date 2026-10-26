"""Supported navigation identity; retired bytes are never an activation fallback."""

MODEL_ID = 'big-nav-v2-map-fp32'
RETIRED_MODEL_ID = 'big-nav-v2'
RETIRED_ARTIFACT_SHA256 = 'feb2532e2fcd65375ca61d82b1ce16b397244683cb2b183aead316f26454feb4'
RETIRED_REASON = 'Original Bigv2 is retired because of its map-overflow defect; select the corrected FP32 Bigv2 or an ordinary model'


def is_retired(identifier=None, artifact_sha256=None):
  return identifier == RETIRED_MODEL_ID or artifact_sha256 == RETIRED_ARTIFACT_SHA256


def is_retired_bundle(bundle):
  """Recognize legacy saved bundles and catalog entries without reading files."""
  if not isinstance(bundle, dict):
    return False
  if any(bundle.get(key) in (RETIRED_MODEL_ID, 'navigation:' + RETIRED_MODEL_ID)
         for key in ('id', 'ref', 'navigation_model_id')) or is_retired(artifact_sha256=bundle.get('artifact_sha256')):
    return True
  artifacts = [bundle.get('compiled_artifact')]
  models = bundle.get('models', [])
  if isinstance(models, list):
    artifacts.extend(model.get('artifact') for model in models if isinstance(model, dict))
  for artifact in artifacts:
    if not isinstance(artifact, dict):
      continue
    if is_retired(artifact_sha256=artifact.get('sha256')):
      return True
    for key in ('downloadUri', 'download_uri'):
      uri = artifact.get(key)
      if isinstance(uri, dict) and is_retired(artifact_sha256=uri.get('sha256')):
        return True
  return False
