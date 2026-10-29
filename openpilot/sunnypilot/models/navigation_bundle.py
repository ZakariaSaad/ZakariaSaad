"""Canonical Selector 21 identity for the native corrected Bigv2 adapter.

The normal Chestnut slot owns selection. Registration paths and compatibility
verification remain loader metadata, never a competing selection preference.
"""
from openpilot.selfdrive.navigation.corrected_bigv2 import ARTIFACT_SHA256, DISPLAY_NAME, MODEL_ID

NAV_REF = 'navigation:' + MODEL_ID


def navigation_bundle() -> dict:
  # No fabricated ordinary model generation: modeld dispatches this ref to its
  # native input/action adapter before consulting ordinary generation metadata.
  from openpilot.selfdrive.navigation.download import MODEL_URL
  return {'internalName': MODEL_ID, 'displayName': DISPLAY_NAME, 'ref': NAV_REF,
          'minimumSelectorVersion': 21, 'runner': 'tinygrad', 'is20hz': True,
          'environment': 'AMD LLVM gfx1200 USB',
          'models': [{'type': 'navigation', 'artifact': {
            'fileName': 'bigv2-amd-map-fp32.sptg',
            'downloadUri': {'uri': MODEL_URL, 'sha256': ARTIFACT_SHA256}}}],
          'overrides': [{'key': 'runtime', 'value': 'native-navigation'},
                        {'key': 'folder', 'value': 'Navigation models'}]}


def is_navigation_bundle(bundle) -> bool:
  return (bundle.get('ref') if isinstance(bundle, dict) else getattr(bundle, 'ref', None)) == NAV_REF


def is_navigation_selected(params) -> bool:
  return is_navigation_bundle(params.get('ModelManager_ActiveBundleChestnut'))
