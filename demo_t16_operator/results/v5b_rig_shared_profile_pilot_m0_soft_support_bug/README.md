# Archived M0 output: soft-support boolean bug

This directory preserves the first v5b development output for audit only. It
must not be used as the current pilot result.

The runner converted the floating reaction support with `astype(bool)`. Every
one of the 320 voxels was strictly positive, so all 320 voxels were active even
though the run was described as support restricted. Four inner cameras provide
160 measurements per field. The field nuisance was therefore underdetermined
before ridge regularization and could absorb optical-parameter changes.

The archived numerical facts remain useful as a failure trace, but all claims
that depend on a restricted support are invalid. The corrected runner requires
an explicit support threshold, records active/total voxel counts, and rejects an
empty or full support mask.

Evidence label: `ARCHIVED_DEVELOPMENT_OUTPUT_WITH_KNOWN_SUPPORT_BUG`.
