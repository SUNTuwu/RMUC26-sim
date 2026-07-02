# External Source Policy

`src/external/RM2026-sentry` is kept in-tree as an external source snapshot and is
prepared for future git submodule conversion.

## Default Build Behavior

To keep the simulation workspace lean, the following external domains are disabled
by default with `COLCON_IGNORE`:

- `decision/`
- `io/`
- `main_bringup/`
- `mapping/`
- `nav/`
- `sim2d/`
- `state_estimation/`
- `vision/`

This means a plain `colcon build` from the repository root only builds the
simulation-owned packages unless you explicitly remove the relevant
`COLCON_IGNORE` files.

## Enabling External Domains

Remove the corresponding `COLCON_IGNORE` file before building, for example:

```bash
rm /home/somo/dev/sentry_sim/src/external/RM2026-sentry/src/nav/COLCON_IGNORE
```

Re-add the file when you want to disable that domain again.

## Future Submodule Conversion

See `docs/submodule-setup.md` for the exact git-based migration steps.
