export interface VersionChange<TOld = unknown, TNew = unknown> {
  oldVersion: string;
  newVersion: string;
  resource: string;
  /**
   * Project the NEW shape that the handler returned into the OLD shape that
   * a client pinned to `oldVersion` expects. Walked by `applyResponseChanges`
   * inside `withApiVersion`.
   *
   * Phase 0 ships zero registered changes, so this field has no live
   * consumers; the wiring is in place so Phase 1's first dated transform
   * activates without code changes to the framework itself.
   */
  transformResponse?: (payload: TNew) => TOld;
}
