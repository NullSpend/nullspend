import { ApiVersion, isCalVer } from "./api-version";
import { RegistryAssertionError } from "./errors";
import type { VersionChange } from "./version-change";

export class VersionRegistry {
  private readonly orderedVersions: ApiVersion[] = [];
  private readonly versionByValue = new Map<string, ApiVersion>();
  private readonly changesByResource = new Map<string, VersionChange[]>();

  registerVersion(value: string): ApiVersion {
    if (!isCalVer(value)) {
      throw new RegistryAssertionError(
        `registerVersion(${JSON.stringify(value)}) — not CalVer YYYY-MM-DD`,
      );
    }
    if (this.versionByValue.has(value)) {
      throw new RegistryAssertionError(
        `registerVersion(${JSON.stringify(value)}) — already registered`,
      );
    }
    const last = this.orderedVersions[this.orderedVersions.length - 1];
    if (last && value <= last.value) {
      throw new RegistryAssertionError(
        `registerVersion(${JSON.stringify(value)}) — must be after last registered ${JSON.stringify(last.value)} (out-of-order)`,
      );
    }
    const v = new ApiVersion(value, this.orderedVersions.length);
    this.orderedVersions.push(v);
    this.versionByValue.set(value, v);
    return v;
  }

  registerChange<O = unknown, N = unknown>(change: VersionChange<O, N>): void {
    if (!this.versionByValue.has(change.oldVersion)) {
      throw new RegistryAssertionError(
        `registerChange.oldVersion ${JSON.stringify(change.oldVersion)} is not registered`,
      );
    }
    if (!this.versionByValue.has(change.newVersion)) {
      throw new RegistryAssertionError(
        `registerChange.newVersion ${JSON.stringify(change.newVersion)} is not registered`,
      );
    }
    const oldV = this.versionByValue.get(change.oldVersion)!;
    const newV = this.versionByValue.get(change.newVersion)!;
    if (newV.index !== oldV.index + 1) {
      throw new RegistryAssertionError(
        `registerChange — gap between ${change.oldVersion} (index ${oldV.index}) and ${change.newVersion} (index ${newV.index}); changes must be adjacent`,
      );
    }
    const list = this.changesByResource.get(change.resource) ?? [];
    list.push(change as VersionChange);
    list.sort((a, b) => this.versionByValue.get(a.oldVersion)!.index - this.versionByValue.get(b.oldVersion)!.index);
    this.changesByResource.set(change.resource, list);
  }

  list(): readonly ApiVersion[] {
    return this.orderedVersions;
  }

  listValues(): readonly string[] {
    return this.orderedVersions.map((v) => v.value);
  }

  get(value: string): ApiVersion | null {
    return this.versionByValue.get(value) ?? null;
  }

  has(value: string): boolean {
    return this.versionByValue.has(value);
  }

  latest(): ApiVersion {
    const last = this.orderedVersions[this.orderedVersions.length - 1];
    if (!last) {
      throw new RegistryAssertionError("latest() called on empty registry");
    }
    return last;
  }

  getChanges(resource: string): readonly VersionChange[] {
    return this.changesByResource.get(resource) ?? [];
  }
}
